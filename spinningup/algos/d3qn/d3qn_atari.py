import math
import os
import random
import time
from copy import deepcopy

import gym
import numpy as np
import torch
from torch.optim import Adam

from ...utils.logx import EpochLogger
from ...utils.mpi_pytorch import mpi_avg_grads, setup_pytorch_for_mpi, sync_params
from ...utils.mpi_tools import mpi_avg, mpi_fork, mpi_statistics_scalar, num_procs, proc_id
from .core import *


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, opt_mem=True):
        self.opt_mem = opt_mem
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.uint8)
        if not opt_mem:
            self.obs2_buf = np.zeros(combined_shape(size, obs_dim),
                                     dtype=np.uint8)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.uint8)
        self.rew_buf = np.zeros(size, dtype=np.float16)
        self.done_buf = np.zeros(size, dtype=np.bool8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        if not self.opt_mem:
            self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        # np.random.choice slow when not replace
        idxs = random.sample(range(0, self.size), batch_size)
        if self.opt_mem:
            next_idxs = (np.array(idxs) + 1) % self.max_size
            batch = dict(obs=self.obs_buf[idxs] / 255.0,
                         obs2=self.obs_buf[next_idxs] / 255.0,
                         act=self.act_buf[idxs],
                         rew=self.rew_buf[idxs],
                         done=self.done_buf[idxs])
        else:
            batch = dict(obs=self.obs_buf[idxs] / 255.0,
                         obs2=self.obs2_buf[idxs] / 255.0,
                         act=self.act_buf[idxs],
                         rew=self.rew_buf[idxs],
                         done=self.done_buf[idxs])
        return {
            k:
            torch.as_tensor(v,
                            dtype=torch.int64 if k == 'act' else torch.float32)
            for k, v in batch.items()
        }


def d3qn(env_fn,
         dqn_model=CNNDualDoubleDQN,
         dqn_kwargs=dict(),
         seed=0,
         gamma=0.99,
         min_eps=0.1,
         eps_decay=10000,
         lr=1e-4,
         epochs=100,
         steps_per_epoch=10000,
         replay_size=int(1e6),
         batch_size=128,
         target_update_interval=2000,
         polyak=None,
         update_every=50,
         num_test_episodes=100,
         warmup=1000,
         random_steps=10000,
         max_ep_len=1000,
         loss_criterion=nn.SmoothL1Loss,
         logger_kwargs=dict(),
         save_freq=10,
         log_success=False):
    # change warmup to several epoch, avoid log error
    if warmup % steps_per_epoch != 0:
        warmup = (warmup // steps_per_epoch + 1) * steps_per_epoch

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # setup logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # get environment setting
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    print('obs_dim ==', obs_dim)
    print('act_dim ==', act_dim)
    eps_threshold = 1 - min_eps
    eps_decay = 1 - 1 / eps_decay
    logger.store(Eps=eps_threshold + min_eps)

    # initialize optimizer
    criterion = loss_criterion()
    dqn = dqn_model(env.observation_space, env.action_space, **dqn_kwargs)
    dqn_targ = deepcopy(dqn)
    for para in dqn_targ.parameters():
        para.requires_grad = False

    # Sync params across processes
    sync_params(dqn)
    sync_params(dqn_targ)

    # set up optimizer
    optimizer = Adam(dqn.parameters(), lr=lr)

    # create replay buffer
    replay_buffer = ReplayBuffer(obs_dim, 1, replay_size, opt_mem=True)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [dqn])
    logger.log('\nNumber of parameters: \t dqn: %d\n' % var_counts)

    # Set up model saving
    logger.setup_pytorch_saver(dqn)

    def compute_loss(data):
        with torch.no_grad():
            o, a, r, o2, d = data['obs'].cuda(), data['act'].cuda(), data['rew'].cuda(), \
                data['obs2'].cuda(), data['done'].cuda()
            # get the Q values for best actions in next_obs, using the smaller one
            q_next = torch.min(*dqn(o2)).max(1)[0]
            # cal target q_s_a
            q_target = r + gamma * (1 - d) * q_next
        # get the Q values for current observations (Q(s,a, theta_i))
        q1, q2 = dqn(o)
        q1 = q1.gather(1, a).squeeze()
        q2 = q2.gather(1, a).squeeze()
        # combine loss
        loss = criterion(q1, q_target) + criterion(q2, q_target)
        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())
        return loss, loss_info

    def get_action(o, eps):
        # epsilon greedy exploration
        action = dqn.get_action(o[np.newaxis] / 255.0, eps)
        return action

    def update(data):
        optimizer.zero_grad()
        # calculate loss
        loss, loss_info = compute_loss(data)
        loss.backward()
        # average grads across MPI processes
        mpi_avg_grads(dqn)
        # clamp grad
        for param in dqn.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        optimizer.step()

        # Record things
        logger.store(Loss=loss.item(), **loss_info)

        # soft target update
        if polyak is not None:
            with torch.no_grad():
                for p, p_targ in zip(dqn.parameters(), dqn_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def test_agent():
        dqn.eval()
        with torch.no_grad():
            for _ in range(num_test_episodes):
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
                while not (d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time (noise_scale=0)
                    o, r, d, info = test_env.step(get_action(o, 0))
                    ep_ret += r
                    ep_len += 1
                # success rate
                if log_success:
                    logger.store(TestSuccess=int(info['score/success']))
                logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        dqn.train()

    # train
    total_steps = epochs * steps_per_epoch
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        # before learning starts, choose actions randomly
        if t <= random_steps:
            a = env.action_space.sample()
        else:
            # get action
            a = get_action(o, min_eps + eps_threshold)
            # decay eps
            logger.store(Eps=eps_threshold + min_eps)
            eps_threshold *= eps_decay

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # consider max episode as done
        if ep_len == max_ep_len:
            d = True

        # store experience
        replay_buffer.store(o, a, r, o2, d)
        o = o2

        # End of trajectory handling
        if d:
            # success rate for robel
            if log_success:
                logger.store(Success=int(info['score/success']))
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # update network only reach learn_interval
        if t >= warmup:
            if t % update_every == 0:
                # adjust update times according to buffer size
                k = 1 + replay_buffer.size / replay_buffer.max_size
                for _ in range(int(update_every * k)):
                    batch = replay_buffer.sample_batch(int(batch_size * k))
                    update(data=batch)
        else:
            logger.store(Loss=0, Q1Vals=0, Q2Vals=0)

        # update target Q network weights with current Q network weights
        if polyak is None and t % target_update_interval == 0:
            dqn_targ.load_state_dict(dqn.state_dict())

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            if log_success:
                logger.log_tabular('Success', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            if log_success:
                logger.log_tabular('TestSuccess', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('Eps', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
