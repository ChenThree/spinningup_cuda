import itertools
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
from .noise import OrnsteinUhlenbeckActionNoise


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim),
                                 dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        # np.random.choice slow when not replace
        idxs = random.sample(range(0, self.size), batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in batch.items()
        }


def test_td3_pytorch(env_fn,
                     resume,
                     actor_critic=MLPActorCritic,
                     ac_kwargs=dict()):
    # prepare test env
    env = env_fn()
    act_limit = env.action_space.high[0]
    # load checkpoint
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    state_dict = torch.load(resume)['state_dict']
    ac.load_state_dict(state_dict)
    ac.eval()

    # test
    def get_action(o):
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).cuda())
        return np.clip(a, -act_limit, act_limit)

    with torch.no_grad():
        o, d, ep_ret, ep_len = env.reset(), False, 0, 0
        while not d:
            # Take deterministic actions at test time (noise_scale=0)
            env.render()
            o, r, d, info = env.step(get_action(o))
            ep_ret += r
            ep_len += 1
        # success rate for robel
        print(
            f'ep_len == {ep_len}  result == {info["score/success"]}  reward == {ep_ret}',
        )


def td3(env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=100,
        eps_decay=10000,
        warmup=1000,
        update_every=50,
        target_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        num_test_episodes=10,
        max_ep_len=1000,
        loss_criterion=nn.SmoothL1Loss,
        logger_kwargs=dict(),
        save_freq=10,
        log_success=False):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        eps_decay (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        warmup (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # change warmup to several epoch, avoid log error
    if warmup % steps_per_epoch != 0:
        warmup = (warmup // steps_per_epoch + 1) * steps_per_epoch

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]
    act_noise = 0.99
    eps_decay = 1 / eps_decay
    noise_process = OrnsteinUhlenbeckActionNoise(mu=np.zeros((act_dim, )),
                                                 sigma=0.2,
                                                 theta=0.15)

    # Create actor-critic module and target networks
    criterion = loss_criterion()
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Sync params across processes
    sync_params(ac)
    sync_params(ac_targ)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' %
               var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'].cuda(), data['act'].cuda(
        ), data['rew'].cuda(), data['obs2'].cuda(), data['done'].cuda()

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        # loss_q1 = ((q1 - backup).square()).mean()
        # loss_q2 = ((q2 - backup).square()).mean()
        loss_q1 = criterion(q1, backup)
        loss_q2 = criterion(q2, backup)
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs'].cuda()
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        # average grads across MPI processes
        mpi_avg_grads(ac.q1)
        mpi_avg_grads(ac.q2)
        # clamp grad
        for param in q_params:
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            # average grads across MPI processes
            mpi_avg_grads(ac.pi)
            # clamp grad
            for param in ac.pi.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale, noise_process: OrnsteinUhlenbeckActionNoise):
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).cuda())
        a += noise_scale * noise_process()
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        ac.eval()
        with torch.no_grad():
            for j in range(num_test_episodes):
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
                noise_process.reset()
                while not (d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time (noise_scale=0)
                    o, r, d, info = test_env.step(
                        get_action(o, 0, noise_process))
                    ep_ret += r
                    ep_len += 1
                # success rate for robel
                if log_success:
                    logger.store(TestSuccess=int(info['score/success']))
                logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        ac.train()

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    noise_process.reset()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until eps_decay have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t <= warmup:
            a = env.action_space.sample()
        else:
            a = get_action(o, act_noise, noise_process)
            # eps decay
            act_noise = max(act_noise - eps_decay, 0)

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # consider max episode as done
        if ep_len == max_ep_len:
            d = True

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d:
            # success rate for robel
            if log_success:
                logger.store(Success=int(info['score/success']))
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            noise_process.reset()

        # Update handling
        if t >= warmup:
            if t % update_every == 0:
                # adjust update times according to buffer size
                k = 1 + replay_buffer.size / replay_buffer.max_size
                for j in range(int(update_every * k)):
                    batch = replay_buffer.sample_batch(int(batch_size * k))
                    update(data=batch, timer=j)
        else:
            logger.store(LossQ=0, LossPi=0, Q1Vals=0, Q2Vals=0)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            # update lr
            new_pi_lr = pi_lr * (1 - epoch / epochs)
            new_q_lr = q_lr * (1 - epoch / epochs)
            update_lr(q_optimizer, new_q_lr)
            update_lr(pi_optimizer, new_pi_lr)

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
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from ...utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3(lambda: gym.make(args.env),
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)
