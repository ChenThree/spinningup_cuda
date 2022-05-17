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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(env_fn,
          dqn_model=MLPDQN,
          dqn_kwargs=dict(),
          seed=0,
          gamma=0.99,
          min_eps=0.1,
          eps_decay=1e5,
          lr=3e-4,
          loss_criterion=nn.SmoothL1Loss,
          epochs=100,
          epoch_per_epoch=10000,
          replay_size=1000000,
          batch_size=128,
          target_update_interval=2000,
          update_every=10,
          warmup=5000,
          save_freq=10,
          logger_kwargs=dict()):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # setup logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    def compute_loss(data):
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
        # Compute huber error
        loss = criterion(q1, q_target) + criterion(q2, q_target)
        return loss

    # get environment setting
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    # define Q target and Q
    dqn = dqn_model(env.observation_space, env.action_space, **dqn_kwargs)
    dqn_target = dqn_model(env.observation_space, env.action_space)
    dqn_target.load_state_dict(dqn.state_dict())
    for para in dqn_target.parameters():
        para.requires_grad = False
    dqn_target.eval()
    dqn.cuda()
    dqn_target.cuda()
    # initialize optimizer
    optimizer = Adam(dqn.parameters(), lr=lr)
    criterion = loss_criterion()
    # create replay buffer
    memory = ReplayBuffer(obs_dim, 1, replay_size)
    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [dqn])
    logger.log('\nNumber of parameters: \t dqn: %d\n' % var_counts)
    # train
    start = time.time()
    sum_loss = 0
    obs, reward, done = env.reset(), 0, False
    max_iter = epochs * epoch_per_epoch
    for t in range(max_iter):
        # before learning starts, choose actions randomly
        if t < warmup:
            action = np.random.randint(act_dim)
        else:
            # decay eps
            eps_threshold = min_eps + (0.9 - min_eps) * math.exp(-t / eps_decay)
            # epsilon greedy exploration
            if random.random() > eps_threshold:
                # get action with max Q
                obs_t = torch.from_numpy(obs[np.newaxis, :]).cuda()
                action = dqn.get_action(obs_t).squeeze().cpu().numpy()
            else:
                action = np.random.randint(act_dim)
        # take action and get reward, clipping to [0, 1]
        obs2, reward, done, _ = env.step(action)
        # store effect of action
        memory.store(obs, action, reward, obs2, done)
        obs = obs2
        # reset env if reached episode boundary
        if done:
            obs, reward, done = env.reset(), 0, False
        # update network only reach learn_interval
        if t < warmup or t % update_every != 0:
            continue
        # set train mode
        dqn.train()
        update_times = int(update_every * (1 + memory.size / memory.max_size))
        for i in range(update_times):
            # sample batch from replay buffer
            data = memory.sample_batch(batch_size)
            # calculate loss
            loss = compute_loss(data)
            sum_loss += loss.data.cpu().numpy()
            # backward
            optimizer.zero_grad()
            loss.backward()
            # clamp grad
            for param in dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # update target Q network weights with current Q network weights
        if t % target_update_interval == 0:
            dqn_target.load_state_dict(dqn.state_dict())
        # log
        if t > 0 and t % epoch_per_epoch == 0:
            # write log
            log_str = 'iter == {}  time == {:.3f} s  err == {:.5f}\n'.format(
                t,
                time.time() - start, sum_loss / epoch_per_epoch)
            sum_loss = 0
            log_str += log_trainning_info(dqn, test_env, lr, 100)
            logger.log(log_str)
            start = time.time()
            # save checkpoint
            if t % (save_freq * epoch_per_epoch) == 0:
                if not os.path.exists('./checkpoint'):
                    os.makedirs('./checkpoint')
                torch.save(
                    {
                        'iter': t,
                        'state_dict': dqn.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, '{}/checkpoint_{}.pth'.format('./checkpoint', t))


def log_trainning_info(dqn, env, lr, test_count):
    # set not train mode
    dqn.eval()
    score_sum = 0
    for i in range(test_count):
        score = test(env, dqn, plot_figure=False)
        score_sum += score
    score_sum /= test_count
    log_str = 'mean score == {:.3f} lr == {:.5f}\n'.format(score_sum, lr)
    return log_str


def test(env, Q, plot_figure):
    # reset environment
    score = 0
    o, r, done = env.reset(), 0, False
    with torch.no_grad():
        while True:
            # visualization
            if plot_figure:
                cur_map = env.show()
                print(cur_map)
            # inference to get action
            a = Q.get_action(torch.from_numpy(
                o[np.newaxis, :]).cuda()).squeeze().cpu().numpy()
            # take action
            o, r, done, _ = env.step(a)
            score += r
            # terminate
            if done:
                break
    return score


def main():
    train(lambda: gym.make('Acrobot-v1'),
          logger_kwargs={'output_dir': './logs-ddpg'})


if __name__ == '__main__':
    main()
