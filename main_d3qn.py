import argparse
import os
import time

import gym
import numpy as np
import robel
import torch
import torch.nn as nn

from spinningup import d3qn_pytorch
from spinningup.utils.mpi_tools import mpi_fork

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def args_parser():
    parser = argparse.ArgumentParser(description='DDPG DKiity')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--env',
                        default='Acrobot-v1',
                        type=str,
                        help='environment name')
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--lr',
                        default=1e-4,
                        type=float,
                        help='dqn learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--batch-size',
                        default=128,
                        type=int,
                        help='minibatch size')
    parser.add_argument('--replay-size',
                        default=1000000,
                        type=int,
                        help='replay-size')
    parser.add_argument('--target-update-interval',
                        default=2000,
                        type=int,
                        help='dqn target network update rate')
    parser.add_argument(
        '--validate-episodes',
        default=100,
        type=int,
        help='how many episode to perform during validate experiment')
    parser.add_argument('--max-episode-length', default=1000, type=int, help='')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument(
        '--warmup',
        default=1000,
        type=int,
        help=
        'Number of env interactions to collect before starting to do gradient descent updates'
    )
    parser.add_argument('--random-steps',
                        default=10000,
                        type=int,
                        help='Number of random steps for boosting exploration')
    parser.add_argument(
        '--update-every',
        default=50,
        type=int,
        help=
        'Number of env interactions that should elapse between gradient descent updates'
    )
    parser.add_argument('--steps-per-epoch',
                        default=10000,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--eps-decay',
                        default=100000,
                        type=int,
                        help='eps-greedy pro decay steps')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--log-dir',
                        default='./logs-d3qn',
                        type=str,
                        help='log dir')
    return parser.parse_args()


def main():
    # read args
    args = args_parser()

    # prepare sim env
    def env_fn():
        return gym.make(args.env)

    # run parallel code with mpi
    mpi_fork(args.cpu)
    # ddpg
    d3qn_pytorch(env_fn,
                 dqn_kwargs={
                     'hidden_sizes': (128, 128 * 4, 128),
                     'activation': nn.ReLU,
                 },
                 steps_per_epoch=args.steps_per_epoch,
                 epochs=args.epochs,
                 seed=args.seed,
                 replay_size=args.replay_size,
                 gamma=args.gamma,
                 target_update_interval=args.target_update_interval,
                 lr=args.lr,
                 batch_size=args.batch_size,
                 eps_decay=args.eps_decay,
                 warmup=args.warmup,
                 random_steps=args.random_steps,
                 update_every=args.update_every,
                 num_test_episodes=args.validate_episodes,
                 max_ep_len=args.max_episode_length,
                 loss_criterion=nn.SmoothL1Loss,
                 logger_kwargs={'output_dir': args.log_dir},
                 log_success=False)


if __name__ == '__main__':
    main()
