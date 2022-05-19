import argparse
import os
import time
from copy import deepcopy

import gym
import numpy as np
import robel
import torch
import torch.nn as nn

from spinningup import td3_pytorch, test_td3_pytorch
from spinningup.utils.mpi_tools import mpi_fork


def args_parser():
    parser = argparse.ArgumentParser(description='TD3')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--env',
                        default='DKittyStandRandom-v0',
                        type=str,
                        help='environment name')
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--plr',
                        default=0.001,
                        type=float,
                        help='policy learning rate')
    parser.add_argument('--qlr',
                        default=0.001,
                        type=float,
                        help='Q-networks learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--batch-size',
                        default=128,
                        type=int,
                        help='minibatch size')
    parser.add_argument('--replay-size',
                        default=1000000,
                        type=int,
                        help='replay-size')
    parser.add_argument('--polyak',
                        default=0.995,
                        type=float,
                        help='moving average for target network')
    parser.add_argument(
        '--policy-delay',
        default=2,
        type=int,
        help=
        'Policy will only be updated once every policy_delay times for each update of the Q-networks.'
    )
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
                        help='linear decay of exploration policy')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--log-dir',
                        default='./logs-td3',
                        type=str,
                        help='log dir')
    return parser.parse_args()


def main():
    # read args
    args = args_parser()

    # prepare sim env
    def env_fn():
        env = gym.make(args.env)
        env.reset(seed=args.seed)
        return env

    # cuda setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # td3
    if args.mode == 'train':
        # run parallel code with mpi
        mpi_fork(args.cpu)
        td3_pytorch(env_fn,
                    ac_kwargs={
                        'hidden_sizes': (256, 256 * 4, 256),
                        'activation': nn.SiLU,
                    },
                    steps_per_epoch=args.steps_per_epoch,
                    epochs=args.epochs,
                    seed=args.seed,
                    replay_size=args.replay_size,
                    gamma=args.gamma,
                    polyak=args.polyak,
                    pi_lr=args.plr,
                    q_lr=args.qlr,
                    policy_delay=args.policy_delay,
                    batch_size=args.batch_size,
                    eps_decay=args.eps_decay,
                    warmup=args.warmup,
                    update_every=args.update_every,
                    num_test_episodes=args.validate_episodes,
                    max_ep_len=args.max_episode_length,
                    logger_kwargs={'output_dir': args.log_dir},
                    log_success=True)
    else:
        test_td3_pytorch(env_fn=env_fn,
                         resume=args.resume,
                         ac_kwargs={
                             'hidden_sizes': (256, 256 * 4, 256),
                             'activation': nn.SiLU,
                         })


if __name__ == '__main__':
    main()
