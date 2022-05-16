import argparse
import os
import time
from copy import deepcopy

import gym
import numpy as np
import robel
import torch
import torch.nn as nn

from spinningup import sac_pytorch, test_sac_pytorch
from spinningup.utils.mpi_tools import mpi_fork


def args_parser():
    parser = argparse.ArgumentParser(description='SAC DKiity')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--env',
                        default='DKittyStandRandom-v0',
                        type=str,
                        help='environment name')
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate for both policy and value learning')
    parser.add_argument('--lr-decay',
                        default=False,
                        type=bool,
                        help='learning rate decay for every epoch')
    parser.add_argument(
        '--alpha',
        default=None,
        type=float,
        help=
        'entropy regularization coefficient, None means learn alpha automatically'
    )
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--batch-size',
                        default=1024,
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
        '--validate-episodes',
        default=100,
        type=int,
        help='how many episode to perform during validate experiment')
    parser.add_argument('--max-episode-length', default=2000, type=int, help='')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument(
        '--warmup',
        default=5000,
        type=int,
        help=
        'Number of env interactions to collect before starting to do gradient descent updates'
    )
    parser.add_argument(
        '--update-every',
        default=10,
        type=int,
        help=
        'Number of env interactions that should elapse between gradient descent updates'
    )
    parser.add_argument('--steps-per-epoch',
                        default=10000,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--random-steps',
                        default=10000,
                        type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--noise', default=0.1, type=float, help='train noise')
    parser.add_argument('--reward-scale',
                        default=0.1,
                        type=float,
                        help='reward scale factor')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--log-dir',
                        default='./logs-sac',
                        type=str,
                        help='log dir')
    parser.add_argument('--gpu-ids',
                        default='0',
                        type=str,
                        help='pytorch gpu device id')
    return parser.parse_args()


def main():
    # read args
    args = args_parser()

    # cuda setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # prepare sim env
    def env_fn():
        return gym.make(args.env)

    # sac
    if args.mode == 'train':
        # run parallel code with mpi
        mpi_fork(args.cpu)
        sac_pytorch(env_fn,
                    ac_kwargs={
                        'hidden_sizes': (128, 128 * 4, 128),
                        'activation': nn.ReLU,
                    },
                    steps_per_epoch=args.steps_per_epoch,
                    epochs=args.epochs,
                    seed=args.seed,
                    replay_size=args.replay_size,
                    gamma=args.gamma,
                    polyak=args.polyak,
                    lr=args.lr,
                    lr_decay=args.lr_decay,
                    reward_scale=args.reward_scale,
                    alpha=args.alpha,
                    batch_size=args.batch_size,
                    random_steps=args.random_steps,
                    warmup=args.warmup,
                    update_every=args.update_every,
                    num_test_episodes=args.validate_episodes,
                    max_ep_len=args.max_episode_length,
                    logger_kwargs={'output_dir': args.log_dir})
    else:
        test_sac_pytorch(env_fn=env_fn,
                         resume=args.resume,
                         ac_kwargs={
                             'hidden_sizes': (256, 256 * 4, 256),
                             'activation': nn.SiLU,
                         })


if __name__ == '__main__':
    main()
