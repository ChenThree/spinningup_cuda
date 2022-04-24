import argparse
import os
import time
from copy import deepcopy

import gym
import numpy as np
import robel
import torch
import torch.nn as nn

from spinningup import sac_pytorch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def args_parser():
    parser = argparse.ArgumentParser(description='DDPG DKiity')
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
    parser.add_argument('--alpha',
                        default=0.2,
                        type=float,
                        help='entropy regularization coefficient')
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
                        default=100000,
                        type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--noise', default=0.1, type=float, help='train noise')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='Resuming model path for testing')
    return parser.parse_args()


def main():
    # read args
    args = args_parser()

    # prepare sim env
    def env_fn():
        return gym.make(args.env)

    # sac
    sac_pytorch(env_fn,
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
                lr=args.lr,
                alpha=args.alpha,
                batch_size=args.batch_size,
                random_steps=args.random_steps,
                warmup=args.warmup,
                update_every=args.update_every,
                num_test_episodes=args.validate_episodes,
                max_ep_len=args.max_episode_length,
                logger_kwargs={'output_dir': './logs-sac'})


if __name__ == '__main__':
    main()
