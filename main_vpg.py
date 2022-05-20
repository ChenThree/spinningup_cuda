import argparse
import os
import time
from copy import deepcopy

import gym
import numpy as np
import robel
import torch
import torch.nn as nn

from spinningup import vpg_pytorch


def args_parser():
    parser = argparse.ArgumentParser(description='DDPG DKiity')
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
                        default=0.0003,
                        type=float,
                        help='policy learning rate')
    parser.add_argument('--vflr',
                        default=0.001,
                        type=float,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
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
    parser.add_argument('--steps-per-epoch',
                        default=10000,
                        type=int,
                        help='train iters each timestep')
    return parser.parse_args()


def main():
    # read args
    args = args_parser()

    # prepare sim env
    def env_fn():
        env = gym.make(args.env)
        env.seed(args.seed)
        return env

    # cuda setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # ddpg
    vpg_pytorch(env_fn,
                ac_kwargs={
                    'hidden_sizes': (128, 128 * 4, 128),
                    'activation': nn.ReLU,
                },
                steps_per_epoch=args.steps_per_epoch,
                epochs=args.epochs,
                seed=args.seed,
                gamma=args.gamma,
                pi_lr=args.plr,
                vf_lr=args.vflr,
                max_ep_len=args.max_episode_length,
                logger_kwargs={'output_dir': './logs-vpg'})


if __name__ == '__main__':
    main()
