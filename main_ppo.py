import argparse
import os
import time
from copy import deepcopy

import gym
import numpy as np
import robel
import torch
import torch.nn as nn

from spinningup import ppo_pytorch

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
    parser.add_argument('--plr',
                        default=0.0003,
                        type=float,
                        help='policy learning rate')
    parser.add_argument('--vflr',
                        default=0.001,
                        type=float,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--batch-size',
                        default=512,
                        type=int,
                        help='minibatch size')
    parser.add_argument('--replay-size',
                        default=1000000,
                        type=int,
                        help='replay-size')
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
        help='time without training but only filling the replay memory')
    parser.add_argument('--steps-per-epoch',
                        default=10000,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--output',
                        default='./checkpoint/ddpg1',
                        type=str,
                        help='output path')
    return parser.parse_args()


def main():
    # read args
    args = args_parser()

    # prepare sim env
    def env_fn():
        return gym.make(args.env)

    # ppo
    ppo_pytorch(env_fn,
                ac_kwargs={
                    'hidden_sizes': (256, 256 * 4, 256),
                    'activation': nn.SiLU,
                },
                steps_per_epoch=args.steps_per_epoch,
                epochs=args.epochs,
                seed=args.seed,
                gamma=args.gamma,
                pi_lr=args.plr,
                vf_lr=args.vflr,
                clip_ratio=0.2,
                train_pi_iters=80,
                train_v_iters=80,
                lam=0.97,
                target_kl=0.01,
                max_ep_len=args.max_episode_length,
                logger_kwargs={'output_dir': './logs-ppo'})


if __name__ == '__main__':
    main()
