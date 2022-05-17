import math
import random

import numpy as np
import torch
import torch.nn as nn

from ...utils.model import BaseModule


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim),
                                 dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.int64)
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
            k:
            torch.as_tensor(v,
                            dtype=torch.int64 if k == 'act' else torch.float32)
            for k, v in batch.items()
        }


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [
                nn.Linear(sizes[j], sizes[j + 1]),
                nn.LayerNorm(sizes[j + 1]),
                activation()
            ]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), output_activation()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPDoubleDQN(BaseModule):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # define Q target and Q
        mlp_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        # set up double q network
        self.q1 = mlp(mlp_sizes, activation)
        self.q2 = mlp(mlp_sizes, activation)
        # init
        self.init_weights()
        # cuda
        self.cuda()

    def forward(self, x):
        return self.q1(x), self.q2(x)
