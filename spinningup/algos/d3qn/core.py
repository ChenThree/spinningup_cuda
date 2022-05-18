import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from ...utils.model import BaseModule


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_type=np.float32):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=obs_type)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.int16)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.bool8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        # np.random.choice slow when not replace
        idxs = random.sample(range(0, self.size), batch_size)
        next_idxs = list((np.array(idxs) + 1) % self.max_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs_buf[next_idxs],
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


# TwinDQN
class MLPDoubleDQN(BaseModule):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # set up double q network
        # feature network
        feature_sizes = [obs_dim] + list(hidden_sizes)
        self.features = mlp(feature_sizes,
                            activation,
                            output_activation=activation)
        # value network
        value_sizes = [hidden_sizes[-1], hidden_sizes[-1], act_dim]
        self.q1 = mlp(value_sizes, activation)
        self.q2 = mlp(value_sizes, activation)
        # init
        self.init_weights()
        # cuda
        self.cuda()

    def forward(self, x):
        features = self.features(x)
        return self.q1(features), self.q2(features)

    def get_action(self, o, eps):
        # using single network to choose action
        with torch.no_grad():
            q = self.q1(
                self.features(torch.as_tensor(o, dtype=torch.float32).cuda()))
        # eps exploration
        if random.random() > eps:
            action = q.argmax(dim=0)
        else:
            action_prob = F.softmax(q, dim=0)
            action = torch.multinomial(action_prob, num_samples=1).squeeze()
        return action.cpu().numpy()


# D3QN
class DualDoubleDQN(BaseModule):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        features = self.features(x)
        q_val1 = self.val1(features)
        q_adv1 = self.adv1(features)
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1

        q_val2 = self.val2(features)
        q_adv2 = self.adv2(features)
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        return q_duel1, q_duel2

    def get_action(self, o, eps):
        # using single network to choose action
        with torch.no_grad():
            features = self.features(
                torch.as_tensor(o, dtype=torch.float32).cuda())
            q = self.val1(features).squeeze()
        # eps exploration
        if random.random() > eps:
            action = q.argmax(dim=0)
        else:
            action_prob = F.softmax(q, dim=0)
            action = torch.multinomial(action_prob, num_samples=1).squeeze()
        return action.cpu().numpy()


class MLPDualDoubleDQN(DualDoubleDQN):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # set up dual double q network
        # feature network
        feature_sizes = [obs_dim] + list(hidden_sizes)
        self.features = mlp(feature_sizes,
                            activation,
                            output_activation=activation)
        # value network
        value_sizes = [hidden_sizes[-1], hidden_sizes[-1], act_dim]
        self.val1 = mlp(value_sizes, activation)
        self.val2 = mlp(value_sizes, activation)
        # advantage network
        adv_sizes = [hidden_sizes[-1], hidden_sizes[-1], 1]
        self.adv1 = mlp(adv_sizes, activation)
        self.adv2 = mlp(adv_sizes, activation)
        # init
        self.init_weights()
        # cuda
        self.cuda()


class CNNDualDoubleDQN(DualDoubleDQN):

    def __init__(self,
                 observation_space,
                 action_space,
                 kernels=(5, 3, 3),
                 channels=(32, 64, 128),
                 activation=nn.ReLU):
        super().__init__()

        obs_shape = observation_space.shape
        self.input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        act_dim = action_space.n

        # set up dual double q network
        # feature network
        self.features = nn.Sequential()
        for i in range(len(kernels)):
            if i == 0:
                self.features.add_module(
                    f'conv{i}',
                    nn.Conv2d(obs_shape[2], channels[i], kernels[i], 2))
            else:
                self.features.add_module(
                    f'conv{i}',
                    nn.Conv2d(channels[i - 1], channels[i], kernels[i]))
            self.features.add_module(f'pool{i}', nn.MaxPool2d(4, 2))
            self.features.add_module(f'act{i}', activation())
        self.features.add_module(f'avrpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.features.add_module(f'flat', nn.Flatten())
        # print network structure
        summary(self.features, self.input_shape, device='cpu')
        # calculate feature shape
        feature_size = channels[-1]
        print('feature size ==', feature_size)
        # value network
        value_sizes = [feature_size, feature_size, act_dim]
        self.val1 = mlp(value_sizes, activation)
        self.val2 = mlp(value_sizes, activation)
        # advantage network
        adv_sizes = [feature_size, feature_size, 1]
        self.adv1 = mlp(adv_sizes, activation)
        self.adv2 = mlp(adv_sizes, activation)
        # init
        self.init_weights()
        # cuda
        self.cuda()
