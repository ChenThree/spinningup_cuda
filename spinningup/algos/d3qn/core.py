import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from ...utils.model import BaseModule


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
                torch.as_tensor(o / 255.0, dtype=torch.float32).cuda())
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
                 strides=(1, 1, 1),
                 pools=(True, False, False),
                 channels=(32, 64, 128),
                 activation=nn.ReLU):
        super().__init__()

        self.input_shape = observation_space.shape
        act_dim = action_space.n

        # set up dual double q network
        # feature network
        self.features = nn.Sequential()
        for i in range(len(kernels)):
            if i == 0:
                self.features.append(
                    nn.Conv2d(self.input_shape[0], channels[i], kernels[i],
                              strides[i]))
            else:
                self.features.append(
                    nn.Conv2d(channels[i - 1], channels[i], kernels[i],
                              strides[i]))
            if pools[i]:
                self.features.append(nn.MaxPool2d(4, 2))
            self.features.append(activation())
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features.append(nn.Flatten())
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
