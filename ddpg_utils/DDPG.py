import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from .memory import SequentialMemory
from .random_process import OrnsteinUhlenbeckProcess

network_size = 256
factor = 4


class Actor(nn.Module):

    def __init__(self, num_states, num_actions):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_states, network_size),
                                 nn.LayerNorm(network_size), nn.GELU(),
                                 nn.Linear(network_size, network_size * factor),
                                 nn.LayerNorm(network_size * factor), nn.GELU(),
                                 nn.Linear(network_size * factor, network_size),
                                 nn.LayerNorm(network_size), nn.GELU(),
                                 nn.Linear(network_size, num_actions),
                                 nn.Tanh())
        self.init_weights()

    def init_weights(self):
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class Critic(nn.Module):

    def __init__(self, num_states, num_actions):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_states + num_actions, network_size),
            nn.LayerNorm(network_size), nn.GELU(),
            nn.Linear(network_size, network_size * factor),
            nn.LayerNorm(network_size * factor), nn.GELU(),
            nn.Linear(network_size * factor, network_size),
            nn.LayerNorm(network_size), nn.GELU(), nn.Linear(network_size, 1))
        self.init_weights()

    def init_weights(self):
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        return self.mlp(torch.cat(xs, 1))


def to_tensor(ndarray, requires_grad=False, dtype=torch.cuda.FloatTensor):
    return Variable(torch.from_numpy(ndarray),
                    requires_grad=requires_grad).type(dtype)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self, num_states, num_actions, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.num_states = num_states
        self.num_actions = num_actions

        # Create Actor and Critic Network
        self.actor = Actor(self.num_states, self.num_actions)
        self.actor_target = Actor(self.num_states, self.num_actions)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.num_states, self.num_actions)
        self.critic_target = Critic(self.num_states, self.num_actions)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target,
                    self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.memory_size,
                                       window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=num_actions,
                                                       theta=args.ou_theta,
                                                       mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon_decay

        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch),
            self.actor_target(to_tensor(next_state_batch)),
        ])

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        criterion = nn.MSELoss()
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        # clamp grad
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(
            [to_tensor(state_batch),
             self.actor(to_tensor(state_batch))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        # clamp grad
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.num_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        self.eval()
        with torch.no_grad():
            action = self.actor(to_tensor(np.array([s_t])))
        action = action.cpu().data.numpy().squeeze(0)
        action += self.is_training * max(self.epsilon,
                                         0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        self.train()
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output, iter):
        if output is None:
            return

        self.actor.load_state_dict(
            torch.load('{}/actor_{}.pth'.format(output, iter)))

        self.critic.load_state_dict(
            torch.load('{}/critic_{}.pth'.format(output, iter)))

    def save_model(self, output, iter):
        torch.save(self.actor.state_dict(),
                   '{}/actor_{}.pth'.format(output, iter))
        torch.save(self.critic.state_dict(),
                   '{}/critic_{}.pth'.format(output, iter))

    def seed(self, s):
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
