import argparse
import os
import time
from copy import deepcopy

import gym
import numpy as np
import robel
import torch

from ddpg_utils import DDPG
from evaluator import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
env_name = 'DKittyStandFixed-v0'


def args_parser():
    parser = argparse.ArgumentParser(description='DDPG DKiity')
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--rate',
                        default=0.001,
                        type=float,
                        help='learning rate')
    parser.add_argument('--prate',
                        default=0.0001,
                        type=float,
                        help='policy net learning rate (only for DDPG)')
    parser.add_argument(
        '--warmup',
        default=10000,
        type=int,
        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--batch-size',
                        default=1024,
                        type=int,
                        help='minibatch size')
    parser.add_argument('--memory-size',
                        default=6000000,
                        type=int,
                        help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau',
                        default=0.001,
                        type=float,
                        help='moving average for target network')
    parser.add_argument('--ou_theta',
                        default=0.15,
                        type=float,
                        help='noise theta')
    parser.add_argument('--ou_sigma',
                        default=0.2,
                        type=float,
                        help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument(
        '--validate_episodes',
        default=100,
        type=int,
        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=2000, type=int, help='')
    parser.add_argument('--validate_steps',
                        default=20000,
                        type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--train_iter',
                        default=400000,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--epsilon_decay',
                        default=100000,
                        type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--resume',
                        default='default',
                        type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--output',
                        default='./checkpoint',
                        type=str,
                        help='output path')
    return parser.parse_args()


def train(agent: DDPG, env, evaluate, args, debug=True):

    agent.is_training = True
    last_time = time.time()
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < args.train_iter:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            with torch.no_grad():
                action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if args.max_episode_length and episode_steps >= args.max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup:
            agent.update_policy()

        # [optional] evaluate
        if evaluate is not None and args.validate_steps > 0 and step % args.validate_steps == 0:
            with torch.no_grad():
                policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env,
                                       policy,
                                       debug=False,
                                       visualize=False)
            if debug:
                print('[Evaluate] Step_{:07d}: mean_reward:{}'.format(
                    step, validate_reward))

        # [optional] save intermideate model
        if step % int(args.train_iter / 3) == 0:
            agent.save_model(args.output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            if debug:
                speed = step / (time.time() - last_time)
                print(
                    '#{}: episode_reward: {:.2f}   steps: {}   speed: {:.2f} steps/s   estimate: {:.2f} s'
                    .format(episode, episode_reward, step, speed,
                            (args.train_iter - step) / speed))
            with torch.no_grad():
                agent.memory.append(observation,
                                    agent.select_action(observation), 0., False)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1


def main():
    # read args
    args = args_parser()
    # prepare sim env
    # action: 12 * [-1, 1] observation: 61 * [-inf, inf]
    env = gym.make(env_name)
    env.reset()
    # set random seed
    np.random.seed(args.seed)
    env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes,
                         args.validate_steps,
                         args.output,
                         max_episode_length=args.max_episode_length)

    train(agent, env, evaluate, args)
    env.close()


if __name__ == '__main__':
    main()
