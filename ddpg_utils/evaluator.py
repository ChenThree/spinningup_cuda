import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


class Evaluator(object):

    def __init__(self,
                 num_episodes,
                 interval,
                 save_path='',
                 max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes, 0)
        self.success = []

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        observation = None
        result = []
        success = 0

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.

            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                observation, reward, done, info = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length - 1:
                    done = True

                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            success += int(info['score/success'])

            if debug:
                print('[Evaluate] #Episode{}: episode_reward:{}'.format(
                    episode, episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])
        self.success.append(success / self.num_episodes)
        print('validate success rate ==', success / self.num_episodes)
        if save:
            self.save_results(self.save_path)
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error = np.std(self.results, axis=0)

        x = range(0, self.results.shape[1] * self.interval, self.interval)
        # reward
        plt.figure(figsize=(10, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(osp.join(fn, 'reward.png'))
        plt.close()
        savemat(osp.join(fn, 'reward.mat'), {'reward': self.results})
        # success rate
        plt.figure(figsize=(10, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Success Rate')
        plt.plot(x, self.success)
        plt.savefig(osp.join(fn, 'success_rate.png'))
        plt.close()
        savemat(osp.join(fn, 'success_rate.mat'),
                {'success_rate': self.success})
