import gym
import numpy as np
import robel

env_name = 'DKittyWalkRandom-v0'


def main():
    # prepare sim env
    # action: 12 * [-1, 1] observation: 61 * [-inf, inf]
    env = gym.make(env_name)
    env.reset()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    print(num_states, num_actions)
    print(env.action_space.high, env.action_space.low)
    print(env.observation_space.high, env.observation_space.low)
    for i in range(10000):
        # env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        # if done:
        #     print(i, reward, done)
        #     print(info['score/success'])
        #     break

    env.close()


if __name__ == '__main__':
    main()
