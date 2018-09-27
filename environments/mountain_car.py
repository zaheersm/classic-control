import random
import gym

from environments.environment import Environment


class MountainCar(Environment):
    def __init__(self, cfg):
        super(MountainCar, self).__init__()
        self.instance = gym.make(cfg.env_name).env
        self.obs_high = self.instance.observation_space.high
        self.obs_low = self.instance.observation_space.low
        self.action_noise = cfg.action_noise

    def scale_obs(self, obs):
        # Scale all observation dimensions to be between 0 and 1
        # obs is between -1 and 1 already, with a few spikes.
        return (obs - self.obs_low) / (self.obs_high - self.obs_low)

    def reset(self):
        obs = self.instance.reset()
        return self.scale_obs(obs)

    def step(self, a):
        eps = random.uniform(0, 1)
        if eps < self.action_noise:
            a = random.choice(range(self.num_actions()))
        obs, reward, done, info = self.instance.step(a)
        return self.scale_obs(obs), reward, done, info

    def num_obs(self):
        return self.instance.observation_space.shape[0]

    def num_actions(self):
        return self.instance.action_space.n

    def close(self):
        self.instance.close()

    def set_seed(self, seed):
        self.instance.seed(seed)
        self.seed = seed
