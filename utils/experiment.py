import random
import numpy as np
import logging
import time


class Experiment(object):
    def __init__(self, agent, environment, episodes=100, max_steps=100000,
                 seed=0, steps_log=None, ep_log=None):
        self.agent = agent
        self.environment = environment
        self.episodes = episodes
        self.step_count = 0
        self.max_steps = max_steps
        self.rewards = []
        self.episode = 0
        self.set_seed(seed)
        self.steps_logger = logging.getLogger(steps_log)
        self.ep_logger = logging.getLogger(ep_log)
        self.start_time = time.time()
        self.episode_count = 0

    def run_step_mode(self):
        while self.step_count < self.max_steps:
            rewards = self.run_episode()
            self.episode_count += 1
            self.ep_logger.info("Time {:12s} | Episode: {:6d} | Num_steps: {:8d} | Rewards: {:6.1f}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time)),
                self.episode_count, self.step_count, np.sum(rewards)
            ))
        self.environment.close()
        return self.rewards

    def run_episode(self):
        obs = self.environment.reset()
        action = self.agent.start(obs)
        done = False
        rewards = []
        while not (done or self.step_count == self.max_steps):
            next_obs, reward, done, info = self.environment.step(action)
            rewards.append(reward)
            self.step_count += 1

            if self.step_count % 500 == 0:
                self.steps_logger.info("Time {:12s} | Num_steps: {:8d} | Episode: {:6d}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time)),
                    self.step_count, self.episode_count
                ))

            self.agent.update(obs, action, reward, next_obs, done)
            action = self.agent.get_action(next_obs)
            obs = next_obs
        return rewards

    def set_seed(self, seed):
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            self.environment.set_seed(seed)
            self.agent.set_seed(seed)
