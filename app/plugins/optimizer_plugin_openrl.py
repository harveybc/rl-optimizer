import pandas as pd
import numpy as np
import openrl
from openrl.algorithms.ppo import PPOAlgorithm as PPO
from openrl.algorithms.dqn import DQNAlgorithm as DQN
import pickle

class Plugin:
    """
    An optimizer plugin using OpenRL, supporting multiple algorithms.
    """

    plugin_params = {
        'algorithm': 'PPO',
        'total_timesteps': 10000,
        'env_params': {
            'time_horizon': 12,
            'observation_space_size': 8,  # Adjust based on your x_train data
            'action_space_size': 1,
        }
    }

    plugin_debug_vars = ['algorithm', 'total_timesteps']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.env = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_environment(self, environment, x_train, y_train):
        self.env = environment  # Correctly receive the environment instance
        self.env.x_train = x_train
        self.env.y_train = y_train

    def build_model(self):
        if self.params['algorithm'] == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN('MlpPolicy', self.env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.params['total_timesteps'])

    def evaluate(self):
        obs = self.env.reset()
        done = False
        rewards = []
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
        # Collect evaluation metrics
        return np.mean(rewards), np.mean(np.abs(rewards))

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        if self.params['algorithm'] == 'PPO':
            self.model = PPO.load(file_path)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN.load(file_path)
