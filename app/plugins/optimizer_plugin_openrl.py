import pandas as pd
import numpy as np
from openrl.algorithms.ppo import PPOAlgorithm
from openrl.algorithms.dqn import DQNAlgorithm


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

    def build_environment(self, environment_class, x_train, y_train):
        self.env = environment_class(x_train, y_train, **self.params['env_params'])
    
    def build_model(self):
        if self.params['algorithm'] == 'PPO':
            self.model = PPOAlgorithm('MlpPolicy', self.env, verbose=1)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQNAlgorithm('MlpPolicy', self.env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.params['total_timesteps'])

    def evaluate(self):
        obs = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
        return total_reward

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        if self.params['algorithm'] == 'PPO':
            self.model = PPOAlgorithm.load(file_path)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQNAlgorithm.load(file_path)
