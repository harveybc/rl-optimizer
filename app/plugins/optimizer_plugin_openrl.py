import pandas as pd
import numpy as np
from openrl.algorithms.ppo import PPOAlgorithm
from torch.optim import Adam
import pickle

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

class Plugin:
    """
    An optimizer plugin using OpenRL, supporting PPO.
    """

    plugin_params = {
        'algorithm': 'PPO',
        'total_timesteps': 10000,
        'clip_param': 0.2,
        'ent_coef': 0.01,
        'learning_rate': 3e-4,
        'mini_batch_size': 64,  # Default value, adjust as needed
        'bc_epoch': 10,         # Default value, adjust as needed
        'env_params': {
            'time_horizon': 12,
            'observation_space_size': 8,  # Adjust based on your x_train data
            'action_space_size': 1,
        }
    }

    plugin_debug_vars = ['algorithm', 'total_timesteps', 'clip_param', 'ent_coef', 'learning_rate', 'mini_batch_size', 'bc_epoch']

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
        self.env = environment
        self.x_train = x_train
        self.y_train = y_train
    
    def build_model(self):
        config = Config(self.params)
        self.model = PPOAlgorithm(cfg=config, init_module=self.env)
        # Set optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.params['learning_rate'])

    def train(self):
        self.model.learn(total_timesteps=self.params['total_timesteps'], optimizer=self.optimizer)

    def evaluate(self):
        obs = self.env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        return reward

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = PPOAlgorithm.load(file_path)
