import pandas as pd
import numpy as np
from openrl.algorithms.ppo import PPOAlgorithm
from openrl.algorithms.dqn import DQNAlgorithm
import pickle

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        # Initialize additional attributes with default values if they are not present in config_dict
        self.use_share_model = getattr(self, 'use_share_model', False)
        self.use_joint_action_loss = getattr(self, 'use_joint_action_loss', False)
        self.use_deepspeed = getattr(self, 'use_deepspeed', False)
        self.world_size = getattr(self, 'world_size', 1)
        self.clip_param = getattr(self, 'clip_param', 0.2)  # Add clip_param with default value
        self.use_valuenorm = getattr(self, 'use_valuenorm', False)  # Add use_valuenorm with default value
        self.value_loss_coef = getattr(self, 'value_loss_coef', 0.5)  # Add value_loss_coef with default value
        self.entropy_coef = getattr(self, 'entropy_coef', 0.01)  # Add entropy_coef with default value
        self.max_grad_norm = getattr(self, 'max_grad_norm', 0.5)  # Add max_grad_norm with default value
        self.use_huber_loss = getattr(self, 'use_huber_loss', False)  # Add use_huber_loss with default value
        self.huber_delta = getattr(self, 'huber_delta', 1.0)  # Add huber_delta with default value
        self.use_clipped_value_loss = getattr(self, 'use_clipped_value_loss', True)  # Add use_clipped_value_loss with default value
        self.use_value_active_masks = getattr(self, 'use_value_active_masks', True)  # Add use_value_active_masks with default value
        self.use_policy_active_masks = getattr(self, 'use_policy_active_masks', True)  # Add use_policy_active_masks with default value
        self.use_policy_vhead = getattr(self, 'use_policy_vhead', False)  # Add use_policy_vhead with default value
        self.use_adv_normalize = getattr(self, 'use_adv_normalize', True)  # Add use_adv_normalize with default value
        self.ppo_epoch = getattr(self, 'ppo_epoch', 10)  # Add ppo_epoch with default value
        self.num_mini_batch = getattr(self, 'num_mini_batch', 32)  # Add num_mini_batch with default value
        self.data_chunk_length = getattr(self, 'data_chunk_length', 10)  # Add data_chunk_length with default value
        self.dual_clip_ppo = getattr(self, 'dual_clip_ppo', False)  # Add dual_clip_ppo with default value
        self.dual_clip_coeff = getattr(self, 'dual_clip_coeff', 10.0)  # Add dual_clip_coeff with default value

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
        self.env = environment
        self.x_train = x_train
        self.y_train = y_train
    
    def build_model(self):
        config = Config(self.params)
        if self.params['algorithm'] == 'PPO':
            self.model = PPOAlgorithm(cfg=config, init_module=self.env)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQNAlgorithm(cfg=config, init_module=self.env)

    def train(self):
        self.model.learn(total_timesteps=self.params['total_timesteps'])

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
        if self.params['algorithm'] == 'PPO':
            self.model = PPOAlgorithm.load(file_path)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQNAlgorithm.load(file_path)
