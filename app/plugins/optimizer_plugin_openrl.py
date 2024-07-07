import pandas as pd
import numpy as np
from openrl.algorithms.ppo import PPOAlgorithm
from openrl.algorithms.dqn import DQNAlgorithm
import pickle

class Config:
    def __init__(self, config_dict):
        # Initialize required attributes with defaults
        self.use_share_model = config_dict.get('use_share_model', False)
        self.use_joint_action_loss = config_dict.get('use_joint_action_loss', False)
        self.use_deepspeed = config_dict.get('use_deepspeed', False)
        self.use_amp = config_dict.get('use_amp', False)
        self.clip_param = config_dict.get('clip_param', 0.2)
        self.huber_delta = config_dict.get('huber_delta', 1.0)
        self.use_huber_loss = config_dict.get('use_huber_loss', True)
        self.use_clipped_value_loss = config_dict.get('use_clipped_value_loss', True)
        self.use_value_active_masks = config_dict.get('use_value_active_masks', False)
        self.use_policy_active_masks = config_dict.get('use_policy_active_masks', False)
        self.use_policy_vhead = config_dict.get('use_policy_vhead', False)
        self.use_max_grad_norm = config_dict.get('use_max_grad_norm', True)
        self.max_grad_norm = config_dict.get('max_grad_norm', 0.5)
        self.ppo_epoch = config_dict.get('ppo_epoch', 4)
        self.num_mini_batch = config_dict.get('num_mini_batch', 32)
        self.data_chunk_length = config_dict.get('data_chunk_length', 10)
        self.use_adv_normalize = config_dict.get('use_adv_normalize', True)
        self.use_popart = config_dict.get('use_popart', False)
        self.use_valuenorm = config_dict.get('use_valuenorm', False)
        self.dual_clip_ppo = config_dict.get('dual_clip_ppo', False)
        self.dual_clip_coeff = config_dict.get('dual_clip_coeff', 10.0)
        self.entropy_coef = config_dict.get('entropy_coef', 0.01)
        self.value_loss_coef = config_dict.get('value_loss_coef', 0.5)
        self.policy_value_loss_coef = config_dict.get('policy_value_loss_coef', 0.5)
        self.tpdv = config_dict.get('tpdv', None)
        self.world_size = config_dict.get('world_size', 1)
        # Add other config parameters
        self.__dict__.update(config_dict)

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
