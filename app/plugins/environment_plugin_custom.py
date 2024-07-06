import pandas as pd
import numpy as np
import openrl
from openrl.algorithms import PPO, DQN
import gym
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

    def build_environment(self, x_train, y_train):
        self.env = CustomPredictionEnv(x_train, y_train, **self.params['env_params'])
    
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
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        # Collect evaluation metrics
        return reward

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        if self.params['algorithm'] == 'PPO':
            self.model = PPO.load(file_path)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN.load(file_path)

class CustomPredictionEnv(gym.Env):
    def __init__(self, x_train, y_train, time_horizon, observation_space_size, action_space_size):
        self.x_train = x_train
        self.y_train = y_train
        self.time_horizon = time_horizon
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_space_size,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        return self.x_train[self.current_step]

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.x_train):
            done = True
        else:
            done = False

        prediction = action[0]
        true_value = self.y_train[self.current_step]
        reward = 1.0 / np.abs(true_value - prediction)  # Inverse of MAE as fitness function
        observation = self.x_train[self.current_step] if not done else np.zeros_like(self.x_train[0])
        return observation, reward, done, {}

    def render(self, mode='human'):
        pass

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    x_train = pd.read_csv('tests/data/encoder_eval.csv').values
    y_train = pd.read_csv('tests/data/csv_sel_unb_norm_512.csv').values

    plugin.set_params(algorithm='PPO', total_timesteps=10000)
    plugin.build_environment(x_train, y_train)
    plugin.build_model()
    plugin.train()
    fitness = plugin.evaluate()
    print(f"Evaluation fitness: {fitness}")
    plugin.save('ppo_model.zip')
