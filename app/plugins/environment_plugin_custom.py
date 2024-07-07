import gym
import numpy as np

class Plugin:
    """
    An environment plugin for prediction tasks, compatible with both NEAT and OpenRL.
    """

    plugin_params = {
        'time_horizon': 10,
        'max_steps': 1000
    }

    plugin_debug_vars = ['time_horizon', 'max_steps']

    def __init__(self):
        self.params = self.plugin_params.copy()
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
        self.env = PredictionEnv(x_train, y_train, self.params['time_horizon'], self.params['max_steps'])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def calculate_fitness(self, y_true, y_pred):
        """Calculate fitness as the inverse of the mean absolute error and mean squared error."""
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred)**2)
        return mae, mse

class PredictionEnv:
    def __init__(self, x_train, y_train, time_horizon, max_steps):
        self.x_train = x_train
        self.y_train = y_train
        self.time_horizon = time_horizon
        self.max_steps = max_steps
        self.current_step = 0
        self.world_size = 1  # Ensure this attribute is included
        self.use_share_model = False  # Ensure this attribute is included
        self.use_joint_action_loss = False  # Ensure this attribute is included
        self.use_deepspeed = False  # Ensure this attribute is included

    def reset(self):
        self.current_step = 0
        return self.x_train[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = self.y_train[self.current_step] if not done else 0
        info = {}
        return self.x_train[self.current_step], reward, done, info

    def render(self, mode='human'):
        pass
