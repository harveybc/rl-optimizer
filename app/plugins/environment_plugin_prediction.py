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

    def build_environment(self, x_train, y_train, config):
        self.time_horizon = config['time_horizon']
        self.max_steps = config['max_steps']
        self.env = PredictionEnv(x_train, y_train, config)


    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def calculate_fitness(self, y_true, y_pred):
        """Calculate fitness as the inverse of the mean absolute error."""
        mae = np.mean(np.abs(y_true - y_pred))
        if mae == 0:
            return float('inf')  # If there is no error, fitness is infinite
        return 1.0 / mae

class PredictionEnv(gym.Env):
    """
    A custom environment for prediction tasks.
    """

    def __init__(self, x_train, y_train, config):
        super(PredictionEnv, self).__init__()
        self.max_steps = config['max_steps']
        self.x_train = x_train
        self.y_train = y_train
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.x_train.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.done = False
        self.reset()

    def reset(self):
        self.current_step = 0
        observation = self.x_train[self.current_step] if not self.done else np.zeros_like(self.x_train[0])
        return observation
        

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.x_train.shape[0]:
            self.current_step = self.x_train.shape[0] - 1
            self.done = True
        else:
            self.done = self.current_step >= self.max_steps
        prediction = action[0]
        true_value = self.y_train[self.current_step]  # Assuming the first column is the target
        reward = 1/np.abs(true_value - prediction) if true_value != prediction else float('inf')  # Fitness function as inverse of error
        observation = self.x_train[self.current_step] if not self.done else np.zeros_like(self.x_train[0])
        return observation, reward, self.done, {}


    def render(self, mode='human'):
        pass

