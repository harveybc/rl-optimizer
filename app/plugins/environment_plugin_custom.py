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
        """Calculate fitness as the inverse of the mean absolute error."""
        mae = np.mean(np.abs(y_true - y_pred))
        if mae == 0:
            return float('inf')  # If there is no error, fitness is infinite
        return 1.0 / mae

class PredictionEnv(gym.Env):
    """
    A custom environment for prediction tasks.
    """

    def __init__(self, x_train, y_train, time_horizon=10, max_steps=1000):
        super(PredictionEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.time_horizon = time_horizon
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(x_train.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        return self.x_train.iloc[self.current_step].to_numpy()

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.x_train):
            done = True
        else:
            done = False

        prediction = action[0]
        true_value = self.y_train.iloc[self.current_step].values[0]
        reward = 1.0 / np.abs(true_value - prediction)  # Inverse of MAE as fitness function
        observation = self.x_train.iloc[self.current_step].to_numpy() if not done else np.zeros_like(self.x_train.iloc[0])
        return observation, reward, done, {}

    def render(self, mode='human'):
        pass


# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    x_train = np.random.rand(100, 8)
    y_train = np.random.rand(100, 1)
    plugin.set_params(time_horizon=10, max_steps=1000)
    plugin.build_environment(x_train, y_train)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
    observation = plugin.reset()
    for _ in range(10):
        action = np.array([0.5])  # Example action
        observation, reward, done, _ = plugin.step(action)
        if done:
            break
    plugin.render()
