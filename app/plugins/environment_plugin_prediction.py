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

    def build_environment(self):
        self.env = PredictionEnv(time_horizon=self.params['time_horizon'], max_steps=self.params['max_steps'])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

class PredictionEnv(gym.Env):
    """
    A custom environment for prediction tasks.
    """

    def __init__(self, time_horizon=10, max_steps=1000):
        super(PredictionEnv, self).__init__()
        self.time_horizon = time_horizon
        self.max_steps = max_steps
        self.current_step = 0
        self.data = self.load_data()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def load_data(self):
        # Load your dataset here
        data = np.random.rand(1000, 10)  # Example data
        return data

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        prediction = action[0]
        true_value = self.data[self.current_step, 0]  # Assuming the first column is the target
        reward = 1.0 / np.abs(true_value - prediction)  # Fitness function as inverse of error
        observation = self.data[self.current_step] if not done else np.zeros_like(self.data[0])
        return observation, reward, done, {}

    def render(self, mode='human'):
        pass

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.set_params(time_horizon=10, max_steps=1000)
    plugin.build_environment()
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
    observation = plugin.reset()
    for _ in range(10):
        action = np.array([0.5])  # Example action
        observation, reward, done, _ = plugin.step(action)
        if done:
            break
    plugin.render()