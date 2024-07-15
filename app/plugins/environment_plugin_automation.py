import gym
import numpy as np
import pandas as pd

class Plugin:
    """
    An environment plugin for forex trading automation tasks, compatible with both NEAT and OpenRL.
    """

    plugin_params = {
        'initial_balance': 10000,
        'max_steps': 500,
        'fitness_function': 'brute_profit'  # 'sharpe_ratio' can be another option
    }

    plugin_debug_vars = ['initial_balance', 'max_steps', 'fitness_function']

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
        self.initial_balance = config.get('initial_balance', self.params['initial_balance'])
        self.max_steps = config.get('max_steps', self.params['max_steps'])
        self.fitness_function = config.get('fitness_function', self.params['fitness_function'])
        self.env = AutomationEnv(x_train, y_train, self.initial_balance, self.max_steps, self.fitness_function)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def calculate_fitness(self, rewards, equity_curve=None):
        if self.fitness_function == 'sharpe_ratio':
            return self._calculate_sharpe_ratio(equity_curve)
        else:  # Default to brute_profit
            return rewards.sum() / len(rewards)

    def _calculate_sharpe_ratio(self, equity_curve):
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        return return_ratio * np.sqrt(252)

class AutomationEnv(gym.Env):
    def __init__(self, x_train, y_train, initial_balance, max_steps, fitness_function):
        super(AutomationEnv, self).__init__()
        self.max_steps = max_steps
        self.x_train = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else x_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.current_step = 0
        self.order_status = 0  # 0 = no order, 1 = buy, -1 = sell
        self.order_price = 0.0
        self.order_volume = 0.0
        self.done = False
        self.reward = 0.0
        self.equity_curve = [initial_balance]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.x_train.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.order_status = 0
        self.order_price = 0.0
        self.order_volume = 0.0
        self.reward = 0.0
        self.done = False
        self.equity_curve = [self.initial_balance]
        observation = self.x_train[self.current_step]
        return observation

    def step(self, action):
        if self.done:
            return np.zeros(self.x_train.shape[1]), self.reward, self.done, {}
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
        
        # Simulate trading action
        close_price = self.y_train[self.current_step, 0]
        if self.order_status == 1:
            self.reward = close_price - self.order_price
        elif self.order_status == -1:
            self.reward = self.order_price - close_price
        else:
            self.reward = 0

        # Update equity
        self.equity += self.reward
        self.equity_curve.append(self.equity)

        # Decide action: -1 sell, 0 hold, 1 buy
        if action[0] > 0:
            self.order_status = 1
            self.order_price = close_price
        elif action[0] < 0:
            self.order_status = -1
            self.order_price = close_price
        else:
            self.order_status = 0
        
        # Check for margin call
        if self.equity <= 0:
            self.done = True
            self.reward -= self.initial_balance

        observation = self.x_train[self.current_step] if not self.done else np.zeros(self.x_train.shape[1])
        return observation, self.reward, self.done, {}

    def render(self, mode='human'):
        pass
