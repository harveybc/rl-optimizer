# environment_plugin_automation.py

import gym
import numpy as np
import pandas as pd
from collections import deque

class Plugin:
    """
    An environment plugin for automated trading tasks, compatible with both NEAT and OpenRL.
    """

    plugin_params = {
        'initial_balance': 10000,
        'commission': 0.0002,
        'slippage': 0.0001,
        'leverage': 50,
        'max_steps': 1000,
        'risk_free_rate': 0.01,
        'fitness_function': 'brute_profit'
    }

    plugin_debug_vars = [
        'initial_balance', 'current_balance', 'commission', 'slippage', 
        'max_steps', 'current_step', 'total_trades', 'successful_trades', 
        'unsuccessful_trades', 'max_drawdown', 'risk_free_rate', 
        'current_position', 'returns'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.env = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_environment(self, x_train, y_train, config):
        self.env = AutomationEnv(x_train, y_train, config)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def calculate_fitness(self):
        return self.env.calculate_fitness()

class AutomationEnv(gym.Env):
    def __init__(self, x_train, y_train, config):
        super(AutomationEnv, self).__init__()
        self.max_steps = config['max_steps']
        self.initial_balance = config['initial_balance']
        self.commission = config['commission']
        self.slippage = config['slippage']
        self.leverage = config['leverage']
        self.risk_free_rate = config['risk_free_rate']
        self.fitness_function = config.get('fitness_function', 'brute_profit')

        self.x_train = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else x_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train

        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.orders = []
        self.done = False

        self.returns = []

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.y_train.shape[1],), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.orders = []
        self.done = False
        self.returns = []
        return self.y_train[self.current_step]

    def step(self, action):
        if self.done:
            raise ValueError("Cannot step in a done environment. Please reset.")

        self.current_step += 1
        current_price = self.x_train[self.current_step]
        previous_price = self.x_train[self.current_step - 1]
        current_features = self.y_train[self.current_step]

        reward = 0

        if action == 1:  # Buy
            self._execute_order(current_price, 1)
        elif action == 2:  # Sell
            self._execute_order(current_price, -1)

        self._update_orders(current_price, previous_price)

        self.equity = self.balance + sum([order['profit'] for order in self.orders])
        if self.current_step > 0 and self.cumulative_returns[-1] != 0:
            self.returns.append((self.equity - self.cumulative_returns[-1]) / self.cumulative_returns[-1])

        if self.equity <= 0 or self.current_step >= self.max_steps:
            self.done = True

        reward = self.equity - self.initial_balance

        return current_features, reward, self.done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Equity: {self.equity}")

    def _execute_order(self, price, direction):
        order = {
            'price': price,
            'direction': direction,
            'volume': self.balance * self.leverage,
            'profit': 0
        }
        self.orders.append(order)
        print(f"Opened {'Buy' if direction == 1 else 'Sell'} order at price {price} with volume {order['volume']}")

    def _update_orders(self, current_price, previous_price):
        for order in self.orders:
            if order['direction'] == 1:  # Buy
                order['profit'] = (current_price - order['price'] - self.slippage) * order['volume']
            elif order['direction'] == -1:  # Sell
                order['profit'] = (order['price'] - current_price - self.slippage) * order['volume']
            order['profit'] -= self.commission * order['volume']

            if order['profit'] >= self.params['tp'] * order['volume']:
                self.balance += order['profit']
                self.orders.remove(order)
                print(f"Closed Buy order with profit {order['profit']} at price {current_price}")
            elif order['profit'] <= -self.params['sl'] * order['volume']:
                self.balance += order['profit']
                self.orders.remove(order)
                print(f"Closed Sell order with loss {order['profit']} at price {current_price}")

    def calculate_fitness(self):
        if self.fitness_function == 'brute_profit':
            return self.equity / self.initial_balance
        elif self.fitness_function == 'sharpe_ratio':
            excess_returns = np.array(self.returns) - self.risk_free_rate
            if excess_returns.std() == 0:
                return 0
            return excess_returns.mean() / excess_returns.std()
