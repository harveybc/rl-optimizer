import gym
import numpy as np
import pandas as pd
from collections import deque

class Plugin:
    """
    An environment plugin for forex trading automation tasks, compatible with both NEAT and OpenRL.
    """

    plugin_params = {
        'initial_balance': 10000,
        'max_steps': 500,
        'fitness_function': 'brute_profit',  # 'sharpe_ratio' can be another option
        'min_orders': 4,
        'sl': 0.001,
        'tp': 0.001,
        'rel_volume': 0.1,
        'leverage': 1,
        'pip_cost': 0.0001,
        'min_order_time': 2,
        'spread': 0.001  # Default spread value
    }

    plugin_debug_vars = ['initial_balance', 'max_steps', 'fitness_function', 'final_balance', 'final_fitness']

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
        self.min_orders = config.get('min_orders', self.params['min_orders'])
        self.sl = config.get('sl', self.params['sl'])
        self.tp = config.get('tp', self.params['tp'])
        self.rel_volume = config.get('rel_volume', self.params['rel_volume'])
        self.leverage = config.get('leverage', self.params['leverage'])
        self.pip_cost = config.get('pip_cost', self.params['pip_cost'])
        self.min_order_time = config.get('min_order_time', self.params['min_order_time'])
        self.spread = config.get('spread', self.params['spread'])
        self.env = AutomationEnv(x_train, y_train, self.initial_balance, self.max_steps, self.fitness_function,
                                 self.min_orders, self.sl, self.tp, self.rel_volume, self.leverage, self.pip_cost, self.min_order_time, self.spread)

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
    def __init__(self, x_train, y_train, initial_balance, max_steps, fitness_function,
                 min_orders, sl, tp, rel_volume, leverage, pip_cost, min_order_time, spread):
        super(AutomationEnv, self).__init__()
        self.max_steps = max_steps
        self.x_train = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else x_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        # for equity variation calculus
        self.balance_ant = self.balance
        self.equity_ant = self.balance
        self.current_step = 0
        self.order_status = 0  # 0 = no order, 1 = buy, -1 = sell
        self.order_price = 0.0
        self.order_volume = 0.0
        self.done = False
        self.reward = 0.0
        self.equity_curve = [initial_balance]
        self.min_orders = min_orders
        self.sl = sl
        self.tp = tp
        self.rel_volume = rel_volume
        self.leverage = leverage
        self.pip_cost = pip_cost
        self.min_order_time = min_order_time
        self.spread = spread
        self.margin = 0.0
        self.order_time = 0

        if y_train is None:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.x_train.shape[1],), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.y_train.shape[1],), dtype=np.float32)

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
        observation = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]
        return observation

    def step(self, action):
        if self.done:
            return np.zeros(self.x_train.shape[1]), self.reward, self.done, {}

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Read time variables from CSV (Format: 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<num_columns>)
        High = self.x_train[self.current_step, 3]
        Low = self.x_train[self.current_step, 2]
        Close = self.x_train[self.current_step, 4]

        # Calculate profit
        self.profit_pips = 0
        self.real_profit = 0
        # Calculate for existing BUY order (status=1)
        if self.order_status == 1:
            self.profit_pips = ((Low - self.order_price) / self.pip_cost)
            self.real_profit = self.profit_pips * self.pip_cost * self.order_volume * 100000
        # Calculate for existing SELL order (status=-1)
        if self.order_status == -1:
            self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
            self.real_profit = self.profit_pips * self.pip_cost * self.order_volume * 100000

        # Calculate equity
        self.equity = self.balance + self.real_profit

        # Verify if Margin Call
        if self.equity < self.margin:
            self.order_status = 0
            self.balance = 0.0
            self.equity = 0.0
            self.margin = 0.0
            self.done = True

        if not self.done:
            # Verify if close by SL
            if self.profit_pips <= (-1 * self.sl):
                self.order_status = 0
                self.balance = self.equity
                self.margin = 0.0

            # Verify if close by TP
            if self.profit_pips >= self.tp:
                self.order_status = 0
                self.balance = self.equity
                self.margin = 0.0

            # Executes BUY action, order status = 1
            if (self.order_status == 0) and action == 1:
                self.order_status = 1
                self.order_price = Close + self.spread
                self.order_volume = self.equity * self.rel_volume * self.leverage / 100000
                self.order_volume = max(0.01, round(self.order_volume, 2))
                self.margin += (self.order_volume * 100000 / self.leverage)
                self.order_time = self.current_step

            # Executes SELL action, order status = 1
            if (self.order_status == 0) and action == 2:
                self.order_status = -1
                self.order_price = Close
                self.order_volume = self.equity * self.rel_volume * self.leverage / 100000
                self.order_volume = max(0.01, round(self.order_volume, 2))
                self.margin += (self.order_volume * 100000 / self.leverage)
                self.order_time = self.current_step

            # Verify if minimum order time has passed before closing
            if (self.current_step - self.order_time) > self.min_order_time:
                if (self.order_status == -1) and action == 1:
                    self.order_status = 0
                    self.balance = self.equity
                    self.margin = 0.0

                if (self.order_status == 1) and action == 2:
                    self.order_status = 0
                    self.balance = self.equity
                    self.margin = 0.0

        # Calculate reward
        equity_increment = self.equity - self.equity_ant
        balance_increment = self.balance - self.balance_ant

        # Calculate reward based on fitness function (TODO: make it depending on the fitness function used)
        bonus = ((self.equity - self.initial_balance) / self.num_ticks)
        reward = (balance_increment + bonus) / 2
        if (self.num_closes < self.min_orders / 2) and reward > 0:
            reward = reward * (self.num_closes / self.min_orders)
        if (self.num_closes < self.min_orders / 2) and reward <= 0:
            reward = reward - (self.initial_balance / self.num_ticks) * (1 - (self.num_closes / self.min_orders))
        if (self.num_closes < self.min_orders) and reward <= 0:
            reward = reward - ((self.initial_balance / (10 * self.num_ticks)) * (1 - (self.num_closes / self.min_orders)))
        if self.c_c == 1:
            reward = -(5.0 * self.initial_balance)
        if self.tick_count >= (self.num_ticks - 2):
            if self.num_closes < self.min_orders:
                reward = -(10 * self.initial_balance * (1 - (self.num_closes / self.min_orders)))
                self.balance = 0
                self.equity = 0
            if self.equity == self.initial_balance:
                reward = -(10.0 * self.initial_balance)
                self.balance = 0
                self.equity = 0
        reward = reward / self.initial_balance

        # Push values from timeseries into state (assumes all values are already normalized)
        for i in range(0, self.num_columns - 1):
            # verify of y_train is none
            if self.y_train is not None:
                obs_normalized = self.y_train[self.current_step, i]
            else:
                obs_normalized = self.x_train[self.current_step, i]
            self.obs_matrix[i].append(obs_normalized)

        obs_normalized = self.order_status
        self.state[0].append(obs_normalized)
        ob = np.concatenate([self.obs_matrix, self.state])

        self.tick_count += 1
        self.equity_ant = self.equity
        self.balance_ant = self.balance
        self.reward += reward

        if self.tick_count >= (self.num_ticks - 1):
            self.done = True

        info = {
            "date": self.x_train[self.current_step, 0],
            "close": self.x_train[self.current_step, 4],
            "high": self.x_train[self.current_step, 3],
            "low": self.x_train[self.current_step, 2],
            "open": self.x_train[self.current_step, 1],
            "action": action,
            "observation": ob,
            "episode_over": self.done,
            "balance": self.balance,
            "tick_count": self.tick_count,
            "num_closes": self.num_closes,
            "equity": self.equity,
            "reward": self.reward,
            "order_status": self.order_status,
            "margin": self.margin,
            "initial_capital": self.initial_balance
        }

        return ob, reward, self.done, info

    def render(self, mode='human'):
        pass

    def calculate_final_debug_vars(self):
        return {
            'final_balance': self.balance,
            'final_fitness': self.reward
        }
