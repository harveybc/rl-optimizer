import datetime
import numpy as np
import sys

class Plugin:
    """
    A dummy agent plugin for making predefined actions based on dates.
    """

    plugin_params = {
        'buy_dates': [
            '16/04/2010 20:00',
            '20/04/2010 02:00',
            '26/04/2010 00:00',
            '30/04/2010 18:00'
        ],
        'sell_dates': [
            '19/04/2010 01:00',
            '21/04/2010 04:00',
            '23/04/2010 12:00',
            '30/04/2010 09:00'
        ],
        'capital_risk': 0.1,
        'leverage': 100,
        'config_file': 'dummy_config.ini',
        'genome_file': 'dummy_winner.pkl'
    }

    plugin_debug_vars = ['balance', 'equity', 'num_closes']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.config = None
        self.balance = 0
        self.equity = 0
        self.num_closes = 0
        self.order_status = 0
        self.order_price = 0.0
        self.order_volume = 0.0
        self.initial_balance = 0.0
        self.spread = 0.001  # Default spread value, update as necessary
        self.pip_cost = 0.0001  # Default pip cost, update as necessary
        self.buy_dates = [datetime.datetime.strptime(date, '%d/%m/%Y %H:%M') for date in self.params['buy_dates']]
        self.sell_dates = [datetime.datetime.strptime(date, '%d/%m/%Y %H:%M') for date in self.params['sell_dates']]

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        self.buy_dates = [datetime.datetime.strptime(date, '%d/%m/%Y %H:%M') for date in self.params['buy_dates']]
        self.sell_dates = [datetime.datetime.strptime(date, '%d/%m/%Y %H:%M') for date in self.params['sell_dates']]

    def get_debug_info(self):
        return {var: getattr(self, var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def load(self, model_path):
        pass  # No model to load for dummy agent

    def load_config(self, config_file):
        pass  # No config to load for dummy agent

    def set_model(self, genome, config):
        pass  # No model to set for dummy agent

    def predict(self, observation, info=None):
        current_date = info["date"]

        action = 0  # Default to hold

        if current_date in self.buy_dates:
            action = 1  # Buy
        elif current_date in self.sell_dates:
            action = 2  # Sell

        # Opening order
        if info["order_status"] == 0 and action != 0:
            self.order_status = action
            self.order_price = info["close"]
            self.order_volume = info["equity"] * self.params['capital_risk'] * self.params['leverage']
            print(f"{current_date} - Opening order - Action: {'Buy' if action == 1 else 'Sell'}, Price: {self.order_price}, Volume: {self.order_volume}")
            print(f"Current balance: {info['balance']}, Equity: {info['equity']}, Number of closes: {info['num_closes']}")

        # Calculate the desired balance when closing an order
        if info["order_status"] == 0 and self.order_status != 0:
            if self.order_status == 1:  # Closing a buy order
                profit_pips = ((info["close"] - self.order_price) / self.pip_cost) - self.spread
            elif self.order_status == 2:  # Closing a sell order
                profit_pips = ((self.order_price - info["close"]) / self.pip_cost) - self.spread
            else:
                profit_pips = 0.0

            real_profit = profit_pips * self.pip_cost * self.order_volume
            desired_balance = self.initial_balance + real_profit

            print(f"Calculating Real Profit: profit_pips: {profit_pips}, pip_cost: {self.pip_cost}, order_volume: {self.order_volume}")
            print(f"{current_date} - Closed order - Action: {'Buy' if self.order_status == 1 else 'Sell'}, Close Price: {info['close']}, Spread: {self.spread}")
            print(f"Profit pips: {profit_pips}, Profit: {real_profit}")
            print(f"Initial balance: {self.initial_balance}, Real Profit: {real_profit}, Order Volume: {self.order_volume}, Pip Cost: {self.pip_cost}")
            print(f"New balance: {info['balance']}, Expected balance: {desired_balance}, Equity: {info['equity']}, Number of closes: {info['num_closes']}")

            if desired_balance != info['balance']:
                print("Error: Balance mismatch! Exiting.")
                sys.exit(1)

            # Reset order status
            self.order_status = 0

        return action

    def save(self, model_path):
        pass  # No model to save for dummy agent
