import datetime
import numpy as np

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

        # Print balance and equity information for verification
        if info["order_status"] == 0 and action != 0:
            self.balance = info['balance']
            self.equity = info['equity']
            self.num_closes = info['num_closes']
            print(f"Opening order - Current balance: {self.balance}, Equity: {self.equity}, Number of closes: {self.num_closes}")

        # Check for closed orders and print verification
        if info["order_status"] == 0 and self.balance != info['balance']:
            print(f"Closed order - New balance: {info['balance']}, Expected balance: {self.balance}, Equity: {info['equity']}, Number of closes: {info['num_closes']}")

        return action

    def save(self, model_path):
        pass  # No model to save for dummy agent
