import datetime
import neat
import numpy as np
import pickle

class Plugin:
    """
    A dummy agent plugin for making predefined actions at specific dates and validating balance updates.
    """

    plugin_params = {
        'config_file': 'tests/data/neat_50.ini',
        'genome_file': 'winner.pkl'
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.config = None
        self.last_order_status = 0
        self.order_open_price = 0.0
        self.actions = {
            "buy": [
                datetime.datetime(2010, 4, 16, 20, 0),
                datetime.datetime(2010, 4, 20, 2, 0),
                datetime.datetime(2010, 4, 26, 0, 0),
                datetime.datetime(2010, 4, 30, 18, 0)
            ],
            "sell": [
                datetime.datetime(2010, 4, 19, 1, 0),
                datetime.datetime(2010, 4, 21, 4, 0),
                datetime.datetime(2010, 4, 23, 12, 0),
                datetime.datetime(2010, 4, 30, 9, 0)
            ]
        }

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def load(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Agent model loaded from {model_path}")

    def load_config(self, config_file):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)
        print(f"Config loaded from {config_file}")

    def set_model(self, genome, config):
        self.config = config  # Ensure config is set for the network creation
        self.model = neat.nn.FeedForwardNetwork.create(genome, self.config)

    def predict(self, observation, info=None):
        current_date = datetime.datetime.strptime(info["date"], '%Y-%m-%d %H:%M:%S')
        action = 0  # Default to hold
        
        # Check for buy or sell actions at specific dates
        if current_date in self.actions["buy"]:
            action = 1
            print(f"BUY Order opened at {current_date}. Current balance: {info['balance']}, Desired balance: {desired_balance}")
        elif current_date in self.actions["sell"]:
            action = 2
            print(f"SELL Order opened at {current_date}. Current balance: {info['balance']}, Desired balance: {desired_balance}")
        
        # Print balance comparison when an order is closed
        if self.last_order_status != 0 and info["order_status"] == 0:
            close_price = info["close"]
            if self.last_order_status == 1:  # Closing a buy order
                desired_balance = self.last_balance + (close_price - self.order_open_price) * info["order_volume"] * 100000
                print(f"BUY Order CLOSED at {current_date}. Current balance: {info['balance']}, Desired balance: {desired_balance}")
            elif self.last_order_status == -1:  # Closing a sell order
                desired_balance = self.last_balance + (self.order_open_price - (close_price + info["spread"])) * info["order_volume"] * 100000
                print(f"SELL Order CLOSED at {current_date}. Current balance: {info['balance']}, Desired balance: {desired_balance}")
            
            
        
        # Update the last order status and balance for next step
        self.last_order_status = info["order_status"]
        if info["order_status"] != 0:
            self.order_open_price = info["close"]
            self.last_balance = info["balance"]
        
        return action

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Agent model saved to {model_path}")
