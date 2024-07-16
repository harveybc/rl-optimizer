import neat
import pickle
import numpy as np

class Plugin:
    """
    An agent plugin for making predictions using a NEAT model.
    """

    plugin_params = {
        'config_file': 'tests/data/neat_50.ini',
        'genome_file': 'winner.pkl'
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.config = None

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

    def load_genome_and_config(self, genome, config):
        self.model = genome
        self.config = config

    def predict(self, observation):
        if self.model is None:
            raise ValueError("Model has not been loaded.")
        if self.config is None:
            raise ValueError("Config has not been loaded.")

        net = neat.nn.FeedForwardNetwork.create(self.model, self.config)
        action_values = net.activate(observation)
        action = np.argmax(action_values)  # Choose the action with the highest value
        return action

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Agent model saved to {model_path}")

# Debugging usage example
if __name__ == "__main__":
    agent = Plugin()
    agent.set_params(config_file='neat_config.ini')
    agent.load('trained_model.pkl')
    agent.load_config('neat_config.ini')
    # Example data for prediction
    import pandas as pd
    test_data = pd.DataFrame([[0.5] * 8, [0.2] * 8])
    predictions = [agent.predict(row.values) for _, row in test_data.iterrows()]
    print(f"Predictions: {predictions}")
