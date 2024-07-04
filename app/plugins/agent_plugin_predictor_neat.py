import neat
import pickle
import os

class Plugin:
    """
    An agent plugin using NEAT for evolutionary neural networks.
    """

    plugin_params = {
        'config_file': 'neat_config.ini'
    }

    plugin_debug_vars = ['config_file']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Agent model loaded from {file_path}")
        if isinstance(self.model, neat.nn.FeedForwardNetwork):
            print("Loaded model is a NEAT FeedForwardNetwork")
        else:
            print(f"Loaded model type: {type(self.model)}")

    def save(self, file_path):
        if self.model is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Agent model saved to {file_path}")
        else:
            print("No model to save.")

    def predict(self, x_data):
        if self.model is None:
            raise ValueError("Model has not been loaded.")
        predictions = []
        for data in x_data:
            action = self.model.activate(data)
            predictions.append(action)
        return predictions

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.set_params(config_file='neat_config.ini')
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
    plugin.load('winner.pkl')
    x_data_example = [[0.1] * 8]  # Example input data
    predictions = plugin.predict(x_data_example)
    print(f"Predictions: {predictions}")
