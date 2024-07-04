import neat
import pickle

class Plugin:
    """
    An agent plugin using NEAT for predictions.
    """

    plugin_params = {
        'config_file': 'neat_config.ini'
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model has not been loaded.")
        predictions = []
        for i in range(len(data)):
            observation = data[i]
            action = self.model.activate(observation)
            predictions.append(action)
        return predictions

    def save(self, file_path):
        if self.model is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Agent model saved to {file_path}")
        else:
            print("No model to save.")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Agent model loaded from {file_path}")

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
