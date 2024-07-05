import neat
import pickle

class Plugin:
    """
    An agent plugin using a trained NEAT model for making predictions.
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
            loaded_model = pickle.load(f)
        if hasattr(loaded_model, 'model'):
            self.model = loaded_model.model
        else:
            raise ValueError("Loaded model does not have the 'model' attribute.")
        print(f"Agent model loaded from {file_path}")

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model has not been loaded.")
        predictions = []
        for i in range(len(data)):
            observation = data.iloc[i].to_numpy()
            action = self.model.activate(observation)
            predictions.append(action)
        return predictions

# Debugging usage example
if __name__ == "__main__":
    agent = Plugin()
    agent.load('winner.pkl')
    debug_info = agent.get_debug_info()
    print(f"Debug Info: {debug_info}")
