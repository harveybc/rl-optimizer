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

    def set_model(self, genome, config):
        self.config = config  # Ensure config is set for the network creation
        self.model = neat.nn.FeedForwardNetwork.create(genome, self.config)

    def predict(self, observation, info=None):
        if self.model is None:
            raise ValueError("Model has not been set.")
        
        # Convert observation to a numpy array if it is not already
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        
        # Ensure observation contains only numeric data
        if not np.issubdtype(observation.dtype, np.number):
            raise ValueError("Observation contains non-numeric data")

        print(f"Observation: {observation}")  # Print observation for debugging

        action_values = self.model.activate(observation)
        #print(f"Action values: {action_values}")

        action = np.argmax(action_values)  # Get the discrete action
        return action

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Agent model saved to {model_path}")
