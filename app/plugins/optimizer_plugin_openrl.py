from openrl.algorithms import PPO, DQN
from app.plugins.environment_plugin_prediction import CustomPredictionEnv

class Plugin:
    """
    An optimizer plugin using OpenRL, supporting multiple algorithms.
    """

    plugin_params = {
        'algorithm': 'PPO',
        'total_timesteps': 10000,
        'env_params': {
            'time_horizon': 12,
            'observation_space_size': 10,
            'action_space_size': 1,
        }
    }

    plugin_debug_vars = ['algorithm', 'total_timesteps']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.env = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self):
        self.env = CustomPredictionEnv(**self.params['env_params'])
        if self.params['algorithm'] == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN('MlpPolicy', self.env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.params['total_timesteps'])

    def evaluate(self):
        # Evaluation logic
        pass

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        if self.params['algorithm'] == 'PPO':
            self.model = PPO.load(file_path)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN.load(file_path)
