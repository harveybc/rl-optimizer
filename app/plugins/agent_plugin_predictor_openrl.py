import gym
import numpy as np
import openrl

class Plugin:
    """
    An agent plugin that uses OpenRL for optimizing predictions.
    """

    plugin_params = {
        'algorithm': 'PPO',
        'policy': 'MlpPolicy',
        'total_timesteps': 10000
    }

    plugin_debug_vars = ['algorithm', 'policy', 'total_timesteps']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.agent = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def train(self, env):
        self.agent = openrl.create(
            self.params['algorithm'],
            env,
            policy=self.params['policy']
        )
        self.agent.learn(total_timesteps=self.params['total_timesteps'])

    def save(self, file_path):
        self.agent.save(file_path)
        print(f"Agent saved to {file_path}")

    def load(self, file_path):
        self.agent = openrl.load(file_path)
        print(f"Agent loaded from {file_path}")

    def get_action(self, observation):
        action, _states = self.agent.predict(observation, deterministic=True)
        return action

# Debugging usage example
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    plugin = Plugin()
    plugin.set_params(algorithm='PPO', policy='MlpPolicy', total_timesteps=10000)
    plugin.train(env)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
    plugin.save('agent_model.zip')
