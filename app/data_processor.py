import pandas as pd
import numpy as np
import os
import time
import json
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from sklearn.metrics import mean_squared_error, mean_absolute_error


def process_data(config):
    print(f"Loading data from CSV file: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    print(f"Data loaded with shape: {x_train_data.shape}")

    y_train_file = config['y_train_file']
    target_column = config['target_column']

    if isinstance(y_train_file, str):
        print(f"Loading y_train data from CSV file: {y_train_file}")
        y_train_data = load_csv(y_train_file, headers=config['headers'])
        print(f"y_train data loaded with shape: {y_train_data.shape}")
    elif isinstance(y_train_file, int):
        y_train_data = x_train_data.iloc[:, y_train_file]
        print(f"Using y_train data at column index: {y_train_file}")
    elif target_column is not None:
        y_train_data = x_train_data.iloc[:, target_column]
        print(f"Using target column at index: {target_column}")
    else:
        raise ValueError("Either y_train_file or target_column must be specified in the configuration.")

    # Ensure input data is numeric
    x_train_data = x_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Apply input offset and time horizon
    offset = config['input_offset'] + config['time_horizon']
    y_train_data = y_train_data[offset:]
    x_train_data = x_train_data[:-config['time_horizon']]

    # Ensure the shapes match
    min_length = min(len(x_train_data), len(y_train_data))
    x_train_data = x_train_data[:min_length]
    y_train_data = y_train_data[:min_length]

    # Debugging messages to confirm types and shapes
    print(f"Returning data of type: {type(x_train_data)}, {type(y_train_data)}")
    print(f"x_train_data shape after adjustments: {x_train_data.shape}")
    print(f"y_train_data shape after adjustments: {y_train_data.shape}")
    
    return x_train_data, y_train_data

def run_prediction_pipeline(config, environment_plugin, agent_plugin, optimizer_plugin):
    start_time = time.time()
    
    print("Running process_data...")
    x_train, y_train = process_data(config)
    print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")

    batch_size = config['batch_size']
    epochs = config['epochs']

    # Plugin-specific parameters
    env_params = environment_plugin.plugin_params
    agent_params = agent_plugin.plugin_params
    optimizer_params = optimizer_plugin.plugin_params

    # Prepare environment
    environment_plugin.set_params(**env_params)
    environment_plugin.build_environment(x_train, y_train)

    # Prepare agent
    agent_plugin.set_params(**agent_params)

    # Prepare optimizer
    optimizer_plugin.set_params(**optimizer_params)
    optimizer_plugin.build_environment(environment_plugin.env, x_train, y_train)
    optimizer_plugin.build_model()

    # Train the model using the optimizer plugin
    optimizer_plugin.train()

    # Save the trained model
    if config['save_model']:
        optimizer_plugin.save(config['save_model'])
        agent_plugin.load(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # Predict using the trained model
    predictions = agent_plugin.predict(x_train)

    # Reshape predictions to match y_train shape
    predictions = np.array(predictions).reshape(y_train.shape)

    # Calculate fitness
    mae, mse = environment_plugin.calculate_fitness(y_train, predictions)
    print(f"Fitness: MAE={mae}, MSE={mse}")

    # Convert predictions to a DataFrame and save to CSV
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    output_filename = config['output_file']
    write_csv(output_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Output written to {output_filename}")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': float(execution_time),
        'fitness_mae': float(mae),
        'fitness_mse': float(mse)
    }

    # Save debug info
    if config.get('save_log'):
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")

    # Remote log debug info and config
    if config.get('remote_log'):
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        print(f"Debug info saved to {config['remote_log']}.")

    print(f"Execution time: {execution_time} seconds")

    # Validate the model if validation data is provided
    if config['x_validation_file'] and config['y_validation_file']:
        print("Validating model...")
        x_validation = load_csv(config['x_validation_file'], headers=config['headers']).to_numpy().astype(np.float32)
        y_validation = load_csv(config['y_validation_file'], headers=config['headers']).to_numpy().astype(np.float32)
        
        #To ensure that the optimizer integrates properly with the environment and agent plugins and performs the training and evaluation steps correctly, we need to verify that the optimizer calls the necessary functions and interfaces correctly with the provided environment and agent plugins. Here’s the corrected version of the `optimizer_plugin_openrl.py`, ensuring proper integration and handling of training and evaluation processes:

### Optimizer Plugin (`optimizer_plugin_openrl.py`)

```python
import pandas as pd
import numpy as np
import openrl
from openrl.algorithms.ppo import PPOAlgorithm as PPO
from openrl.algorithms.dqn import DQNAlgorithm as DQN

class Plugin:
    """
    An optimizer plugin using OpenRL, supporting multiple algorithms.
    """

    plugin_params = {
        'algorithm': 'PPO',
        'total_timesteps': 10000,
        'env_params': {
            'time_horizon': 12,
            'observation_space_size': 8,  # Adjust based on your x_train data
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

    def build_environment(self, environment, x_train, y_train):
        self.env = environment  # Correctly receive the environment instance
        self.env.x_train = x_train
        self.env.y_train = y_train

    def build_model(self):
        if self.params['algorithm'] == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN('MlpPolicy', self.env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.params['total_timesteps'])

    def evaluate(self):
        obs = self.env.reset()
        done = False
        rewards = []
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
        # Collect evaluation metrics
        return np.mean(rewards), np.mean(np.abs(rewards))

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        if self.params['algorithm'] == 'PPO':
            self.model = PPO.load(file_path)
        elif self.params['algorithm'] == 'DQN':
            self.model = DQN.load(file_path)


def load_and_evaluate_model(config, agent_plugin):
    # Load the model
    agent_plugin.load(config['load_model'])

    # Load the input data
    x_train, _ = process_data(config)

    # Predict using the loaded model
    predictions = agent_plugin.predict(x_train.to_numpy())

    # Save the predictions to CSV
    evaluate_filename = config['evaluate_file']
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    write_csv(evaluate_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Predicted data saved to {evaluate_filename}")
