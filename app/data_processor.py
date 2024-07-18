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

    # Ensure input data is numeric except for the first column of x_train assumed to contain the date
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

    # Plugin-specific parameters
    env_params = environment_plugin.plugin_params
    agent_params = agent_plugin.plugin_params
    optimizer_params = optimizer_plugin.plugin_params

    # Prepare environment
    environment_plugin.set_params(**env_params)
    environment_plugin.build_environment(x_train, y_train, config)

    # Prepare agent
    agent_plugin.set_params(**agent_params)

    # Prepare optimizer
    optimizer_plugin.set_params(**optimizer_params)
    optimizer_plugin.set_environment(environment_plugin.env)
    optimizer_plugin.set_agent(agent_plugin)
    optimizer_plugin.train(config['epochs'])
    
    # Save the trained model
    if config['save_model']:
        optimizer_plugin.save(config['save_model'])
        agent_plugin.load(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # Show trades and calculate fitness for the best genome
    fitness = optimizer_plugin.evaluate_genome(optimizer_plugin.best_genome, 0, agent_plugin.config, verbose=True)
    print(f"Fitness: {fitness}")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': float(execution_time),
        'fitness': float(fitness)
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
        x_validation, y_validation = process_validation_data(config)
        
        print(f"x_validation shape: {x_validation.shape}")
        print(f"y_validation shape: {y_validation.shape}")
        
        validation_predictions = agent_plugin.decide_action(x_validation)
        validation_predictions = np.array(validation_predictions).reshape(y_validation.shape)
        
        validation_fitness = environment_plugin.calculate_fitness(y_validation, validation_predictions)
        print(f"Validation Fitness: {validation_fitness}")

def process_validation_data(config):
    print(f"Loading validation data from CSV file: {config['x_validation_file']}")
    x_validation_data = load_csv(config['x_validation_file'], headers=config['headers'])
    print(f"Validation data loaded with shape: {x_validation_data.shape}")

    y_validation_file = config['y_validation_file']
    target_column = config['target_column']

    if isinstance(y_validation_file, str):
        print(f"Loading y_validation data from CSV file: {y_validation_file}")
        y_validation_data = load_csv(y_validation_file, headers=config['headers'])
        print(f"y_validation data loaded with shape: {y_validation_data.shape}")
    elif isinstance(y_validation_file, int):
        y_validation_data = x_validation_data.iloc[:, y_validation_file]
        print(f"Using y_validation data at column index: {y_validation_file}")
    elif target_column is not None:
        y_validation_data = x_validation_data.iloc[:, target_column]
        print(f"Using target column at index: {target_column}")
    else:
        raise ValueError("Either y_validation_file or target_column must be specified in the configuration.")

    # Ensure input data is numeric except for the first column of x_train assumed to contain the date
    y_validation_data = y_validation_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Apply input offset and time horizon
    offset = config['input_offset'] + config['time_horizon']
    y_validation_data = y_validation_data[offset:]
    x_validation_data = x_validation_data[:-config['time_horizon']]

    # Ensure the shapes match
    min_length = min(len(x_validation_data), len(y_validation_data))
    x_validation_data = x_validation_data[:min_length]
    y_validation_data = y_validation_data[:min_length]

    # Debugging messages to confirm types and shapes
    print(f"Returning validation data of type: {type(x_validation_data)}, {type(y_validation_data)}")
    print(f"x_validation_data shape after adjustments: {x_validation_data.shape}")
    print(f"y_validation_data shape after adjustments: {y_validation_data.shape}")
    
    return x_validation_data, y_validation_data

def load_and_evaluate_model(config, agent_plugin):
    # Load the model
    agent_plugin.load(config['load_model'])

    # Load the input data
    x_train, _ = process_data(config)

    # Predict using the loaded model
    predictions = agent_plugin.decide_action(x_train.to_numpy())

    # Save the predictions to CSV
    evaluate_filename = config['evaluate_file']
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    write_csv(evaluate_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Predicted data saved to {evaluate_filename}")
