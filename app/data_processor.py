import pandas as pd
import numpy as np
import os
import time
import json
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import zlib

def process_data(config):
    print(f"Loading data from CSV file: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    print(f"Data loaded with shape: {x_train_data.shape}")

    y_train_file = config['y_train_file']

    if isinstance(y_train_file, str):
        print(f"Loading y_train data from CSV file: {y_train_file}")
        y_train_data = load_csv(y_train_file, headers=config['headers'])
        print(f"y_train data loaded with shape: {y_train_data.shape}")
    elif isinstance(y_train_file, int):
        y_train_data = x_train_data.iloc[:, y_train_file]
        print(f"Using y_train data at column index: {y_train_file}")
    else:
        raise ValueError("Either y_train_file  must be specified in the configuration.")

    # Ensure input data is numeric except for the first column of x_train assumed to contain the date
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Apply input offset and time horizon
    offset = config['input_offset']
    print(f"Applying input offset: {offset}")
    #y_train_data = y_train_data[offset:]
    x_train_data = x_train_data[offset:]
    print(f"Data shape after applying offset: {x_train_data.shape}, {y_train_data.shape}")
    # if the first dimension of x_train and y_train do not match, exit
    if len(x_train_data) != len(y_train_data):
        raise ValueError("x_train_data (market observation) and y_train_data(data observation) data shapes do not match.")

    # Ensure the shapes match
    min_length = min(len(x_train_data), len(y_train_data))
    x_train_data = x_train_data[:min_length]
    y_train_data = y_train_data[:min_length]
    # Divide the data into three parts: training, pruning, and stabilization
    third_index = min_length // 3

    x_train_data_split = x_train_data[:third_index]  # First third for training
    y_train_data_split = y_train_data[:third_index]

    x_prunning_data = x_train_data[third_index:2*third_index]  # Second third for pruning
    y_prunning_data = y_train_data[third_index:2*third_index]

    x_stabilization_data = x_train_data[2*third_index:]  # Last third for stabilization
    y_stabilization_data = y_train_data[2*third_index:]

    # Verify the sizes of each dataset after splitting
    print(f"Training data size: {len(x_train_data_split)}")
    print(f"Pruning data size: {len(x_prunning_data)}")
    print(f"Stabilization data size: {len(x_stabilization_data)}")

    if config['x_validation_file'] and config['y_validation_file']:
        print("loading Validation data...")
        x_validation = load_csv(config['x_validation_file'], headers=config['headers'])
        y_validation = load_csv(config['y_validation_file'], headers=config['headers'])

        print(f"Validation market data loaded with shape: {x_validation.shape}")
        print(f"Validation processed data loaded with shape: {y_validation.shape}")
        
        # Ensure x_validation is a 2D array
        if x_validation.ndim == 1:
            x_validation = x_validation.reshape(-1, 1)
        
        # Ensure input data is numeric except for the first column of x_train assumed to contain the date
        y_validation = y_validation.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_validation = x_validation.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Apply the  input_offset to the x validation data
        x_validation = x_validation[config['input_offset']:]
        
        print(f"x_validation shape: {x_validation.shape}")
        print(f"y_validation shape: {y_validation.shape}")
        # if sizes do not match, exit
        if len(x_validation) != len(y_validation):
            raise ValueError("x_validation and y_validation data shapes do not match.")

    

    # Debugging messages to confirm types and shapes
    print(f"Returning data of type: {type(x_train_data)}, {type(y_train_data)}")
    print(f"x_train_data shape after adjustments: {x_train_data_split.shape}")
    print(f"y_train_data shape after adjustments: {x_train_data_split.shape}")
    print(f"x_prunning_data shape: {x_prunning_data.shape}")
    print(f"y_prunning_data shape: {y_prunning_data.shape}")
    print(f"x_validation_data shape after adjustments: {x_validation.shape}")
    print(f"y_validation_data shape after adjustments: {y_validation.shape}")
    print(f"x_stabilization_data shape: {x_stabilization_data.shape}")
    print(f"y_stabilization_data shape: {y_stabilization_data.shape}")

    # if any of the data to be returned is zero, exit with erro showing the exact dataset that have zero size
    if len(x_train_data) == 0:
        raise ValueError("x_train_data is empty.")
    if len(y_train_data) == 0:
        raise ValueError("y_train_data is empty.")
    if len(x_prunning_data) == 0:
        raise ValueError("x_prunning_data is empty.")
    if len(y_prunning_data) == 0:
        raise ValueError("y_prunning_data is empty.")
    if len(x_stabilization_data) == 0:
        raise ValueError("x_stabilization_data is empty.")
    if len(y_stabilization_data) == 0:
        raise ValueError("y_stabilization_data is empty.")
    if config['x_validation_file'] and config['y_validation_file']:
        if len(x_validation) == 0:
            raise ValueError("x_validation is empty.")
        if len(y_validation) == 0:
            raise ValueError("y_validation is empty.")
    
    return x_train_data_split, y_train_data_split, x_prunning_data, y_prunning_data, x_validation, y_validation, x_stabilization_data, y_stabilization_data

def run_prediction_pipeline(config, environment_plugin, agent_plugin, optimizer_plugin):
    start_time = time.time()
    
    print("Running process_data...")
    x_train, y_train, x_prunning, y_prunning, x_validation, y_validation, x_stabilization, y_stabilization = process_data(config)
    print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")

    # Plugin-specific parameters
    env_params = environment_plugin.plugin_params
    agent_params = agent_plugin.plugin_params
    optimizer_params = optimizer_plugin.plugin_params

    # Prepare environment
    environment_plugin.set_params(**env_params)
    # set the genome in the config variable
    config['genome'] = optimizer_plugin.current_genome 
    environment_plugin.build_environment(x_train, y_train, config)

    # Prepare agent
    agent_plugin.set_params(**agent_params)

    # Prepare optimizer
    optimizer_plugin.set_params(**optimizer_params)
    optimizer_plugin.set_environment(environment_plugin.env, config['num_hidden'])
    optimizer_plugin.set_agent(agent_plugin)

    neat_config = optimizer_plugin.train(config['epochs'], x_train, y_train, x_stabilization, y_stabilization, x_prunning, y_prunning, x_validation,y_validation, config, environment_plugin)


    # Save the trained model
    if config['save_model']:
        optimizer_plugin.save(config['save_model'])
        agent_plugin.load(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # for the final training performance, concatenate (vertically) the training, prunning and stabilization datasets
    x_train_full = pd.concat([x_train, x_prunning, x_stabilization], axis=0)
    y_train_full = pd.concat([y_train, y_prunning, y_stabilization], axis=0)

    # sets the environment data as the training data, since the optimizer changes it to the validation data for debugging 
    environment_plugin.build_environment(x_train_full, y_train_full, config)
    optimizer_plugin.set_environment(environment_plugin.env, config['num_hidden'])

    # Show trades and calculate fitness for the best genome
    fitness, info = optimizer_plugin.evaluate_genome(optimizer_plugin.best_genome, 0, agent_plugin.config, verbose=False)
    training_fitness = fitness
    training_outputs = optimizer_plugin.outputs
    training_node_values = optimizer_plugin.node_values
    


    # Validate the model if validation data is provided
    if config['x_validation_file'] and config['y_validation_file']:
        print("Validating model...")
        print(f"x_validation shape: {x_validation.shape}")
        print(f"y_validation shape: {y_validation.shape}")
        # if sizes do not match, exit
        if len(x_validation) != len(y_validation):
            raise ValueError("x_validation and y_validation data shapes do not match.")

        # Set the model to use the best genome for evaluation
        agent_plugin.set_model(optimizer_plugin.best_genome, neat_config)
        
        environment_plugin.build_environment(x_validation, y_validation, config)
        
        observation, info = environment_plugin.reset()
        done = False
        # Initialize total_reward
        total_reward = []

        # Set the best genome for the agent
        agent_plugin.set_model(optimizer_plugin.best_genome, agent_plugin.config)

        # Set the environment and agent for the optimizer
        optimizer_plugin.set_environment(environment_plugin.env, config['num_hidden'])
        optimizer_plugin.set_agent(agent_plugin)

        # Calculate fitness for the best genome using the same method as in training
        validation_fitness, info = optimizer_plugin.evaluate_genome(optimizer_plugin.best_genome, 0, agent_plugin.config, verbose=True)
        validation_outputs = optimizer_plugin.outputs
        validation_node_values = optimizer_plugin.node_values
        # validation_outputs is a list of lists (table of 4 columns), print the first 5 files
        print(f"Validation outputs: {validation_outputs[:5]}")

        # Print the final balance and fitness
        print(f"*****************************************************************")
        print(f"TRAINING FITNESS: {training_fitness}")
        print(f"VALIDATION FITNESS: {validation_fitness}")
        print(f"*****************************************************************")
        # Print complexity
        kolmogorov_c = optimizer_plugin.kolmogorov_complexity(optimizer_plugin.best_genome)
        print(f"Kolmogorov Complexity (bits): {kolmogorov_c*8}")
        # Print number of connections of the champion genome
        num_connections = len(optimizer_plugin.best_genome.connections)
        print(f"Number of connections: {num_connections}")
        # Print number of nodes of the champion genome
        num_nodes = len(optimizer_plugin.best_genome.nodes)
        print(f"Number of nodes: {num_nodes}")
        # Convert the genome to a string representation
        genome_bytes = pickle.dumps(optimizer_plugin.best_genome)
        # print the lenght of the genome
        print(f"Genome length (bits): {len(genome_bytes)*8}")
        # Print the Shannon entropy of the weights
        weights_entropy = calculate_weights_entropy(optimizer_plugin.best_genome)
        print(f"Weights entropy (bits): {weights_entropy}")

        print(f"*****************************************************************")
        # Print training information for input and output
        # calculate the total input training information y_train 
        training_input_information = shannon_hartley_information(y_train, config['periodicity_minutes'])
        print(f"Training Input Information (bits): {training_input_information}")
        # calculate the total training_outputs information
        training_output_information = shannon_hartley_information(training_outputs, config['periodicity_minutes'])
        print(f"Training Output Information (bits): {training_output_information}")
        # calculate the total training_node_values_information
        training_node_values_information = shannon_hartley_information(training_node_values, config['periodicity_minutes'])
        print(f"Total Training Node Values Information (bits): {training_node_values_information}")
        # print the total training information as the entropy multiplied by the training_node_values_information
        #veryfy for None o NoneType
        if training_node_values_information is None:
            training_total_information = num_connections*weights_entropy
        else:
            training_total_information = num_connections*weights_entropy + training_node_values_information
        print(f"Total Training Information (bits): {training_total_information}")



        print(f"*****************************************************************")
        # Print validation information for input and output
        # calculate the total input validation information y_validation
        input_information_validation = shannon_hartley_information(y_validation, config['periodicity_minutes'])
        print(f"Validation Input Information (bits): {input_information_validation}")
        # calculate total validation_outputs information
        output_information_validation = shannon_hartley_information(validation_outputs, config['periodicity_minutes'])
        print(f"Validation Output Information (bits): {output_information_validation}")
        # calculate total validation_node_values_information
        node_values_information_validation = shannon_hartley_information(validation_node_values, config['periodicity_minutes'])
        print(f"Total Validation Node Values Information (bits): {node_values_information_validation}")
        # print the total validation information as the entropy multiplied by the node_values_information
        if node_values_information_validation is None:
            validation_total_information = num_connections*weights_entropy
        else:
            validation_total_information = num_connections*weights_entropy + node_values_information_validation
        print(f"Total Validation Information (bits): {validation_total_information}")
        print(f"*****************************************************************")
        
        # Save final configuration and debug information
        end_time = time.time()
        execution_time = end_time - start_time
        debug_info = {
            'execution_time': float(execution_time),
            'training_fitness': float(training_fitness),
            'validation_fitness': float(validation_fitness)
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

def load_and_evaluate_model(config, agent_plugin):
    # Load the model
    agent_plugin.load(config['load_model'])

    # Load the input data
    x_train, _ = process_data(config)


    predictions = agent_plugin.decide_action(pd.DataFrame(x_train.to_numpy()))

    # Save the predictions to CSV
    evaluate_filename = config['evaluate_file']
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    write_csv(evaluate_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Predicted data saved to {evaluate_filename}")


def kolmogorov_complexity(genome):
        # Convert the genome to a string representation
        #genome_connections_bytes = pickle.dumps(genome.connections)
        #genome_nodes_bytes = pickle.dumps(genome.nodes)
        #genome_bytes = genome_connections_bytes + genome_nodes_bytes
        genome_bytes = pickle.dumps(genome)
        # Compress the genome, using the highest compression level, with no header or trailing checksum
        compressed_data = zlib.compress(genome_bytes,level=9, wbits=-15)
        # Return the length of the compressed data as an estimate of Kolmogorov complexity
        return len(compressed_data)

import numpy as np
import pandas as pd

def shannon_hartley_information(input, period_minutes):
    try:
        # Convert input to NumPy array if necessary
        if isinstance(input, pd.DataFrame):
            np_input = input.to_numpy()
        elif isinstance(input, list):
            # Verify the size of each element in the list
            input_lengths = [len(i) if hasattr(i, '__len__') else 1 for i in input]
            if len(set(input_lengths)) != 1:
                print(f"Warning: Found inhomogeneous lengths in input: {input_lengths}")
                return None  # Early exit
            # Convert list to NumPy array
            np_input = np.array(input)
        else:
            np_input = input
        
        # Verify that np_input is a NumPy array
        if not isinstance(np_input, np.ndarray):
            print("Warning: The input must be a pandas DataFrame, a list of lists, or a NumPy array.")
            return None
        
        # Check if np_input has consistent dimensions
        if np_input.ndim != 2:
            print(f"Warning: Input array must be 2D (rows, columns). Got {np_input.ndim}D.")
            return None
        
        # Ensure array is not empty and all columns have data
        if np_input.shape[1] == 0:
            print("Warning: Input array must have at least one column.")
            return None
        
        # Normalize each column between 0 and 1
        min_vals = np.min(np_input, axis=0)
        max_vals = np.max(np_input, axis=0)
        
        # Check for division by zero
        if np.any(max_vals - min_vals == 0):
            print("Warning: One or more columns have constant values, which causes division by zero in normalization.")
            return None
        else:
            np_input = (np_input - min_vals) / (max_vals - min_vals)
        
        # Print input shape
        print(f"Shape: {np_input.shape}")
        
        # Concatenate columns vertically
        input_concat = np.concatenate(np_input, axis=0)
        
        # Print concatenated shape
        print(f"Concat Shape: {input_concat.shape}")
        
        # Calculate mean and standard deviation of concatenated input
        input_mean = np.mean(input_concat)
        input_std = np.std(input_concat)
        
        # Check that standard deviation is not zero (avoid division by zero)
        if input_std == 0:
            print("Warning: Standard deviation of the input is zero, cannot calculate SNR.")
            return None
        
        # Calculate SNR as (mean/std)^2
        input_SNR = (input_mean / input_std) ** 2
        
        # Calculate the sampling frequency in Hz
        sampling_frequency = 1 / (period_minutes * 60)
        
        # Calculate total capacity in bits per second using Shannon-Hartley formula
        input_capacity = sampling_frequency * np.log2(1 + input_SNR)
        
        # Calculate total input information in bits by multiplying capacity by total time in seconds
        input_information = input_capacity * len(input_concat)
        
        return input_information

    except Exception as e:
        # Catch any unexpected errors and continue
        print(f"Warning: An error occurred: {e}")
        return None



import math

def calculate_weights_entropy(genome, num_bins=50):
    """
    Calculate the Shannon entropy of the weights of a NEAT genome.
    
    Parameters:
        genome: The NEAT genome containing connection weights.
        num_bins: The number of bins to use for discretizing the weight values.

    Returns:
        entropy: The Shannon entropy of the weight distribution in bits.
    """
    # Extract the weights from the genome's connections
    weights = [conn.weight for conn in genome.connections.values() if conn.enabled]
    
    # Normalize the weights to be between 0 and 1
    min_weight = min(weights)
    max_weight = max(weights)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    
    # Create a histogram to get the probability distribution
    hist, bin_edges = np.histogram(normalized_weights, bins=num_bins, range=(0, 1), density=True)
    
    # Calculate the probabilities for each bin
    probabilities = hist / np.sum(hist)
    
    # Calculate the Shannon entropy
    entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
    
    return entropy

      

    


    

