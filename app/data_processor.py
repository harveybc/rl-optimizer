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
    # set the genome in the config variable
    config['genome'] = optimizer_plugin.current_genome 
    environment_plugin.build_environment(x_train, y_train, config)

    # Prepare agent
    agent_plugin.set_params(**agent_params)

    # Prepare optimizer
    optimizer_plugin.set_params(**optimizer_params)
    optimizer_plugin.set_environment(environment_plugin.env)
    optimizer_plugin.set_agent(agent_plugin)

    neat_config = optimizer_plugin.train(config['epochs'])


    # Save the trained model
    if config['save_model']:
        optimizer_plugin.save(config['save_model'])
        agent_plugin.load(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # Show trades and calculate fitness for the best genome
    fitness = optimizer_plugin.evaluate_genome(optimizer_plugin.best_genome, 0, agent_plugin.config, verbose=False)
    training_fitness = fitness
    print(f"Training Fitness: {training_fitness}")
    training_outputs = optimizer_plugin.outputs
    


    # Validate the model if validation data is provided
    if config['x_validation_file'] and config['y_validation_file']:
        print("Validating model...")
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
        optimizer_plugin.set_environment(environment_plugin.env)
        optimizer_plugin.set_agent(agent_plugin)

        # Calculate fitness for the best genome using the same method as in training
        validation_fitness = optimizer_plugin.evaluate_genome(optimizer_plugin.best_genome, 0, agent_plugin.config, verbose=True)
        validation_outputs = optimizer_plugin.outputs

        # Print the final balance and fitness
        print(f"*****************************************************************")
        print(f"TRAINING FITNESS: {training_fitness}")
        print(f"VALIDATION FITNESS: {validation_fitness}")
        print(f"*****************************************************************")
        # Print complexity
        kolmogorov_c = optimizer_plugin.kolmogorov_complexity(optimizer_plugin.best_genome)
        print(f"Kolmogorov Complexity: {kolmogorov_c}")
        print(f"*****************************************************************")
        # Print training information for input and output
        # calculate the total input training information y_train 
        training_input_information = shannon_hartley_information(y_train, config['periodicity_minutes'])
        print(f"Training Input Information: {training_input_information}")
        # calculate the total training_outputs information
        training_output_information = shannon_hartley_information(training_outputs, config['periodicity_minutes'])
        print(f"Training Output Information: {training_output_information}")
        print(f"*****************************************************************")
        # Print validation information for input and output
        # calculate the total input validation information y_validation
        input_information_validation = shannon_hartley_information(y_validation, config['periodicity_minutes'])
        print(f"Validation Input Information: {input_information_validation}")
        # calculate total validation_outputs information
        output_information_validation = shannon_hartley_information(validation_outputs, config['periodicity_minutes'])
        print(f"Validation Output Information: {output_information_validation}")
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
        genome_bytes = pickle.dumps(genome)
        # Compress the genome
        compressed_data = zlib.compress(genome_bytes)
        # Return the length of the compressed data as an estimate of Kolmogorov complexity
        return len(compressed_data)

def shannon_hartley_information(input, period_minutes):
    # Convertir el input a un arreglo de NumPy si es necesario
    if isinstance(input, pd.DataFrame):
        np_input = input.to_numpy()
    elif isinstance(input, list):
        # Convertir la lista a un arreglo de NumPy
        np_input = np.array(input)
    else:
        np_input = input
    
    # Verificar que np_input es ahora un arreglo de NumPy
    if not isinstance(np_input, np.ndarray):
        raise ValueError("The input must be a pandas DataFrame, a list of lists, or a NumPy array.")
    # normaliza cada columna entre 0 y 1
    min_vals = np.min(np_input, axis=0)
    max_vals = np.max(np_input, axis=0)
    np_input = (np_input - min_vals) / (max_vals - min_vals)

    # print input shape
    print(f"Shape: {np_input.shape}")

    # Concatenar las columnas verticalmente
    input_concat = np.concatenate(np_input, axis=0)
    
    # print concatenated shape
    print(f"Concat Shape: {input_concat.shape}")
    
    # Calcular la media y desviación estándar del input concatenado
    input_mean = np.mean(input_concat)
    input_std = np.std(input_concat)
    
    # Calcular SNR como (mean/std)^2
    input_SNR = (input_mean / input_std) ** 2
    
    # Calcular la frecuencia de muestreo en Hz
    sampling_frequency = 1 / (period_minutes * 60)
    
    # Calcular la capacidad total en bits por segundo con la fórmula de Shannon-Hartley
    input_capacity = sampling_frequency * np.log2(1 + input_SNR)
    
    # Calcular la información total de entrada en bits multiplicando la capacidad por el tiempo total en segundos
    input_information = input_capacity * len(input_concat)
    
    return input_information



      

    


    

