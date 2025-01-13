# app/data_processor.py

import pandas as pd
import numpy as np
import os
import time
import json
import pickle
from typing import Tuple, Optional

from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.utils import (
    kolmogorov_complexity,
    shannon_hartley_information,
    calculate_weights_entropy
)
from app.logger import get_logger

logger = get_logger(__name__)

def process_data(config: dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
                                       Optional[pd.DataFrame], Optional[pd.Series],
                                       pd.DataFrame, pd.Series]:
    """
    Processes the input data for the optimization pipeline.

    This function loads training and validation data, applies offsets, splits the data
    into training, pruning, and stabilization datasets, and ensures data integrity.

    Parameters
    ----------
    config : dict
        Configuration parameters.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[pd.DataFrame],
          Optional[pd.Series], pd.DataFrame, pd.Series]
        Processed datasets:
        - x_train_data_split, y_train_data_split
        - x_prunning_data, y_prunning_data
        - x_validation, y_validation
        - x_stabilization_data, y_stabilization_data

    Raises
    ------
    ValueError
        If data shapes do not match or required data is missing.
    """
    logger.info(f"Loading data from CSV file: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    logger.info(f"Data loaded with shape: {x_train_data.shape}")

    y_train_file = config['y_train_file']

    if isinstance(y_train_file, str):
        logger.info(f"Loading y_train data from CSV file: {y_train_file}")
        y_train_data = load_csv(y_train_file, headers=config['headers'])
        logger.info(f"y_train data loaded with shape: {y_train_data.shape}")
    elif isinstance(y_train_file, int):
        y_train_data = x_train_data.iloc[:, y_train_file]
        logger.info(f"Using y_train data at column index: {y_train_file}")
    else:
        logger.error("Invalid type for y_train_file in configuration.")
        raise ValueError("Either y_train_file must be specified as a string (file path) or as an integer (column index).")

    # Ensure y_train_data is numeric
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Apply input offset and time horizon
    offset = config['input_offset']
    logger.info(f"Applying input offset: {offset}")
    x_train_data = x_train_data[offset:]
    logger.info(f"Data shape after applying offset: {x_train_data.shape}, {y_train_data.shape}")

    # Ensure x_train_data and y_train_data have the same length
    if len(x_train_data) != len(y_train_data):
        logger.warning("x_train_data and y_train_data shapes do not match after applying offset.")
        min_length = min(len(x_train_data), len(y_train_data))
        x_train_data = x_train_data[:min_length]
        y_train_data = y_train_data[:min_length]
        logger.info(f"Data trimmed to minimum length: {min_length}")

    # Split the data into three parts: training, pruning, and stabilization
    third_index = len(x_train_data) // 3

    x_train_data_split = x_train_data[:third_index]  # First third for training
    y_train_data_split = y_train_data[:third_index]

    x_prunning_data = x_train_data[third_index:2*third_index]  # Second third for pruning
    y_prunning_data = y_train_data[third_index:2*third_index]

    x_stabilization_data = x_train_data[2*third_index:]  # Last third for stabilization
    y_stabilization_data = y_train_data[2*third_index:]

    logger.info(f"Training data size: {len(x_train_data_split)}")
    logger.info(f"Pruning data size: {len(x_prunning_data)}")
    logger.info(f"Stabilization data size: {len(x_stabilization_data)}")

    # Initialize validation data as None
    x_validation = None
    y_validation = None

    if config.get('x_validation_file') and config.get('y_validation_file'):
        logger.info("Loading validation data...")
        x_validation = load_csv(config['x_validation_file'], headers=config['headers'])
        y_validation = load_csv(config['y_validation_file'], headers=config['headers'])

        logger.info(f"Validation market data loaded with shape: {x_validation.shape}")
        logger.info(f"Validation processed data loaded with shape: {y_validation.shape}")

        # Ensure x_validation is a 2D array
        if x_validation.ndim == 1:
            x_validation = x_validation.to_frame()

        # Ensure y_validation is numeric
        y_validation = y_validation.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_validation = x_validation.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Apply the input_offset to the x validation data
        x_validation = x_validation[config['input_offset']:]

        logger.info(f"x_validation shape: {x_validation.shape}")
        logger.info(f"y_validation shape: {y_validation.shape}")

        # Ensure x_validation and y_validation have the same length
        if len(x_validation) != len(y_validation):
            logger.error("x_validation and y_validation data shapes do not match.")
            raise ValueError("x_validation and y_validation data shapes do not match.")

    # Debugging messages to confirm types and shapes
    logger.debug(f"Returning data types: {type(x_train_data_split)}, {type(y_train_data_split)}, "
                 f"{type(x_prunning_data)}, {type(y_prunning_data)}, "
                 f"{type(x_validation)}, {type(y_validation)}, "
                 f"{type(x_stabilization_data)}, {type(y_stabilization_data)}")
    logger.info(f"x_train_data shape after adjustments: {x_train_data_split.shape}")
    logger.info(f"y_train_data shape after adjustments: {y_train_data_split.shape}")
    logger.info(f"x_prunning_data shape: {x_prunning_data.shape}")
    logger.info(f"y_prunning_data shape: {y_prunning_data.shape}")
    if config.get('x_validation_file') and config.get('y_validation_file'):
        logger.info(f"x_validation shape after adjustments: {x_validation.shape}")
        logger.info(f"y_validation shape after adjustments: {y_validation.shape}")
    logger.info(f"x_stabilization_data shape: {x_stabilization_data.shape}")
    logger.info(f"y_stabilization_data shape: {y_stabilization_data.shape}")

    # Check for empty datasets
    if len(x_train_data_split) == 0:
        logger.error("x_train_data is empty.")
        raise ValueError("x_train_data is empty.")
    if len(y_train_data_split) == 0:
        logger.error("y_train_data is empty.")
        raise ValueError("y_train_data is empty.")
    if len(x_prunning_data) == 0:
        logger.error("x_prunning_data is empty.")
        raise ValueError("x_prunning_data is empty.")
    if len(y_prunning_data) == 0:
        logger.error("y_prunning_data is empty.")
        raise ValueError("y_prunning_data is empty.")
    if len(x_stabilization_data) == 0:
        logger.error("x_stabilization_data is empty.")
        raise ValueError("x_stabilization_data is empty.")
    if len(y_stabilization_data) == 0:
        logger.error("y_stabilization_data is empty.")
        raise ValueError("y_stabilization_data is empty.")
    if config.get('x_validation_file') and config.get('y_validation_file'):
        if len(x_validation) == 0:
            logger.error("x_validation is empty.")
            raise ValueError("x_validation is empty.")
        if len(y_validation) == 0:
            logger.error("y_validation is empty.")
            raise ValueError("y_validation is empty.")

    return (
        x_train_data_split,
        y_train_data_split,
        x_prunning_data,
        y_prunning_data,
        x_validation,
        y_validation,
        x_stabilization_data,
        y_stabilization_data
    )

def run_prediction_pipeline(config: dict, environment_plugin, agent_plugin, optimizer_plugin) -> None:
    """
    Runs the prediction pipeline based on the provided configuration and plugins.

    This function orchestrates the entire pipeline:
    - Processes the data.
    - Initializes and trains the optimizer and agent.
    - Evaluates the trained model.
    - Validates the model if validation data is provided.
    - Saves debug information locally or remotely.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    environment_plugin : object
        The loaded environment plugin.
    agent_plugin : object
        The loaded agent plugin.
    optimizer_plugin : object
        The loaded optimizer plugin.

    Raises
    ------
    Exception
        If any step in the pipeline fails.
    """
    start_time = time.time()
    logger.info("Running process_data...")
    try:
        x_train, y_train, x_prunning, y_prunning, x_validation, y_validation, x_stabilization, y_stabilization = process_data(config)
    except Exception as e:
        logger.error(f"Error during process_data: {e}")
        raise

    logger.info(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")

    # Plugin-specific parameters
    env_params = getattr(environment_plugin, 'plugin_params', {})
    agent_params = getattr(agent_plugin, 'plugin_params', {})
    optimizer_params = getattr(optimizer_plugin, 'plugin_params', {})

    # Prepare environment
    logger.info("Setting environment plugin parameters.")
    try:
        environment_plugin.set_params(**env_params)
    except Exception as e:
        logger.error(f"Error setting environment plugin parameters: {e}")
        raise

    # Set the genome in the config variable
    config['genome'] = optimizer_plugin.current_genome

    logger.info("Building environment with training data.")
    try:
        environment_plugin.build_environment(x_train, y_train, config)
    except Exception as e:
        logger.error(f"Error building environment: {e}")
        raise

    # Prepare agent
    logger.info("Setting agent plugin parameters.")
    try:
        agent_plugin.set_params(**agent_params)
    except Exception as e:
        logger.error(f"Error setting agent plugin parameters: {e}")
        raise

    # Prepare optimizer
    logger.info("Setting optimizer plugin parameters.")
    try:
        optimizer_plugin.set_params(**optimizer_params)
        optimizer_plugin.set_environment(environment_plugin.env, config.get('num_hidden', 0))
        optimizer_plugin.set_agent(agent_plugin)
    except Exception as e:
        logger.error(f"Error setting optimizer plugin parameters: {e}")
        raise

    logger.info(f"Max steps: {config.get('max_steps', 0)}")
    
    try:
        logger.info("Starting optimizer training.")
        neat_config = optimizer_plugin.train(
            config.get('epochs', 0),
            x_train,
            y_train,
            x_stabilization,
            y_stabilization,
            x_prunning,
            y_prunning,
            x_validation,
            y_validation,
            config,
            environment_plugin
        )
    except Exception as e:
        logger.error(f"Error during optimizer training: {e}")
        raise

    # Save the trained model
    if config.get('save_model'):
        try:
            logger.info(f"Saving model to: {config['save_model']}")
            optimizer_plugin.save(config['save_model'])
            agent_plugin.load(config['save_model'])
            logger.info(f"Model saved to {config['save_model']}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    # Concatenate training, pruning, and stabilization datasets
    logger.info("Concatenating training, pruning, and stabilization datasets.")
    x_train_full = pd.concat([x_train, x_prunning, x_stabilization], axis=0)
    y_train_full = pd.concat([y_train, y_prunning, y_stabilization], axis=0)

    # Update configuration for full training
    temp_config = config.copy()
    temp_config['max_steps'] = config.get('max_steps', 0) * 3
    logger.info("Building environment with full training data.")
    try:
        environment_plugin.build_environment(x_train_full, y_train_full, temp_config)
        optimizer_plugin.set_environment(environment_plugin.env, temp_config.get('num_hidden', 0))
    except Exception as e:
        logger.error(f"Error building environment with full training data: {e}")
        raise

    # Evaluate the best genome
    logger.info("Evaluating the best genome on training data.")
    try:
        fitness, info = optimizer_plugin.evaluate_genome(optimizer_plugin.best_genome, 0, agent_plugin.config, verbose=False)
        training_fitness = fitness
        training_outputs = optimizer_plugin.outputs
        training_node_values = optimizer_plugin.node_values
    except Exception as e:
        logger.error(f"Error evaluating best genome: {e}")
        raise

    # Validate the model if validation data is provided
    if config.get('x_validation_file') and config.get('y_validation_file'):
        logger.info("Validating model with validation data.")
        try:
            logger.info(f"x_validation shape: {x_validation.shape}")
            logger.info(f"y_validation shape: {y_validation.shape}")
            
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
            optimizer_plugin.set_environment(environment_plugin.env, config.get('num_hidden', 0))
            optimizer_plugin.set_agent(agent_plugin)

            # Calculate fitness for the best genome using the same method as in training
            logger.info("Evaluating genome fitness on validation data.")
            validation_fitness, info = optimizer_plugin.evaluate_genome(
                optimizer_plugin.best_genome,
                0,
                agent_plugin.config,
                verbose=True
            )
            validation_outputs = optimizer_plugin.outputs
            validation_node_values = optimizer_plugin.node_values
            logger.info(f"Validation outputs (first 5): {validation_outputs[:5]}")

            # Log the final balance and fitness
            logger.info("*****************************************************************")
            logger.info(f"TRAINING FITNESS: {training_fitness}")
            logger.info(f"VALIDATION FITNESS: {validation_fitness}")
            logger.info("*****************************************************************")

            # Log complexity
            kolmogorov_c = kolmogorov_complexity(optimizer_plugin.best_genome)
            logger.info(f"Kolmogorov Complexity (bytes): {kolmogorov_c}")
            
            # Log number of connections of the champion genome
            num_connections = len(optimizer_plugin.best_genome.connections)
            logger.info(f"Number of connections: {num_connections}")
            
            # Log number of nodes of the champion genome
            num_nodes = len(optimizer_plugin.best_genome.nodes)
            logger.info(f"Number of nodes: {num_nodes}")
            
            # Convert the genome to a string representation
            genome_bytes = pickle.dumps(optimizer_plugin.best_genome)
            # Log the length of the genome
            logger.info(f"Genome length (bits): {len(genome_bytes) * 8}")
            
            # Log the Shannon entropy of the weights
            weights_entropy = calculate_weights_entropy(optimizer_plugin.best_genome)
            logger.info(f"Weights entropy (bits): {weights_entropy}")

            logger.info("*****************************************************************")

            # Calculate training information
            logger.info("Calculating training information.")
            training_input_information = shannon_hartley_information(y_train, config.get('periodicity_minutes', 1))
            logger.info(f"Training Input Information (bits): {training_input_information}")
            
            training_output_information = shannon_hartley_information(training_outputs, config.get('periodicity_minutes', 1))
            logger.info(f"Training Output Information (bits): {training_output_information}")
            
            training_node_values_information = shannon_hartley_information(training_node_values, config.get('periodicity_minutes', 1))
            logger.info(f"Total Training Node Values Information (bits): {training_node_values_information}")
            
            # Calculate total training information
            if training_node_values_information is None:
                training_total_information = num_connections * weights_entropy
            else:
                training_total_information = num_connections * weights_entropy + training_node_values_information
            logger.info(f"Total Training Information (bits): {training_total_information}")

            logger.info("*****************************************************************")

            # Calculate validation information
            logger.info("Calculating validation information.")
            input_information_validation = shannon_hartley_information(y_validation, config.get('periodicity_minutes', 1))
            logger.info(f"Validation Input Information (bits): {input_information_validation}")
            
            output_information_validation = shannon_hartley_information(validation_outputs, config.get('periodicity_minutes', 1))
            logger.info(f"Validation Output Information (bits): {output_information_validation}")
            
            node_values_information_validation = shannon_hartley_information(validation_node_values, config.get('periodicity_minutes', 1))
            logger.info(f"Total Validation Node Values Information (bits): {node_values_information_validation}")
            
            # Calculate total validation information
            if node_values_information_validation is None:
                validation_total_information = num_connections * weights_entropy
            else:
                validation_total_information = num_connections * weights_entropy + node_values_information_validation
            logger.info(f"Total Validation Information (bits): {validation_total_information}")

            logger.info("*****************************************************************")

            # Save debug info
            end_time = time.time()
            execution_time = end_time - start_time
            debug_info = {
                'execution_time': float(execution_time),
                'training_fitness': float(training_fitness),
                'validation_fitness': float(validation_fitness)
            }

            # Save debug info
            if config.get('save_log'):
                try:
                    logger.info(f"Saving debug info to: {config['save_log']}")
                    save_debug_info(debug_info, config['save_log'])
                    logger.debug("Debug info saved locally.")
                except Exception as e:
                    logger.error(f"Failed to save debug info to {config['save_log']}: {e}")
                    raise

            # Remote log debug info and config
            if config.get('remote_log'):
                try:
                    logger.info(f"Saving debug info remotely to: {config['remote_log']}")
                    remote_log(config, debug_info, config['remote_log'], config.get('username'), config.get('password'))
                    logger.debug("Debug info saved remotely.")
                except Exception as e:
                    logger.error(f"Failed to remote save debug info to {config['remote_log']}: {e}")
                    raise

            logger.info(f"Execution time: {execution_time} seconds")

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise

def load_and_evaluate_model(config: dict, agent_plugin) -> None:
    """
    Loads a trained model and evaluates it on the provided data.

    This function loads the agent model, processes the training data,
    generates predictions, and saves the predictions to a CSV file.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    agent_plugin : object
        The loaded agent plugin.

    Raises
    ------
    Exception
        If any step in the evaluation fails.
    """
    try:
        logger.info(f"Loading model from: {config['load_model']}")
        agent_plugin.load(config['load_model'])
    except Exception as e:
        logger.error(f"Failed to load model from {config['load_model']}: {e}")
        raise

    try:
        logger.info("Loading and processing input data for evaluation.")
        x_train, _, _, _, _, _, _, _ = process_data(config)
    except Exception as e:
        logger.error(f"Failed to process data for evaluation: {e}")
        raise

    try:
        logger.info("Generating predictions using the loaded model.")
        predictions = agent_plugin.decide_action(pd.DataFrame(x_train.to_numpy()))
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise

    try:
        evaluate_filename = config['evaluate_file']
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        write_csv(
            evaluate_filename,
            predictions_df,
            include_date=config.get('force_date', False),
            headers=config.get('headers', False)
        )
        logger.info(f"Predicted data saved to {evaluate_filename}")
    except Exception as e:
        logger.error(f"Failed to save predicted data to {evaluate_filename}: {e}")
        raise
