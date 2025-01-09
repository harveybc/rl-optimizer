# app/utils.py

import math
import pickle
import zlib
import logging
import numpy as np
import pandas as pd
from app.logger import get_logger
from typing import List, Dict, Any

logger = get_logger(__name__)


def process_unknown_args(unknown_args: List[str]) -> Dict[str, Any]:
    """
    Processes unknown command-line arguments into a configuration dictionary.

    Parameters
    ----------
    unknown_args : List[str]
        List of unknown command-line arguments.

    Returns
    -------
    Dict[str, Any]
        Dictionary of processed unknown arguments.
    """
    logger.debug("Processing unknown command-line arguments.")
    config = {}
    it = iter(unknown_args)
    for arg in it:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            try:
                value = next(it)
                config[key] = value
                logger.debug(f"Processed unknown arg: {key} = {value}")
            except StopIteration:
                logger.warning(f"No value provided for argument: {arg}")
                config[key] = True  # Flag argument
    logger.debug(f"Processed unknown arguments: {config}")
    return config

def convert_type(value: Any) -> Any:
    """
    Attempts to convert a value to int or float. Returns the original value if conversion fails.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    Any
        The converted value or the original value if conversion fails.
    """
    logger.debug(f"Converting type for value: {value}")
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.debug(f"Value remains as string: {value}")
            return value

def kolmogorov_complexity(genome, compression_level: int = 9) -> int:
    """
    Estimate the Kolmogorov complexity of a NEAT genome by compressing its serialized form.

    Parameters
    ----------
    genome : object
        The NEAT genome to calculate complexity for.
    compression_level : int, optional
        The compression level for zlib (1-9), where 9 is the highest level. Default is 9.

    Returns
    -------
    int
        The length of the compressed genome in bytes as an estimate of its Kolmogorov complexity.
    """
    logger.debug("Calculating Kolmogorov complexity for the genome.")
    try:
        genome_bytes = pickle.dumps(genome)
        compressed_data = zlib.compress(genome_bytes, level=compression_level, wbits=-15)
        complexity = len(compressed_data)
        logger.debug(f"Kolmogorov complexity calculated: {complexity} bytes.")
        return complexity
    except Exception as e:
        logger.error(f"Error calculating Kolmogorov complexity: {e}")
        raise

def shannon_hartley_information(input_data, period_minutes: int) -> float:
    """
    Calculate the Shannon-Hartley information of the input data.

    Parameters
    ----------
    input_data : pd.DataFrame, list, or np.ndarray
        The input data to calculate information for.
    period_minutes : int
        The periodicity of the data in minutes.

    Returns
    -------
    float or None
        The total information in bits, or None if calculation fails.
    """
    logger.debug("Calculating Shannon-Hartley information.")
    try:
        # Convert input to NumPy array if necessary
        if isinstance(input_data, pd.DataFrame):
            np_input = input_data.to_numpy()
        elif isinstance(input_data, list):
            # Verify the size of each element in the list
            input_lengths = [len(i) if hasattr(i, '__len__') else 1 for i in input_data]
            if len(set(input_lengths)) != 1:
                logger.warning(f"Inhomogeneous lengths in input: {input_lengths}")
                return None  # Early exit
            # Convert list to NumPy array
            np_input = np.array(input_data)
        elif isinstance(input_data, np.ndarray):
            np_input = input_data
        else:
            logger.warning("Input must be a pandas DataFrame, a list of lists, or a NumPy array.")
            return None

        # Verify that np_input is a 2D array
        if np_input.ndim != 2:
            logger.warning(f"Input array must be 2D (rows, columns). Got {np_input.ndim}D.")
            return None

        # Ensure array is not empty and all columns have data
        if np_input.shape[1] == 0:
            logger.warning("Input array must have at least one column.")
            return None

        # Normalize each column between 0 and 1
        min_vals = np.min(np_input, axis=0)
        max_vals = np.max(np_input, axis=0)

        # Check for division by zero
        if np.any(max_vals - min_vals == 0):
            logger.warning("One or more columns have constant values, which causes division by zero in normalization.")
            return None
        else:
            np_input = (np_input - min_vals) / (max_vals - min_vals)

        # Concatenate columns vertically
        input_concat = np.concatenate(np_input, axis=0)

        # Calculate mean and standard deviation of concatenated input
        input_mean = np.mean(input_concat)
        input_std = np.std(input_concat)

        # Check that standard deviation is not zero (avoid division by zero)
        if input_std == 0:
            logger.warning("Standard deviation of the input is zero, cannot calculate SNR.")
            return None

        # Calculate SNR as (mean/std)^2
        input_SNR = (input_mean / input_std) ** 2

        # Calculate the sampling frequency in Hz
        sampling_frequency = 1 / (period_minutes * 60)

        # Calculate total capacity in bits per second using Shannon-Hartley formula
        input_capacity = sampling_frequency * np.log2(1 + input_SNR)

        # Calculate total input information in bits by multiplying capacity by total time in seconds
        total_time_seconds = len(input_concat) * period_minutes * 60
        input_information = input_capacity * total_time_seconds

        logger.debug(f"Shannon-Hartley information calculated: {input_information} bits.")
        return input_information

    except Exception as e:
        logger.warning(f"An error occurred during Shannon-Hartley information calculation: {e}")
        return None

def calculate_weights_entropy(genome, num_bins: int = 50) -> float:
    """
    Calculate the Shannon entropy of the weights of a NEAT genome.

    Parameters
    ----------
    genome : object
        The NEAT genome containing connection weights.
    num_bins : int, optional
        The number of bins to use for discretizing the weight values (default is 50).

    Returns
    -------
    float
        The Shannon entropy of the weight distribution in bits.
    """
    logger.debug("Calculating weights entropy.")
    try:
        # Extract the weights from the genome's connections
        weights = [conn.weight for conn in genome.connections.values() if conn.enabled]

        if not weights:
            logger.warning("No weights found in the genome.")
            return 0.0

        # Normalize the weights to be between 0 and 1
        min_weight = min(weights)
        max_weight = max(weights)
        if max_weight == min_weight:
            logger.warning("All weights are equal; entropy is zero.")
            return 0.0
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]

        # Create a histogram to get the probability distribution
        hist, _ = np.histogram(normalized_weights, bins=num_bins, range=(0, 1), density=True)

        # Calculate the probabilities for each bin
        probabilities = hist / np.sum(hist)

        # Calculate the Shannon entropy
        entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])

        logger.debug(f"Weights entropy calculated: {entropy} bits.")
        return entropy

    except Exception as e:
        logger.error(f"Error calculating weights entropy: {e}")
        raise
