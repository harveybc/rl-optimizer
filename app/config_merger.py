# app/config_merger.py

import sys
from typing import List, Dict, Any
from app.config import DEFAULT_VALUES, ARGUMENT_MAPPING
from app.logger import get_logger

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
    logger.debug("Starting to process unknown command-line arguments.")
    processed_args = {}
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i].lstrip('-')
        value = unknown_args[i + 1] if i + 1 < len(unknown_args) else None

        # Convert short-form to long-form using the mapping
        if key in ARGUMENT_MAPPING:
            original_key = key
            key = ARGUMENT_MAPPING[key]
            logger.debug(f"Converted short-form argument '{original_key}' to long-form '{key}'.")

        processed_args[key] = value
        logger.debug(f"Processed argument: {key} = {value}")
        i += 2

    logger.debug(f"Completed processing unknown arguments: {processed_args}")
    return processed_args

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
    logger.debug(f"Attempting to convert value: {value}")
    if value is None:
        logger.debug("Value is None; returning as is.")
        return value

    try:
        converted = int(value)
        logger.debug(f"Converted value to int: {converted}")
        return converted
    except (ValueError, TypeError):
        try:
            converted = float(value)
            logger.debug(f"Converted value to float: {converted}")
            return converted
        except (ValueError, TypeError):
            logger.debug(f"Value remains as string: {value}")
            return value

def merge_config(defaults: Dict[str, Any],
                plugin_params: Dict[str, Any],
                config: Dict[str, Any],
                cli_args: Dict[str, Any],
                unknown_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges configuration dictionaries with the following precedence:
    CLI arguments > Unknown arguments > File configuration > Plugin parameters > Default configuration.

    Parameters
    ----------
    defaults : Dict[str, Any]
        The default configuration parameters.
    plugin_params : Dict[str, Any]
        Plugin-specific configuration parameters.
    config : Dict[str, Any]
        Configuration loaded from files (local or remote).
    cli_args : Dict[str, Any]
        Configuration parameters passed via command-line arguments.
    unknown_args : Dict[str, Any]
        Additional configuration parameters not recognized by the CLI.

    Returns
    -------
    Dict[str, Any]
        The merged configuration dictionary.
    """
    logger.debug("Starting configuration merge process.")

    # Step 1: Start with default values from config.py
    merged_config = defaults.copy()
    logger.debug(f"Step 1 - Defaults: {merged_config}")

    # Step 2: Merge with plugin default parameters
    for key, value in plugin_params.items():
        logger.debug(f"Step 2 - Merging plugin_param '{key}' = {value}")
        merged_config[key] = value
    logger.debug(f"Step 2 Output: {merged_config}")

    # Step 3: Merge with file configuration
    for key, value in config.items():
        logger.debug(f"Step 3 - Merging from file config: '{key}' = {value}")
        merged_config[key] = value
    logger.debug(f"Step 3 Output: {merged_config}")

    # Step 4: Merge with CLI arguments (CLI args always override)
    # Exclude sys.argv[0] to prevent script name from being treated as positional argument
    cli_keys_single = [arg.lstrip('-') for arg in sys.argv[1:] if arg.startswith('-') and not arg.startswith('--')]
    cli_expanded = []
    for key in cli_keys_single:
        if key in ARGUMENT_MAPPING:
            original_key = key
            key = ARGUMENT_MAPPING[key]
            cli_expanded.append(key)
            logger.debug(f"Expanded CLI short-form argument '{original_key}' to '{key}'.")

    cli_keys_double = [arg.lstrip('--') for arg in sys.argv[1:] if arg.startswith('--')]
    cli_keys = cli_keys_double + cli_expanded
    logger.debug(f"CLI keys to merge: {cli_keys}")

    for key in cli_keys:
        if key in cli_args and cli_args[key] is not None:
            logger.debug(f"Step 4 - Merging from CLI args: '{key}' = {cli_args[key]}")
            merged_config[key] = cli_args[key]
        elif key in unknown_args and unknown_args[key] is not None:
            converted_value = convert_type(unknown_args[key])
            logger.debug(f"Step 4 - Merging from unknown args: '{key}' = {converted_value}")
            merged_config[key] = converted_value

    # Step 5: Handle additional positional arguments
    # We only consider them if they are non-numeric, and x_train_file wasn't explicitly set.
    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    leftover_non_numeric = []
    for arg in positional_args:
        # Attempt to see if it is purely numeric (could be "20" for --epochs)
        try:
            float(arg)
            # It's numeric => skip
            continue
        except ValueError:
            # It's a non-numeric leftover => candidate
            leftover_non_numeric.append(arg)

    # If x_train_file was never set or remains the default,
    # we take the first leftover non-numeric argument (if any).
    if 'x_train_file' not in merged_config or merged_config['x_train_file'] == defaults['x_train_file']:
        if len(leftover_non_numeric) > 0:
            merged_config['x_train_file'] = leftover_non_numeric[0]
            logger.debug(f"Special handling - Set 'x_train_file' to positional argument: {leftover_non_numeric[0]}")

    logger.debug(f"Final merged configuration: {merged_config}")
    return merged_config

