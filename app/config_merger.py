# app/config_merger.py

from typing import Dict, Any, List
from app.logger import get_logger
from app.utils import process_unknown_args, convert_type

logger = get_logger(__name__)

def merge_config(defaults: Dict[str, Any],
                plugin_params: Dict[str, Any],
                file_config: Dict[str, Any],
                cli_args: Dict[str, Any],
                unknown_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges configuration dictionaries with a specific precedence:
    CLI arguments > Unknown arguments > File configuration > Plugin parameters > Default configuration.

    Parameters
    ----------
    defaults : Dict[str, Any]
        The default configuration parameters.
    plugin_params : Dict[str, Any]
        Plugin-specific configuration parameters.
    file_config : Dict[str, Any]
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
    logger.debug("Starting configuration merge.")
    
    # Step 1: Start with default values
    merged_config = defaults.copy()
    logger.debug(f"Step 1 - Defaults: {merged_config}")
    
    # Step 2: Merge with plugin default parameters
    logger.debug(f"Step 2 - Merging with plugin parameters: {plugin_params}")
    merged_config.update(plugin_params)
    logger.debug(f"Step 2 Output: {merged_config}")
    
    # Step 3: Merge with file configuration
    logger.debug(f"Step 3 - Merging with file configuration: {file_config}")
    merged_config.update(file_config)
    logger.debug(f"Step 3 Output: {merged_config}")
    
    # Step 4: Merge with CLI arguments (CLI args override)
    logger.debug(f"Step 4 - Merging with CLI arguments: {cli_args}")
    merged_config.update(cli_args)
    logger.debug(f"Step 4 Output (after CLI args): {merged_config}")
    
    # Step 5: Merge with unknown arguments (Unknown args override)
    logger.debug(f"Step 5 - Merging with unknown arguments: {unknown_args}")
    merged_config.update(unknown_args)
    logger.debug(f"Step 5 Output (after unknown args): {merged_config}")
    
    return merged_config
