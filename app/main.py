# app/main.py

import sys
import json
import logging
from app.logger import setup_logging, get_logger  # Import the centralized logger
from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log,
)
from app.cli import parse_args
from app.data_processor import (
    process_data,
    load_and_evaluate_model,
    run_prediction_pipeline,
)
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    """
    Main entry point for the RL optimizer application.

    This function orchestrates the overall workflow:
    - Sets up centralized logging.
    - Parses command-line arguments.
    - Loads and merges configurations from default, local, and remote sources.
    - Dynamically loads optimizer, environment, and agent plugins.
    - Executes the training or prediction pipeline based on the configuration.
    - Saves configurations locally or remotely if specified.
    """
    # Initialize centralized logging
    setup_logging(
        log_level=logging.DEBUG,
        log_file="rl_optimizer.log",
        max_bytes=10*1024*1024,  # 10 MB
        backup_count=5
    )
    logger = get_logger(__name__)
    logger.debug("Starting main application.")

    try:
        logger.info("Parsing command-line arguments...")
        args, unknown_args = parse_args()
        cli_args = vars(args)
        logger.debug(f"Parsed CLI arguments: {cli_args}")

        logger.info("Loading default configuration...")
        config = DEFAULT_VALUES.copy()
        logger.debug(f"Default configuration: {config}")

        file_config = {}
        
        # Remote config file load
        if args.remote_load_config:
            logger.info(f"Loading remote configuration from: {args.remote_load_config}")
            file_config = remote_load_config(
                args.remote_load_config, args.username, args.password
            )
            logger.debug(f"Loaded remote config: {file_config}")

        # Local config file load
        if args.load_config:
            logger.info(f"Loading local configuration from: {args.load_config}")
            local_config = load_config(args.load_config)
            file_config.update(local_config)
            logger.debug(f"Loaded local config: {local_config}")

        logger.info("Merging configurations...")
        unknown_args_dict = process_unknown_args(unknown_args)
        logger.debug(f"Unknown arguments: {unknown_args_dict}")
        config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)
        logger.debug(f"Merged configuration: {config}")

        # Load and initialize optimizer plugin
        optimizer_plugin_name = config.get("optimizer_plugin")
        logger.info(f"Loading optimizer plugin: {optimizer_plugin_name}")
        optimizer_class, optimizer_module = load_plugin(
            "rl_optimizer.optimizers", optimizer_plugin_name
        )
        optimizer_plugin = optimizer_class()
        logger.debug(f"Loaded optimizer plugin '{optimizer_plugin_name}' from '{optimizer_module}'.")

        # Load and initialize environment plugin
        environment_plugin_name = config.get("environment_plugin")
        logger.info(f"Loading environment plugin: {environment_plugin_name}")
        environment_class, environment_module = load_plugin(
            "rl_optimizer.environments", environment_plugin_name
        )
        environment_plugin = environment_class()
        logger.debug(f"Loaded environment plugin '{environment_plugin_name}' from '{environment_module}'.")

        # Load and initialize agent plugin
        agent_plugin_name = config.get("agent_plugin")
        logger.info(f"Loading agent plugin: {agent_plugin_name}")
        agent_class, agent_module = load_plugin(
            "rl_optimizer.agents", agent_plugin_name
        )
        agent_plugin = agent_class()
        logger.debug(f"Loaded agent plugin '{agent_plugin_name}' from '{agent_module}'.")

        # Merge environment-specific parameters
        logger.info("Merging environment-specific parameters...")
        environment_params = getattr(environment_plugin, 'plugin_params', {})
        config = merge_config(config, environment_params, file_config, cli_args, unknown_args_dict)
        logger.debug(f"Configuration after merging environment parameters: {config}")

        # Set parameters for the environment plugin
        environment_plugin.set_params(**config)
        logger.debug("Environment plugin parameters set.")

        # Determine whether to load an existing model or run the prediction pipeline
        if config.get("load_model"):
            logger.info("Loading and evaluating the existing model...")
            load_and_evaluate_model(config, agent_plugin)
        else:
            logger.info("Processing data and running the prediction pipeline...")
            run_prediction_pipeline(config, environment_plugin, agent_plugin, optimizer_plugin)

        # Save configuration locally if specified
        if config.get("save_config"):
            logger.info(f"Saving configuration to: {config['save_config']}")
            save_config(config, config["save_config"])
            logger.debug("Configuration saved locally.")

        # Save configuration remotely if specified
        if config.get("remote_save_config"):
            logger.info(f"Saving configuration remotely to: {config['remote_save_config']}")
            remote_save_config(
                config,
                config["remote_save_config"],
                config.get("username"),
                config.get("password"),
            )
            logger.debug("Configuration saved remotely.")

        logger.info("Main application completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
