import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config, remote_load_config, remote_save_config, remote_log
from app.cli import parse_args
from app.data_processor import process_data, load_and_evaluate_model, run_prediction_pipeline
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    print("Initial sys.path:", sys.path)
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()

    cli_args = vars(args)

    print("Loading default configuration...")
    config = DEFAULT_VALUES.copy()

    file_config = {}
    # remote config file load
    if args.remote_load_config:
        file_config = remote_load_config(args.remote_load_config, args.username, args.password)
        print(f"Loaded remote config: {file_config}")

    # local config file load
    if args.load_config:
        file_config = load_config(args.load_config)
        print(f"Loaded local config: {file_config}")

    print("Merging configuration with CLI arguments and unknown args...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)

    # Load and set optimizer plugin
    optimizer_plugin_name = config['optimizer_plugin']
    print(f"Loading optimizer plugin: {optimizer_plugin_name}")
    optimizer_class, _ = load_plugin('rl_optimizer.optimizers', optimizer_plugin_name)
    optimizer_plugin = optimizer_class()

    # Load and set environment plugin
    environment_plugin_name = config['environment_plugin']
    print(f"Loading environment plugin: {environment_plugin_name}")
    environment_class, _ = load_plugin('rl_optimizer.environments', environment_plugin_name)
    environment_plugin = environment_class()

    # Load and set agent plugin
    agent_plugin_name = config['agent_plugin']
    print(f"Loading agent plugin: {agent_plugin_name}")
    agent_class, _ = load_plugin('rl_optimizer.agents', agent_plugin_name)
    agent_plugin = agent_class()

    # Merging environment-specific parameters
    config = merge_config(config, environment_plugin.plugin_params, file_config, cli_args, unknown_args_dict)
    environment_plugin.set_params(**config)

    if config['load_model']:
        print("Loading and evaluating model...")
        load_and_evaluate_model(config, agent_plugin)
    else:
        print("Processing and running prediction pipeline...")
        run_prediction_pipeline(config, environment_plugin, agent_plugin, optimizer_plugin)  # Pass all required plugins

    if 'save_config' in config and config['save_config']:
        save_config(config, config['save_config'])
        print(f"Configuration saved to {config['save_config']}.")

    if 'remote_save_config' in config and config['remote_save_config']:
        print(f"Remote saving configuration to {config['remote_save_config']}")
        remote_save_config(config, config['remote_save_config'], config['username'], config['password'])
        print(f"Remote configuration saved.")

if __name__ == "__main__":
    main()
