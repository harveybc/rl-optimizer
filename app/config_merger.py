import sys
from app.config import DEFAULT_VALUES, ARGUMENT_MAPPING

def process_unknown_args(unknown_args):
    processed_args = {}
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i].lstrip('-')
        value = unknown_args[i + 1] if i + 1 < len(unknown_args) else None
        # Convert short-form to long-form using the mapping
        if key in ARGUMENT_MAPPING:
            key = ARGUMENT_MAPPING[key]
        processed_args[key] = value
        i += 2
    return processed_args

def convert_type(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def merge_config(defaults, plugin_params, config, cli_args, unknown_args):
    # Step 1: Start with default values from config.py
    merged_config = defaults.copy()
    print(f"Actual Step 1 Output: {merged_config}")

    # Step 2: Merge with plugin default parameters
    for k, v in plugin_params.items():
        print(f"Step 2 merging: plugin_param {k} = {v}")
        merged_config[k] = v
    print(f"Actual Step 2 Output: {merged_config}")

    # Step 3: Merge with file configuration
    for k, v in config.items():
        print(f"Step 3 merging from file config: {k} = {v}")
        merged_config[k] = v
    print(f"Actual Step 3 Output: {merged_config}")

    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    cli_keys_single = [arg.lstrip('-') for arg in sys.argv if arg.startswith('-') and not arg.startswith('--')]
    cli_expanded = []
    for key in cli_keys_single:
        if key in ARGUMENT_MAPPING:
            key = ARGUMENT_MAPPING[key]
            cli_expanded.append(key)

    cli_keys_double = [arg.lstrip('--') for arg in sys.argv if arg.startswith('--')]
    cli_keys = cli_keys_double + cli_expanded
    print(f"CLI keys: {cli_keys}")

    for key in cli_keys:
        original_key = key
        if original_key in cli_args:
            print(f"Step 4 merging from CLI args: {key} = {cli_args[original_key]}")
            merged_config[key] = cli_args[original_key]
        elif original_key in unknown_args:
            value = convert_type(unknown_args[original_key])
            print(f"Step 4 merging from unknown args: {key} = {value}")
            merged_config[key] = value

    # Special handling for input_file
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        merged_config['x_train_file'] = sys.argv[1]

    print(f"Actual Step 4 Output: {merged_config}")

    return merged_config
