# config.py

DEFAULT_VALUES = {
    'x_train_file': 'tests\\data\\x_d2_original.csv',
    'y_train_file': '..\\Documents\\encoder_eval_d2_2.csv',
    'x_validation_file': 'tests\\data\\x_d3_original.csv',
    'y_validation_file': '..\\Documents\\encoder_eval_d3_2.csv',
    'target_column': None,
    'output_file': 'csv_output.csv',
    'save_model': 'model.keras',
    'load_model': None,
    'evaluate_file': 'model_eval.csv',
    'optimizer_plugin': 'neat_a',
    'environment_plugin': 'gym-fx-env',
    'agent_plugin': 'neat_a',
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None,
    'load_config': None,
    'save_config': 'config_out.json',
    'save_log': 'debug_out.json',
    'quiet_mode': False,
    'force_date': False,
    'headers': True,
    'max_steps': 10000,
    'batch_size': 32,
    'epochs': 1000,
    'input_offset': 128,
    'mse_threshold': 0.001,
    'time_horizon': 0
}

# mapping of short-form to long-form arguments
ARGUMENT_MAPPING = {
    'ytf': 'y_train_file',
    'xvf': 'x_validation_file',
    'yvf': 'y_validation_file',
    'tc': 'target_column',
    'of': 'output_file',
    'sm': 'save_model',
    'lm': 'load_model',
    'ef': 'evaluate_file',
    'op': 'optimizer_plugin',
    'ep': 'environment_plugin',
    'ap': 'agent_plugin',
    'rl': 'remote_log',
    'rlc': 'remote_load_config',
    'rsc': 'remote_save_config',
    'u': 'username',
    'p': 'password',
    'lc': 'load_config',
    'sc': 'save_config',
    'sl': 'save_log',
    'qm': 'quiet_mode',
    'fd': 'force_date',
    'hdr': 'headers',
    'max_steps': 'max_steps',
    'bs': 'batch_size',
    'e': 'epochs'
}
