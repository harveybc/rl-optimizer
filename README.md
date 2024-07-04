# RL-Optimizer

## Description

RL-Optimizer is a powerful and flexible tool designed for optimizing reinforcement learning agents in various environments, with support for dynamic plugins. It utilizes multiple optimization techniques, including NEAT, NEAT_P2P, and those supported by OpenRL, to optimize models controlling trading agents. The tool is capable of handling multi-asset trading environments, making it highly versatile for both training and evaluation tasks in the domain of financial trading and beyond.

### Key Features:
- **Dynamic Plugins:** Easily switch between different optimizer techniques (e.g., NEAT, NEAT_P2P, OpenRL algorithms), environments (e.g., gym-fx for multi-asset management), and agents.
- **Configurable Parameters:** Customize the optimization process with parameters such as trading actions, portfolio balancing strategies, and evaluation metrics.
- **Model Management:** Save and load models for reuse, avoiding the need to retrain models from scratch.
- **Remote Configuration:** Load and save configurations remotely, facilitating seamless integration with other systems and automation pipelines.
- **Incremental Search:** Optimize the model parameters dynamically during training to achieve the best performance based on specified error thresholds.

This tool is designed for data scientists, machine learning engineers, and financial analysts who need to optimize trading strategies and models efficiently, and it can be easily integrated into larger machine learning workflows.

## Installation Instructions

To install and set up the RL-Optimizer application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/rl-optimizer.git
    cd rl-optimizer
    ```

2. **Create and Activate a Virtual Environment (Anaconda is required)**:

    - **Using `conda`**:
        ```bash
        conda create --name rl-optimizer-env python=3.9
        conda activate rl-optimizer-env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **(Optional) Run the RL-Optimizer**:
    - On Windows, run the following command to verify installation:
        ```bash
        rl-optimizer.bat tests\data\config.json
        ```

    - On Linux, run:
        ```bash
        sh rl-optimizer.sh tests\data\config.json
        ```

7. **(Optional) Run Tests**:
    - On Windows, run the following command to run the tests:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **(Optional) Generate Documentation**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```

9. **(Optional) Install Nvidia CUDA GPU support**:
    - Please read: [Readme - CUDA](https://github.com/harveybc/rl-optimizer/blob/master/README_CUDA.md)

## Usage

The application supports several command line arguments to control its behavior:
```bash
usage: rl-optimizer.bat tests\data\config.json
```

### Command Line Arguments

#### Required Arguments

- `config_file` (str): Path to the configuration JSON file.

#### Optional Arguments

- `-of, --output_file` (str): Path to the output CSV file.
- `-sm, --save_model` (str): Filename to save the trained model.
- `-lm, --load_model` (str): Filename to load a trained model from.
- `-ef, --evaluate_file` (str): Filename for outputting evaluation results.
- `-op, --optimizer_plugin` (str, default='neat'): Name of the optimizer plugin to use.
- `-ep, --environment_plugin` (str, default='gym_fx'): Name of the environment plugin to use.
- `-ap, --agent_plugin` (str, default='agent_multi'): Name of the agent plugin to use.
- `-te, --threshold_error` (float): Error threshold to stop the optimization process.
- `-rl, --remote_log` (str): URL of a remote API endpoint for saving debug variables in JSON format.
- `-rlc, --remote_load_config` (str): URL of a remote JSON configuration file to download and execute.
- `-rsc, --remote_save_config` (str): URL of a remote API endpoint for saving configuration in JSON format.
- `-u, --username` (str): Username for the API endpoint.
- `-p, --password` (str): Password for the API endpoint.
- `-lc, --load_config` (str): Path to load a configuration file.
- `-sc, --save_config` (str): Path to save the current configuration.
- `-sl, --save_log` (str): Path to save the current debug info.
- `-qm, --quiet_mode` (flag): Suppress output messages.
- `-fd, --force_date` (flag): Include date in the output CSV files.
- `-inc, --incremental_search` (flag): Enable incremental search for interface size.

### Examples of Use

#### Optimization Example

To optimize an agent using the NEAT optimizer with the gym-fx environment and agent-multi plugin, use the following command:

```bash
rl-optimizer.bat tests\data\config.json --optimizer_plugin neat --environment_plugin gym_fx --agent_plugin agent_multi
```

## Project Directory Structure
```bash
rl-optimizer/
│
├── app/                           # Main application package
│   ├── cli.py                    # Handles command-line argument parsing
│   ├── config.py                 # Stores default configuration values
│   ├── config_handler.py         # Manages configuration loading, saving, and merging
│   ├── config_merger.py          # Merges configuration from various sources
│   ├── data_handler.py           # Handles data loading and saving
│   ├── data_processor.py         # Processes input data and runs the optimization pipeline
│   ├── main.py                   # Main entry point for the application
│   ├── plugin_loader.py          # Dynamically loads optimizer, environment, and agent plugins
│   ├── plugins/                       # Plugin directory
│       ├── optimizer_plugin_neat.py        # Optimizer plugin using NEAT
│       ├── optimizer_plugin_neat_p2p.py    # Optimizer plugin using NEAT_P2P
│       ├── environment_plugin_gym_fx.py    # Environment plugin for gym-fx
│       ├── agent_plugin_multi.py           # Agent plugin using agent-multi
│       ├── agent_plugin_other.py           # Placeholder for additional agent plugin
│
├── tests/              # Test modules for the application
│   ├── acceptance          # User acceptance tests
│   ├── system              # System tests
│   ├── integration         # Integration tests
│   └── unit                # Unit tests
│
├── README.md                     # Overview and documentation for the project
├── requirements.txt              # Lists Python package dependencies
├── setup.py                      # Script for packaging and installing the project
├── set_env.bat                   # Batch script for environment setup
├── set_env.sh                    # Shell script for environment setup
└── .gitignore                         # Specifies intentionally untracked files to ignore

```

Contributing
Contributions to the project are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to make contributions.

License
This project is licensed under the MIT License - see the LICENSE file for details.

