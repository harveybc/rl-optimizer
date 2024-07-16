import neat
import os
import pickle
import numpy as np

class Plugin:
    """
    An optimizer plugin using NEAT for evolutionary neural networks.
    """

    plugin_params = {
        'config_file': 'tests/data/neat_50.ini',
        'genome_file': 'winner.pkl',
        'epochs': 1,
        'batch_size': 256,
    }

    plugin_debug_vars = ['config_file', 'epochs', 'batch_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.environment = None
        self.agent = None
        self.best_genome = None
        self.num_inputs = 0

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def set_environment(self, environment):
        self.environment = environment
        if self.environment.y_train is not None:
            self.num_inputs = self.environment.y_train.shape[1]
        else:
            self.num_inputs = self.environment.x_train.shape[1]

    def set_agent(self, agent):
        self.agent = agent

    def train(self, epochs):
        config_path = self.params['config_file']
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        
        # Overwrite the num_inputs and input_nodes as the number of columns of self.environment.x_train or y_train
        config.genome_config.num_inputs = self.num_inputs
        config.genome_config.input_keys = [-i - 1 for i in range(self.num_inputs)]

        # Overwrite the num_outputs for discrete actions
        config.genome_config.num_outputs = 3  # For buy, sell, and hold actions
        config.genome_config.output_keys = [i for i in range(3)]

        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                self.agent.set_model(genome, config)  # Set the genome in the agent
                genome.fitness = self.evaluate_genome(genome)

        self.best_genome = population.run(eval_genomes, epochs)

        # Save the best genome
        with open(self.params['genome_file'], 'wb') as f:
            pickle.dump(self.best_genome, f)
        
        # Print the champion genome
        print(f"Champion Genome:\n{self.best_genome}")

        # Print the nodes and connections of the best genome
        nodes = self.best_genome.nodes
        connections = self.best_genome.connections
        print("Nodes:")
        for node_key, node in nodes.items():
            print(f"Node {node_key}: {node}")
        print("Connections:")
        for conn_key, conn in connections.items():
            print(f"Connection {conn_key}: {conn}")

    def evaluate_genome(self, genome):
        fitness = 0.0
        observation, info = self.environment.reset()
        done = False
        action_counts = {'buy': 0, 'sell': 0, 'hold': 0}

        # Print observation statistics
        #print("Observation statistics:")
        #print(f"Mean: {np.mean(observation)}, Std: {np.std(observation)}, Min: {np.min(observation)}, Max: {np.max(observation)}")

        while not done:
            action = self.agent.predict(observation, info)  # Get action values from the agent
            # Increment action counts and print only if action is different from 'hold'
            if action == 1:
                action_counts['buy'] += 1
                #print(f"Action taken: Buy (1)")
            elif action == 2:
                action_counts['sell'] += 1
                #print(f"Action taken: Sell (2)")
            else:
                action_counts['hold'] += 1

            observation, reward, done, info = self.environment.step(action)

            fitness += reward

        #print(f"Action counts - Buy: {action_counts['buy']}, Sell: {action_counts['sell']}, Hold: {action_counts['hold']}")

        return float(fitness)  # Explicitly return float

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.best_genome, f)
        print(f"Optimizer model saved to {file_path}")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.best_genome = pickle.load(f)
        print(f"Optimizer model loaded from {file_path}")
