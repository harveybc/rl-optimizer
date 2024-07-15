import neat
import os
import pickle

class Plugin:
    """
    An optimizer plugin using NEAT for evolutionary neural networks.
    """

    plugin_params = {
        'config_file': 'tests/data/neat_50.ini',
        'genome_file': 'winner.pkl',
        'epochs': 10,
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
        self.num_inputs = self.environment.x_train.shape[1]

    def set_agent(self, agent):
        self.agent = agent

    def train(self, epochs):
        config_path = self.params['config_file']
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        
        # Overwrite the num_inputs and input_nodes as the number of columns of self.environment.x_train
        config.genome_config.num_inputs = self.num_inputs
        config.genome_config.input_keys = [-i - 1 for i in range(self.num_inputs)]

        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = self.evaluate_genome(genome, config)

        self.best_genome = population.run(eval_genomes, epochs)

        # Save the best genome
        with open(self.params['genome_file'], 'wb') as f:
            pickle.dump(self.best_genome, f)

    def evaluate_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        observation = self.environment.reset()
        done = False
        while not done:
            action = net.activate(observation)
            observation, reward, done, info = self.environment.step(action)

            fitness += reward

        return float(fitness)  # Explicitly return float

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.best_genome, f)
        print(f"Optimizer model saved to {file_path}")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.best_genome = pickle.load(f)
        print(f"Optimizer model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.set_params(config_file='neat_config.ini', epochs=10, batch_size=256)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")