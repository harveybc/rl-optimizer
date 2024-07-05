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

    def set_agent(self, agent):
        self.agent = agent

    def train(self, x_train, y_train, epochs, batch_size):
        config_path = self.params['config_file']
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

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
        total_error = 0.0
        total_predictions = 0
        
        observation = self.environment.reset()
        done = False
        while not done:
            action = net.activate(observation)
            observation, reward, done, info = self.environment.step(action)

            total_error += 1/reward  
            total_predictions += 1

        mae = total_error / total_predictions if total_predictions > 0 else float('inf')
        fitness = 1/mae
        #print(f"MAE for genome {genome.key}: {mae}")
        return fitness

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
