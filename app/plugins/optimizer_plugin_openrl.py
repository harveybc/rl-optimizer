import pandas as pd
import numpy as np
from openrl.algorithms.ppo import PPOAlgorithm
from openrl.algorithms.dqn import DQNAlgorithm
import pickle
import torch
import torch.optim as optim
import torch.nn as nn


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        self.use_share_model = False
        self.use_joint_action_loss = False
        self.use_deepspeed = False
        self.world_size = 1
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.bc_epoch = 2
        self.num_mini_batch = 1
        self.mini_batch_size = 32
        self.data_chunk_length = 10
        self.policy_value_loss_coef = 0.5
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.huber_delta = 10.0
        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = True
        self.use_popart = False
        self.use_valuenorm = True
        self.use_value_active_masks = True
        self.use_policy_active_masks = True
        self.use_policy_vhead = False
        self.use_adv_normalize = False
        self.dec_actor = False
        self.use_amp = False
        self.dual_clip_ppo = False
        self.dual_clip_coeff = 3.0

class Plugin:
    """
    An optimizer plugin using PPO for reinforcement learning.
    """

    plugin_params = {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'hidden_dim': 64,
        'genome_file': 'ppo_model.pkl'
    }

    plugin_debug_vars = ['lr', 'gamma', 'lmbda', 'eps_clip', 'K_epochs', 'hidden_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.environment = None
        self.agent = None
        self.optimizer = None

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
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.params['lr'])

    def train(self, epochs):
        gamma = self.params['gamma']
        lmbda = self.params['lmbda']
        eps_clip = self.params['eps_clip']
        K_epochs = self.params['K_epochs']
class Plugin:
    """
    An optimizer plugin using PPO for reinforcement learning.
    """

    plugin_params = {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'hidden_dim': 64,
        'genome_file': 'ppo_model.pkl'
    }

    plugin_debug_vars = ['lr', 'gamma', 'lmbda', 'eps_clip', 'K_epochs', 'hidden_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.environment = None
        self.agent = None
        self.optimizer = None

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

    def set_agent(self, agent_plugin):
        self.agent = agent_plugin.get_agent()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.params['lr'])

    def train(self, epochs):
        gamma = self.params['gamma']
        lmbda = self.params['lmbda']
        eps_clip = self.params['eps_clip']
        K_epochs = self.params['K_epochs']

        for _ in range(epochs):
            state = self.environment.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            rewards = []
            states = []
            actions = []
            old_log_probs = []
            values = []

            for t in range(self.environment.max_steps):
                policy_dist, value = self.agent(state)
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)
                state, reward, done, _ = self.environment.step(action.numpy())

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_log_probs.append(log_prob)
                values.append(value)

                if done:
                    break

            # Convert to tensors
            rewards = torch.tensor(rewards, dtype=torch.float32)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)

            # Compute advantages
            returns = []
            advantages = []
            G = 0
            for reward in reversed(rewards):
                G = reward + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = returns - values

            # Update policy
            for _ in range(K_epochs):
                policy_dist, value = self.agent(states)
                log_probs = policy_dist.log_prob(actions)
                ratios = torch.exp(log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean() + nn.MSELoss()(value, returns)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.agent.state_dict(), f)
        print(f"Optimizer model saved to {file_path}")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.agent.load_state_dict(pickle.load(f))
        print(f"Optimizer model loaded from {file_path}")