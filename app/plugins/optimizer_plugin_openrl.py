import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

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
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy_dist, value = self.agent(state_tensor)
                policy_dist = torch.distributions.Normal(policy_dist, torch.ones_like(policy_dist))
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action).sum(dim=-1)
                state, reward, done, _ = self.environment.step(action.detach().numpy())

                states.append(state_tensor)
                actions.append(action)
                rewards.append(torch.tensor([reward], dtype=torch.float32))
                old_log_probs.append(log_prob)
                values.append(value)

                if done:
                    break

            # Convert to tensors
            print(f"Shapes before concatenation - rewards: {len(rewards)}, states: {len(states)}, actions: {len(actions)}, old_log_probs: {len(old_log_probs)}, values: {len(values)}")
            rewards = torch.stack(rewards)
            states = torch.cat(states)
            actions = torch.cat(actions)
            old_log_probs = torch.cat(old_log_probs)
            values = torch.cat(values)

            print(f"Shapes after concatenation - rewards: {rewards.shape}, states: {states.shape}, actions: {actions.shape}, old_log_probs: {old_log_probs.shape}, values: {values.shape}")

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
                policy_dist = torch.distributions.Normal(policy_dist, torch.ones_like(policy_dist))
                log_probs = policy_dist.log_prob(actions).sum(dim=-1)
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
