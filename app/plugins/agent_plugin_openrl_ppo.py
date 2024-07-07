import torch
import torch.nn as nn

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        policy_dist = self.actor(state)
        return policy_dist, value

class Plugin:
    """
    An agent plugin for making predictions using a PPO model.
    """

    plugin_params = {
        'state_dim': 64,
        'action_dim': 1,
        'hidden_dim': 64,
        'genome_file': 'ppo_model.pkl'
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.agent = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        self.agent = PPOAgent(
            state_dim=self.params['state_dim'],
            action_dim=self.params['action_dim'],
            hidden_dim=self.params['hidden_dim']
        )

    def load(self, model_path):
        with open(model_path, 'rb') as f:
            self.agent.load_state_dict(torch.load(f))
        print(f"Agent model loaded from {model_path}")

    def predict(self, data):
        self.agent.eval()
        with torch.no_grad():
            data = torch.FloatTensor(data.to_numpy())
            policy_dist, _ = self.agent(data)
            predictions = policy_dist.numpy()
        return predictions

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            torch.save(self.agent.state_dict(), f)
        print(f"Agent model saved to {model_path}")

    def get_agent(self):
        return self.agent
