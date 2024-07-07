class Plugin:
    def train(self, epochs):
        gamma = self.params['gamma']
        lmbda = self.params['lmbda']
        eps_clip = self.params['eps_clip']
        K_epochs = self.params['K_epochs']

        for epoch in range(epochs):
            state = self.environment.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            rewards = []
            states = []
            actions = []
            old_log_probs = []
            values = []

            last_valid_state = state.clone()

            for t in range(self.environment.max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy_dist, value = self.agent(state_tensor)
                policy_dist = torch.distributions.Normal(policy_dist, torch.ones_like(policy_dist))
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action).sum(dim=-1)
                state, reward, done, _ = self.environment.step(action.detach().numpy())

                print(f"Step {t}:")
                print(f"  State Tensor: {state_tensor.shape} {state_tensor}")
                print(f"  Policy Dist: {policy_dist}")
                print(f"  Action: {action.shape} {action}")
                print(f"  Log Prob: {log_prob.shape} {log_prob}")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")

                states.append(state_tensor.squeeze(0))  # Ensure state_tensor is 1D
                actions.append(action.squeeze(0))  # Ensure action is 1D
                rewards.append(torch.tensor([reward], dtype=torch.float32).view(1, 1))  # Ensure reward is 2D
                old_log_probs.append(log_prob.view(1))  # Ensure log_prob is 1D
                values.append(value.view(1))  # Ensure value is 1D

                if not done:
                    last_valid_state = state_tensor.squeeze(0).clone()

                if done:
                    if len(states) != self.environment.max_steps:
                        states.append(last_valid_state)  # Append the last valid state to ensure correct dimensions
                    break

            # Convert to tensors and ensure dimensions match
            print(f"Shapes before concatenation - rewards: {rewards[0].shape}, states: {states[0].shape}, actions: {actions[0].shape}, old_log_probs: {old_log_probs[0].shape}, values: {values[0].shape}")
            rewards = torch.cat(rewards, dim=0).view(-1, 1)
            print(f"Rewards shape after concatenation: {rewards.shape}")
            print(f"States shape before concatenation: {states[0].shape}")
            states = torch.cat(states, dim=0).view(-1, self.environment.x_train.shape[1])
            print(f"States shape after concatenation: {states.shape}")
            print(f"Actions shape before concatenation: {actions[0].shape}")
            actions = torch.cat(actions, dim=0).view(-1, 1)
            print(f"Actions shape after concatenation: {actions.shape}")
            print(f"Old Log Probs shape before concatenation: {old_log_probs[0].shape}")
            old_log_probs = torch.cat(old_log_probs, dim=0).view(-1, 1)
            print(f"Old Log Probs shape after concatenation: {old_log_probs.shape}")
            values = torch.cat(values, dim=0).view(-1, 1)

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

                loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - value).pow(2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
