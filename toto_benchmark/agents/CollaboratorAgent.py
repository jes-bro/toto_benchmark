# Import necessary modules
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import json
from collections import namedtuple
import sys

# Add the necessary paths to the system path if needed
# sys.path.append("/path/to/toto_benchmark")
# sys.path.append("/path/to/networks")

from toto_benchmark.networks.policies import MLPPolicyPG
from toto_benchmark.networks.critics import ValueCritic
from toto_benchmark.agents.Agent import Agent

# This namedtuple will help in storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class CollaboratorAgent(Agent):
    def __init__(self, config):
        super(CollaboratorAgent, self).__init__()

        # Agent configuration
        self.config = config
        self.ob_dim = config.agent.ob_dim
        self.ac_dim = config.agent.ac_dim
        self.discrete = config.agent.discrete
        self.n_layers = config.agent.n_layers  # Changed
        self.layer_size = config.agent.layer_size  # Changed
        self.gamma = config.agent.gamma  # Changed
        self.learning_rate = config.agent.learning_rate  # Changed
        self.use_baseline = config.agent.use_baseline  # Changed
        self.use_reward_to_go = config.agent.use_reward_to_go  # Changed
        self.baseline_learning_rate = config.agent.baseline_learning_rate  # Changed
        self.baseline_gradient_steps = config.agent.baseline_gradient_steps  # Changed
        self.gae_lambda = config.agent.gae_lambda  # Changed
        self.normalize_advantages = config.agent.normalize_advantages  # Changed

        # Initialize networks
        self.actor = MLPPolicyPG(self.ob_dim, self.ac_dim, self.discrete, self.n_layers, self.layer_size, self.learning_rate)
        if self.use_baseline:
            self.critic = ValueCritic(self.ob_dim, self.n_layers, self.layer_size, self.baseline_learning_rate)
        else:
            self.critic = None

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        if self.critic:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.baseline_learning_rate)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[i])
            advantages.insert(0, gae)
        return advantages

    def train(self, experiences):
        # Extract experiences
        states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))

        # Convert numpy arrays to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Compute values
        values = self.critic(states_t) if self.critic else None
        next_values = self.critic(next_states_t) if self.critic and not dones.all() else torch.zeros_like(rewards_t)

        # Compute advantages
        advantages = self.compute_advantages(rewards_t, values, next_values, dones_t)

        # Normalize advantages if enabled
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert advantages to tensor
        advantages_t = torch.FloatTensor(advantages)

        # Calculate actor (policy) loss
        log_probs = self.actor.get_log_probs(states_t, actions_t)
        actor_loss = -(log_probs * advantages_t).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic network
        if self.critic:
            critic_loss = (rewards_t + self.gamma * next_values * (1 - dones_t) - values).pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def predict(self, state):
        """
        Predict the action for a given state

        :param state: The current state
        :return: The action predicted by the policy network
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state_tensor)
        return action.numpy()[0]

    #@classmethod
def _init_agent_from_config(config, device):
        """
        Initialize the PGAgent from a configuration file

        :param config: The configuration settings or the file path to the configuration file
        :param device: The device to run the model on (e.g., 'cpu' or 'cuda')
        :return: An instance of the PGAgent
        """
        if isinstance(config, str):
            with open(config, 'r') as config_file:
                config = json.load(config_file)
        # Assuming the config dictionary includes necessary model initialization parameters
        return CollaboratorAgent(config)