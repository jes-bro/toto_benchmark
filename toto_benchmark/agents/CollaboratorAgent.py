"""
If you are contributing a new agent, implement your 
agent initialization & predict functions here. 

Next, (optionally) update your agent's name in agent/__init__.py.
"""

import numpy as np
import sys
import torch
import os
#os.chdir('/home/jess/toto_benchmark/toto_benchmark')
import torchvision.transforms as T
from typing import Optional, Sequence
import numpy as np
#sys.path.append("/home/jess/toto_benchmark/toto_benchmark/networks")
#toto_benchmark/networks
from toto_benchmark.networks.policies import MLPPolicyPG
from toto_benchmark.networks.critics import ValueCritic
from torch import nn

from .Agent import Agent
NUM_JOINTS = 7


class CollaboratorAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages


    def predict(self, observation: dict):
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(np.array(observation['observation'])).float()
        # Sample an action from the policy
        action, _ = self.actor.sample_action(obs_tensor)
        return action.cpu().numpy()

def _init_agent_from_config(config):
    # Extract agent configuration
    agent_config = config['agent']
    
    # Initialize PGAgent with parameters from config
    agent = CollaboratorAgent(
        ob_dim=agent_config['ob_dim'],
        ac_dim=agent_config['ac_dim'],
        discrete=agent_config['discrete'],
        n_layers=agent_config['n_layers'],
        layer_size=agent_config['layer_size'],
        gamma=agent_config['gamma'],
        learning_rate=agent_config['learning_rate'],
        use_baseline=agent_config['use_baseline'],
        use_reward_to_go=agent_config['use_reward_to_go'],
        baseline_learning_rate=agent_config['baseline_learning_rate'],
        baseline_gradient_steps=agent_config['baseline_gradient_steps'],
        gae_lambda=agent_config['gae_lambda'],
        normalize_advantages=agent_config['normalize_advantages'],
    )
    
    # Initialize any other components such as the vision model, if applicable
    # ...
    
    return agent
