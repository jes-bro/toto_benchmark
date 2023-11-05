import torch
from torch.utils.data import DataLoader
from toto_benchmark.agents.BCAgent import BCAgent, Policy
from toto_benchmark.agents.Agent import Agent
import os 
import json

class CollaboratorAgent(Agent):
    def __init__(self, num_agents, cfg, device='cpu'):
        super(CollaboratorAgent, self).__init__()
        self.agents = [self._init_agent_from_config(cfg, device) for _ in range(num_agents)]

    def _init_agent_from_config(self, config, device):
        # Create the model
        policy_model = Policy(
            inp_dim=config['data']['in_dim'],
            out_dim=config['data']['out_dim'],
            hidden_dim=config['agent'].get('hidden_dim', 128)  # use the hidden_dim from config or default to 128
        )
        
        # Move model to device
        policy_model.to(device)
        
        # Initialize the BCAgent with the created model and other parameters from the config
        bc_agent = BCAgent(
            models={'decoder': policy_model},
            learning_rate=config['training']['lr'],
            device=device,
            H=config['data'].get('H', 1)  # use the horizon 'H' from config or default to 1
        )
        
        # If the config specifies pre-trained weights, load them
        if 'weights_path' in config:
            bc_agent.load(config['weights_path'])
        
        return bc_agent
    
    def predict(self, inputs):
            # Assuming each agent returns a prediction of the same shape
            # and that you want to average the predictions:
            predictions = [agent.predict(inputs) for agent in self.agents]
            average_prediction = torch.mean(torch.stack(predictions), dim=0)
            return average_prediction

    def train(self, data_loader: DataLoader):
        # Put all agents in training mode
        for agent in self.agents:
            agent.models['decoder'].train()

        total_loss = 0.0
        for sample in data_loader:
            for agent in self.agents:
                agent.optimizer.zero_grad()  # Reset gradients for the agent
                agent.compute_loss(sample)   # Compute loss for the given batch
                agent.loss.backward()        # Backpropagate the loss
                agent.optimizer.step()       # Update the model parameters
                total_loss += agent.loss.item()  # Accumulate the loss from the agent

        average_loss = total_loss / (len(self.agents) * len(data_loader))
        return average_loss

    def evaluate(self, data_loader: DataLoader):
        # Put all agents in evaluation mode
        for agent in self.agents:
            agent.models['decoder'].eval()

        total_loss = 0.0
        with torch.no_grad():
            for sample in data_loader:
                for agent in self.agents:
                    agent.compute_loss(sample)  # Compute loss for the given batch
                    total_loss += agent.loss.item()  # Accumulate the loss from the agent

        average_loss = total_loss / (len(self.agents) * len(data_loader))
        return average_loss

    def save_models(self, save_directory):
        for idx, agent in enumerate(self.agents):
            agent.save(os.path.join(save_directory, f'agent_{idx}.pth'))

    def load_models(self, load_directory):
        for idx, agent in enumerate(self.agents):
            agent.load(os.path.join(load_directory, f'agent_{idx}.pth'))

    # Implement any additional methods needed


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