"""
Base Agent Class for Multi-Agent Reinforcement Learning

This abstract class defines the interface that all MARL algorithms must implement.
It provides common functionality for:
- Neural network initialization
- Experience storage and retrieval
- Learning and action selection
- Model saving/loading
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for MARL agents.
    
    All CTDE algorithms (MADDPG, MAPPO, QMIX) inherit from this class
    and implement their specific training logic.
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int, config: Dict):
        """
        Initialize base agent.
        
        Args:
            state_dim (int): Dimension of state/observation space
            action_dim (int): Dimension of action space
            agent_id (int): Unique identifier for this agent
            config (Dict): Configuration dictionary with hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.001)  # Soft update rate
        
        # Device (CPU/GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Exploration
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action based on state.
        
        Args:
            state (np.ndarray): Current observation/state
            training (bool): Whether in training or evaluation mode
            
        Returns:
            np.ndarray: Action to take
        """
        pass
    
    @abstractmethod
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute loss for gradient update.
        
        Args:
            batch (Dict): Batch of experiences from replay buffer
            
        Returns:
            torch.Tensor: Loss value
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict) -> float:
        """
        Update agent networks and return loss value.
        
        Args:
            batch (Dict): Batch of experiences
            
        Returns:
            float: Loss value for logging
        """
        pass
    
    def soft_update(self, source_net: nn.Module, target_net: nn.Module):
        """
        Soft update of target network using source network.
        Implements: target = tau * source + (1-tau) * target

        Args:
            source_net (nn.Module): Source network to copy from
            target_net (nn.Module): Target network to update
        """
        for source_param, target_param in zip(source_net.parameters(), 
                                              target_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_exploration(self):
        """Decay exploration rate (epsilon) for epsilon-greedy policy."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path: str):
        """
        Save agent's neural networks to file.
        
        Args:
            path (str): Directory path to save models
        """
        raise NotImplementedError("Subclass must implement save_model()")
    
    def load_model(self, path: str):
        """
        Load agent's neural networks from file.
        
        Args:
            path (str): Directory path to load models from
        """
        raise NotImplementedError("Subclass must implement load_model()")


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron network used in MARL algorithms.
    
    Architecture:
        Input → Dense(256, ReLU) → Dense(256, ReLU) → Dense(output_dim)
    
    This is the standard network for policy and value functions in MARL.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """
        Initialize MLP network.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dim (int): Hidden layer dimension (default: 256)
        """
        super(MLPNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.net(x)


class NormalizedMLP(nn.Module):
    """
    MLP with input normalization for better training stability.
    
    This network applies layer normalization to hidden layers which helps with:
    - Training stability
    - Faster convergence
    - Better generalization
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """
        Initialize normalized MLP.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dim (int): Hidden layer dimension
        """
        super(NormalizedMLP, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalized network."""
        return self.net(x)


class GaussianActor(nn.Module):
    """
    Actor network that outputs Gaussian distribution parameters.
    
    Used in continuous action space algorithms (MADDPG, MAPPO).
    Outputs both mean and log_std of a Gaussian distribution.
    
    This allows for:
    - Smooth exploration through stochastic policies
    - Better gradient flow
    - Adjustable exploration (via log_std)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 log_std_bounds: Tuple[float, float] = (-20, 2)):
        """
        Initialize Gaussian actor.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
            log_std_bounds (Tuple): Bounds for log standard deviation
        """
        super(GaussianActor, self).__init__()
        
        self.log_std_bounds = log_std_bounds
        self.action_dim = action_dim
        
        # Mean network
        self.mean_net = MLPNetwork(state_dim, action_dim, hidden_dim)
        
        # Log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - returns mean and std of Gaussian distribution.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            Tuple: (mean, std) of action distribution
        """
        mean = self.mean_net(state)
        log_std = torch.clamp(self.log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from distribution and compute log probability.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            Tuple: (action, log_prob, mean)
        """
        mean, std = self.forward(state)
        
        # Sample from distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        
        # Squash to [-1, 1] using tanh and correct log_prob
        log_prob = dist.log_prob(action)
        action = torch.tanh(action)
        
        # Correct log probability for tanh squashing
        log_prob -= (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))).sum(dim=-1)
        
        return action, log_prob, mean


def hard_update(source_net: nn.Module, target_net: nn.Module):
    """
    Hard update: directly copy weights from source to target.
    
    Args:
        source_net (nn.Module): Source network
        target_net (nn.Module): Target network to update
    """
    for source_param, target_param in zip(source_net.parameters(), 
                                          target_net.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update(source_net: nn.Module, target_net: nn.Module, tau: float):
    """
    Soft update: blend source and target weights.
    
    Formula: target = tau * source + (1-tau) * target
    
    This creates a moving average of network weights, preventing
    instability from rapid target network changes.
    
    Args:
        source_net (nn.Module): Source network
        target_net (nn.Module): Target network
        tau (float): Update rate (0 to 1). Higher = more aggressive updates.
    """
    for source_param, target_param in zip(source_net.parameters(), 
                                          target_net.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
