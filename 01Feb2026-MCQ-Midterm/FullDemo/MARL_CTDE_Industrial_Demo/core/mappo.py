"""
MAPPO: Multi-Agent Proximal Policy Optimization

MAPPO extends PPO to multi-agent settings with CTDE architecture.

Key Ideas:
1. Uses policy gradient instead of Q-learning
2. Each agent has its own actor (policy) network
3. CENTRALIZED critic network that sees all agents' observations
4. Clipped surrogate objective for stable updates
5. Better sample efficiency through shared value baseline

Architecture:
    Actor(πᵢ):     observation_i → action_distribution
    Critic(V):     [obs_1,...,obs_n] → value_estimate
    
Loss Function:
    L_actor = -E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
    L_critic = E[(V(s) - V_target)²]
    
Where:
    r_t = π_new(a|s) / π_old(a|s)    (probability ratio)
    A_t = return - V(s)               (advantage)

Advantages:
    ✓ Sample efficient (policy gradient + baseline)
    ✓ Stable training (clipped surrogate objective)
    ✓ Works well with continuous and discrete actions
    ✓ Centralized value reduces variance
    ✓ Good for cooperative multi-agent problems
    ✗ Slightly more complex than MADDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from typing import Dict, Tuple, List
from .base_agent import BaseAgent, MLPNetwork


class MAPPOActor(nn.Module):
    """
    Actor (Policy) network for MAPPO.
    
    Outputs mean and log_std of Gaussian distribution for continuous actions.
    This allows for stochastic policies with learnable exploration.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize actor network.
        
        Args:
            state_dim (int): Dimension of local observation
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
        """
        super(MAPPOActor, self).__init__()
        
        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Log standard deviation (learnable, shared across actions)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute mean and std of policy distribution.
        
        Args:
            state (torch.Tensor): Local observation
            
        Returns:
            Tuple: (mean, std) of Gaussian distribution
        """
        mean = self.mean_net(state)
        
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution.
        
        Args:
            state (torch.Tensor): Local observation
            
        Returns:
            Tuple: (action, log_probability)
        """
        mean, std = self.forward(state)
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        
        # Compute log probability
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        # Clamp action to valid range
        action = torch.clamp(action, -1, 1)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of given action.
        
        Args:
            state (torch.Tensor): State
            action (torch.Tensor): Action
            
        Returns:
            Tuple: (log_prob, entropy)
        """
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    Centralized Critic (Value) network for MAPPO.
    
    The KEY to CTDE in MAPPO:
    - Takes concatenated observations from ALL agents
    - Outputs state value estimate
    - Helps with baseline subtraction and variance reduction
    
    By seeing all agents' observations, the critic can better estimate
    the value of the current joint state.
    """
    
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 256):
        """
        Initialize critic network.
        
        Args:
            state_dim (int): Dimension of single agent's observation
            num_agents (int): Number of agents
            hidden_dim (int): Hidden layer dimension
        """
        super(MAPPOCritic, self).__init__()
        
        # Input: concatenate all agents' observations
        self.input_dim = state_dim * num_agents
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute state value.
        
        Args:
            states (torch.Tensor): Concatenated observations from all agents
                                  (batch_size, state_dim*num_agents)
            
        Returns:
            torch.Tensor: State value estimates (batch_size, 1)
        """
        return self.net(states)


class MAPPOAgent(BaseAgent):
    """
    Single agent using MAPPO algorithm.
    
    Uses policy gradient with centralized value function baseline.
    The actor is local (decentralized) while critic is centralized (trained with all agents).
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int,
                 num_agents: int, config: Dict):
        """
        Initialize MAPPO agent.
        
        Args:
            state_dim (int): Dimension of local observation
            action_dim (int): Dimension of action space
            agent_id (int): Unique agent identifier
            num_agents (int): Total number of agents
            config (Dict): Configuration dictionary
        """
        super().__init__(state_dim, action_dim, agent_id, config)
        
        self.num_agents = num_agents
        
        # Actor network (local policy)
        self.actor = MAPPOActor(state_dim, action_dim).to(self.device)
        
        # Critic network (centralized value function)
        # Note: Critic is shared across agents, but we create one per agent for consistency
        self.critic = MAPPOCritic(state_dim, num_agents).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # PPO specific parameters
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.n_epochs = config.get('n_epochs', 5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Store old policy for ratio computation
        self.old_actor = MAPPOActor(state_dim, action_dim).to(self.device)
        self._copy_actor_weights()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """
        Select action using stochastic policy.
        
        Args:
            state (np.ndarray): Local observation
            training (bool): Whether in training mode
            
        Returns:
            Tuple: (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.sample(state_tensor)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().item()
        
        return action, log_prob
    
    def compute_actor_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute actor (policy) loss using PPO objective.
        
        L_actor = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        
        Where:
        - r_t: probability ratio (new policy / old policy)
        - A_t: advantage estimate
        - ε: clipping parameter
        
        Args:
            batch (Dict): Batch of episodes
            
        Returns:
            torch.Tensor: Actor loss
        """
        observations = batch['observations']  # (episode_length, state_dim)
        actions = batch['actions']            # (episode_length, action_dim)
        advantages = batch['advantages']      # (episode_length,)
        old_log_probs = batch['old_log_probs']  # (episode_length,)
        
        # Get log probs under new policy
        new_log_probs, entropy = self.actor.evaluate(observations, actions)
        
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Actor loss
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus for exploration
        actor_loss = actor_loss - self.entropy_coef * entropy.mean()
        
        return actor_loss
    
    def compute_critic_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute critic (value function) loss.
        
        L_critic = E[(V(s) - V_target)²]
        
        Args:
            batch (Dict): Batch of episodes with concatenated state observations
            
        Returns:
            torch.Tensor: Critic loss
        """
        states_concat = batch['states_concat']  # (episode_length, state_dim*num_agents)
        returns = batch['returns']              # (episode_length,)
        
        # Get value estimates
        values = self.critic(states_concat).squeeze(-1)
        
        # Critic loss: MSE
        critic_loss = nn.MSELoss()(values, returns)
        
        return critic_loss
    
    def update(self, batch: Dict) -> Tuple[float, float]:
        """
        Update actor and critic networks.
        
        Performs multiple epochs of updates on the batch.
        
        Args:
            batch (Dict): Batch of episodes
            
        Returns:
            Tuple: (avg_actor_loss, avg_critic_loss)
        """
        total_actor_loss = 0
        total_critic_loss = 0
        
        # Update for n_epochs
        for epoch in range(self.n_epochs):
            # Compute losses
            actor_loss = self.compute_actor_loss(batch)
            critic_loss = self.compute_critic_loss(batch)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Copy new policy to old policy
        self._copy_actor_weights()
        
        return total_actor_loss / self.n_epochs, total_critic_loss / self.n_epochs
    
    def _copy_actor_weights(self):
        """Copy weights from actor to old_actor."""
        for old_param, new_param in zip(self.old_actor.parameters(), self.actor.parameters()):
            old_param.data.copy_(new_param.data)
    
    def save_model(self, path: str):
        """Save actor and critic models."""
        torch.save(self.actor.state_dict(), f"{path}/actor_{self.agent_id}.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_{self.agent_id}.pt")
    
    def load_model(self, path: str):
        """Load actor and critic models."""
        self.actor.load_state_dict(torch.load(f"{path}/actor_{self.agent_id}.pt"))
        self.critic.load_state_dict(torch.load(f"{path}/critic_{self.agent_id}.pt"))


def create_mappo_agents(state_dim: int, action_dim: int, num_agents: int, config: Dict) -> list:
    """
    Create multiple MAPPO agents.
    
    Args:
        state_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        num_agents (int): Number of agents
        config (Dict): Configuration dictionary
        
    Returns:
        list: List of MAPPOAgent instances
    """
    agents = []
    for agent_id in range(num_agents):
        agent = MAPPOAgent(state_dim, action_dim, agent_id, num_agents, config)
        agents.append(agent)
    
    return agents
