"""
MADDPG: Multi-Agent Deep Deterministic Policy Gradient

MADDPG is a CTDE algorithm that extends DDPG to multi-agent settings.

Key Ideas:
1. Each agent has its own actor network (local policy)
2. Each agent has its own critic network, BUT the critic sees all agents' observations and actions
3. During training, critics are centralized (see full state)
4. During execution, actors are decentralized (use only local observations)

Architecture:
    Actor(πᵢ):     observation_i → action_i
    Critic(Qᵢ):    [obs_1,...,obs_n, act_1,...,act_n] → Q-value
    
Loss Function:
    L_actor = -E[Q(s, a)]                              (maximize Q-value)
    L_critic = E[(r + γQ(s', a') - Q(s, a))²]        (minimize TD-error)

Advantages:
    ✓ Works with continuous action spaces
    ✓ Centralized critic helps learn from other agents' actions
    ✓ Deterministic policy enables stable gradient computation
    ✓ Good sample efficiency
    ✗ Requires other agents' observations and actions during training
    ✗ Not suitable for large action dimensions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from .base_agent import BaseAgent, MLPNetwork, soft_update


class MADDPGActor(nn.Module):
    """
    Actor network for MADDPG.
    
    Maps agent's local observation to deterministic action.
    Uses tanh activation to bound output to [-1, 1].
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize actor network.
        
        Args:
            state_dim (int): Dimension of local observation
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
        """
        super(MADDPGActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map state to action.
        
        Args:
            state (torch.Tensor): Local observation
            
        Returns:
            torch.Tensor: Deterministic action in [-1, 1]
        """
        return self.net(state)


class MADDPGCritic(nn.Module):
    """
    Critic network for MADDPG.
    
    This is the KEY to CTDE in MADDPG:
    - Takes concatenated observations and actions from ALL agents
    - Outputs Q-value of the joint action
    - Sees the full state during training
    
    This helps the critic understand how other agents' actions affect rewards,
    enabling better value estimates and gradient signals.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 256):
        """
        Initialize critic network.
        
        Args:
            state_dim (int): Dimension of single agent's observation
            action_dim (int): Dimension of single agent's action
            num_agents (int): Number of agents
            hidden_dim (int): Hidden layer dimension
        """
        super(MADDPGCritic, self).__init__()
        
        # Input: concatenate all agents' observations and actions
        self.input_dim = state_dim * num_agents + action_dim * num_agents
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-value.
        
        Args:
            states (torch.Tensor): Concatenated observations from all agents (batch_size, state_dim*num_agents)
            actions (torch.Tensor): Concatenated actions from all agents (batch_size, action_dim*num_agents)
            
        Returns:
            torch.Tensor: Q-values (batch_size, 1)
        """
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


class MADDPGAgent(BaseAgent):
    """
    Single agent using MADDPG algorithm.
    
    This agent learns in a multi-agent environment using:
    - Local actor network (decentralized execution)
    - Centralized critic network (for training)
    
    The critic sees all agents' states and actions during training,
    but the actor only uses local observation during execution.
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int, 
                 num_agents: int, config: Dict):
        """
        Initialize MADDPG agent.
        
        Args:
            state_dim (int): Dimension of local observation
            action_dim (int): Dimension of action space
            agent_id (int): Unique agent identifier
            num_agents (int): Total number of agents
            config (Dict): Configuration dictionary
        """
        super().__init__(state_dim, action_dim, agent_id, config)
        
        self.num_agents = num_agents
        
        # Actor network (local - uses only this agent's observation)
        self.actor = MADDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target = MADDPGActor(state_dim, action_dim).to(self.device)
        
        # Critic network (centralized - sees all agents)
        self.critic = MADDPGCritic(state_dim, action_dim, num_agents).to(self.device)
        self.critic_target = MADDPGCritic(state_dim, action_dim, num_agents).to(self.device)
        
        # Initialize target networks with same weights
        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using local actor network.
        
        This is the DECENTRALIZED part - only uses local observation.
        
        Args:
            state (np.ndarray): Local observation
            training (bool): Whether in training mode (add noise) or evaluation
            
        Returns:
            np.ndarray: Action in [-1, 1]
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Forward through actor
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise during training
        if training and np.random.random() < self.epsilon:
            action = np.random.uniform(-1, 1, size=action.shape)
        
        # Add small noise for smoothness
        if training:
            action += np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1, 1)
        
        return action
    
    def compute_loss(self, batch: Dict, other_agents: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actor and critic losses.
        
        CTDE Key Point: Critic uses observations and actions from ALL agents.
        
        Args:
            batch (Dict): Batch of experiences (from shared replay buffer)
            other_agents (list): Other MADDPG agents (needed to get their actions for critic)
            
        Returns:
            Tuple: (actor_loss, critic_loss)
        """
        observations = batch['observations']  # (batch_size, num_agents, state_dim)
        actions = batch['actions']            # (batch_size, num_agents, action_dim)
        rewards = batch['rewards']            # (batch_size, num_agents)
        next_observations = batch['next_observations']
        dones = batch['dones']
        
        batch_size = observations.shape[0]
        
        # Extract this agent's data
        agent_obs = observations[:, self.agent_id, :]       # (batch_size, state_dim)
        agent_action = actions[:, self.agent_id, :]         # (batch_size, action_dim)
        agent_reward = rewards[:, self.agent_id]            # (batch_size,)
        agent_next_obs = next_observations[:, self.agent_id, :]
        agent_done = dones[:, self.agent_id]
        
        # Reshape for critic input (ensure contiguous memory to avoid view/backward issues)
        obs_flat = observations.reshape(batch_size, -1).contiguous()     # (batch_size, state_dim*num_agents)
        action_flat = actions.reshape(batch_size, -1).contiguous()       # (batch_size, action_dim*num_agents)
        next_obs_flat = next_observations.reshape(batch_size, -1).contiguous()
        
        # ===== CRITIC LOSS =====
        # Compute target Q-value using target networks
        with torch.no_grad():
            # Get target actions from ALL agents using their target actors
            target_actions_list = []
            for i in range(self.num_agents):
                if i == self.agent_id:
                    target_actions_list.append(self.actor_target(next_observations[:, i, :]))
                else:
                    target_actions_list.append(other_agents[i].actor_target(next_observations[:, i, :]))
            
            target_actions_flat = torch.cat(target_actions_list, dim=1)
            
            # Compute target Q using next states and target actions
            target_q = self.critic_target(next_obs_flat, target_actions_flat)
            
            # Bellman target
            y = agent_reward.unsqueeze(1) + self.gamma * (1 - agent_done.unsqueeze(1)) * target_q
        
        # Compute current Q-value
        current_q = self.critic(obs_flat, action_flat)
        
        # Critic loss: MSE between current and target Q
        critic_loss = nn.MSELoss()(current_q, y)
        
        # ===== ACTOR LOSS =====
        # Get current actions from all agents
        # When computing actor loss for this agent, other agents' actions should be treated as constants
        # (no gradient should flow into other agents' actors). Detach their outputs.
        current_actions_list = []
        for i in range(self.num_agents):
            if i == self.agent_id:
                current_actions_list.append(self.actor(observations[:, i, :]))
            else:
                # Detach other agents' actor outputs so gradients do not propagate to them
                current_actions_list.append(other_agents[i].actor(observations[:, i, :]).detach())

        current_actions_flat = torch.cat(current_actions_list, dim=1)

        # Actor loss: negative expected Q-value (we want to maximize Q)
        actor_loss = -self.critic(obs_flat, current_actions_flat).mean()
        
        return actor_loss, critic_loss
    
    def update(self, batch: Dict, other_agents: list = None) -> Tuple[float, float]:
        """
        Update actor and critic networks.
        
        Args:
            batch (Dict): Batch of experiences
            other_agents (list): Other agents (for getting their target actions)
            
        Returns:
            Tuple: (actor_loss, critic_loss) for logging
        """
        # Compute losses
        actor_loss, critic_loss = self.compute_loss(batch, other_agents)
        
        # Update critic (enable anomaly detection to help trace any in-place errors)
        self.critic_optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        # Freeze critic parameters so gradients from actor update don't propagate to critic
        for p in self.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # Soft update target networks
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)
        
        return actor_loss.item(), critic_loss.item()
    
    def _hard_update(self, source: nn.Module, target: nn.Module):
        """Hard copy weights from source to target."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)
    
    def save_model(self, path: str):
        """Save actor and critic models."""
        torch.save(self.actor.state_dict(), f"{path}/actor_{self.agent_id}.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_{self.agent_id}.pt")
    
    def load_model(self, path: str):
        """Load actor and critic models."""
        self.actor.load_state_dict(torch.load(f"{path}/actor_{self.agent_id}.pt"))
        self.critic.load_state_dict(torch.load(f"{path}/critic_{self.agent_id}.pt"))


def create_maddpg_agents(state_dim: int, action_dim: int, num_agents: int, config: Dict) -> list:
    """
    Create multiple MADDPG agents.
    
    Args:
        state_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        num_agents (int): Number of agents
        config (Dict): Configuration dictionary
        
    Returns:
        list: List of MADDPGAgent instances
    """
    agents = []
    for agent_id in range(num_agents):
        agent = MADDPGAgent(state_dim, action_dim, agent_id, num_agents, config)
        agents.append(agent)
    
    return agents
