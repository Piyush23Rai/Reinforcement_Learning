"""
QMIX: Value Decomposition Networks

QMIX is a CTDE algorithm specifically designed for cooperative multi-agent learning.

Key Insight:
The optimal joint action is the combination of optimal individual actions.
This is achieved through value decomposition with a monotonic mixing network.

Architecture:
    Local Q-Networks: Each agent learns Q(s_i, a_i) independently
    Mixing Network: Combines local Q-values into global Q-value
    
Constraint (Monotonicity):
    Q_total must be monotonically increasing in individual Q-values:
    ∂Q_total / ∂Q_i ≥ 0  for all i
    
This ensures: argmax_joint(Q_total) = sum(argmax_i(Q_i))

Loss Function:
    L = E[(r + γmax_a' Q_total(s', a') - Q_total(s, a))²]

Where Q_total = MixingNetwork([Q_1, Q_2, ..., Q_n], state)

Advantages:
    ✓ Scalable: Only trains local Q-networks
    ✓ Implicit coordination: No communication needed
    ✓ Works for cooperative tasks
    ✓ Better scalability than MADDPG/MAPPO
    ✗ Requires monotonicity constraint
    ✗ Only for cooperative settings
    ✗ Discrete actions only
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
from .base_agent import BaseAgent, MLPNetwork


class LocalQNetwork(nn.Module):
    """
    Local Q-network for individual agent.
    
    Each agent learns Q(s_i, a_i) independently.
    This reduces the problem from O(|A|^n) to O(n|A|).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize local Q-network.
        
        Args:
            state_dim (int): Dimension of local observation
            action_dim (int): Number of actions
            hidden_dim (int): Hidden layer dimension
        """
        super(LocalQNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values for all actions.
        
        Args:
            state (torch.Tensor): Local observation (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Q-values for each action (batch_size, action_dim)
        """
        return self.net(state)


class MixingNetwork(nn.Module):
    """
    Mixing network that combines local Q-values into global Q-value.
    
    Key Property: MONOTONICITY
    The mixing network must be monotonically increasing in each input Q-value.
    This is enforced by using non-negative weights.
    
    Architecture:
        QMix network structure:
        1. Compute mixing weights from global state
        2. Apply weights to local Q-values with non-negative constraint
        3. Add bias term
        
    Formula:
        Q_total = sum_i(w_i * Q_i + b_i)
        
    Where w_i are computed from state and constrained to be ≥ 0.
    """
    
    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int = 32):
        """
        Initialize mixing network.
        
        Args:
            num_agents (int): Number of agents
            state_dim (int): Dimension of global state
            hidden_dim (int): Hidden dimension of weight networks
        """
        super(MixingNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Network to compute mixing weights (must output non-negative values)
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )
        
        # Bias term
        self.bias_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Mix local Q-values into global Q-value.
        
        Args:
            q_values (torch.Tensor): Local Q-values from all agents
                                    (batch_size, num_agents)
            state (torch.Tensor): Global state (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Global Q-value (batch_size, 1)
        """
        # Compute weights (non-negative)
        weights = torch.abs(self.weight_net(state))  # Absolute value ensures ≥ 0
        
        # Compute bias
        bias = self.bias_net(state)
        
        # Mix Q-values: Q_total = sum(w_i * Q_i) + bias
        # This is monotonic in Q-values because w_i ≥ 0
        mixed = (weights * q_values).sum(dim=1, keepdim=True) + bias
        
        return mixed


class QMIXAgent(BaseAgent):
    """
    Agent using QMIX algorithm.
    
    Each agent learns its local Q-function independently.
    During training, Q-values are mixed using a centralized mixing network
    that sees the global state.
    
    This enables decentralized execution (use local Q-function)
    with centralized training (via mixing network).
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int,
                 num_agents: int, config: Dict):
        """
        Initialize QMIX agent.
        
        Args:
            state_dim (int): Dimension of local observation
            action_dim (int): Number of discrete actions
            agent_id (int): Unique agent identifier
            num_agents (int): Total number of agents
            config (Dict): Configuration dictionary
        """
        super().__init__(state_dim, action_dim, agent_id, config)
        
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # Local Q-network for this agent
        self.local_q = LocalQNetwork(state_dim, action_dim).to(self.device)
        self.local_q_target = LocalQNetwork(state_dim, action_dim).to(self.device)
        
        # Mixing network (shared across all agents, but we keep one per agent)
        global_state_dim = config.get('global_state_dim', state_dim * num_agents)
        self.mixing = MixingNetwork(num_agents, global_state_dim).to(self.device)
        self.mixing_target = MixingNetwork(num_agents, global_state_dim).to(self.device)
        
        # Initialize target networks
        self._hard_update(self.local_q, self.local_q_target)
        self._hard_update(self.mixing, self.mixing_target)
        
        # Optimizers
        params = list(self.local_q.parameters()) + list(self.mixing.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy on local Q-values.
        
        This is the DECENTRALIZED part - only uses local Q-network.
        
        Args:
            state (np.ndarray): Local observation
            training (bool): Whether in training mode
            
        Returns:
            np.ndarray: Action index (0 to action_dim-1)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return np.array([np.random.randint(0, self.action_dim)])
        
        # Exploitation: greedy action from local Q-network
        with torch.no_grad():
            q_values = self.local_q(state_tensor)
            action = q_values.max(dim=1)[1].cpu().numpy()
        
        return action
    
    def compute_loss(self, batch: Dict, other_local_q_nets: List[nn.Module] = None) -> torch.Tensor:
        """
        Compute TD loss using target mixing network.
        
        CTDE Key Point:
        - Local Q-networks are used for exploration
        - Mixing network (sees global state) combines them for training
        
        Args:
            batch (Dict): Batch of experiences
            other_local_q_nets (List): Target Q-networks of other agents
            
        Returns:
            torch.Tensor: TD loss
        """
        observations = batch['observations']  # (batch_size, num_agents, state_dim)
        actions = batch['actions']            # (batch_size, num_agents, 1) - discrete
        rewards = batch['rewards']            # (batch_size, num_agents)
        next_observations = batch['next_observations']
        dones = batch['dones']
        
        batch_size = observations.shape[0]
        
        # Extract agent's data
        agent_obs = observations[:, self.agent_id, :]
        agent_action = actions[:, self.agent_id, :].long()
        agent_reward = rewards[:, self.agent_id]
        agent_next_obs = next_observations[:, self.agent_id, :]
        agent_done = dones[:, self.agent_id]
        
        # Flatten global state for mixing network
        global_state = observations.reshape(batch_size, -1)
        next_global_state = next_observations.reshape(batch_size, -1)
        
        # ===== Current Q-values =====
        # Get local Q-values for this agent
        local_q = self.local_q(agent_obs)  # (batch_size, action_dim)
        
        # Select Q-value for the action that was taken
        chosen_q = local_q.gather(1, agent_action)  # (batch_size, 1)
        
        # Get Q-values from all agents
        all_q_values = [chosen_q]
        for i in range(self.num_agents):
            if i != self.agent_id:
                other_obs = observations[:, i, :]
                if other_local_q_nets and i < len(other_local_q_nets):
                    q_i = other_local_q_nets[i](other_obs)
                    all_q_values.insert(i, q_i.max(dim=1, keepdim=True)[0])
        
        # Mix Q-values using mixing network
        all_q_values_tensor = torch.cat(all_q_values, dim=1)
        q_total = self.mixing(all_q_values_tensor, global_state)
        
        # ===== Target Q-values =====
        with torch.no_grad():
            # Get max Q-values for next state from all agents
            next_q_values = []
            for i in range(self.num_agents):
                if i == self.agent_id:
                    next_q_i = self.local_q_target(agent_next_obs)
                else:
                    other_next_obs = next_observations[:, i, :]
                    if other_local_q_nets and i < len(other_local_q_nets):
                        next_q_i = other_local_q_nets[i](other_next_obs)
                    else:
                        next_q_i = self.local_q_target(other_next_obs)
                
                next_q_values.append(next_q_i.max(dim=1, keepdim=True)[0])
            
            # Mix target Q-values
            next_q_total = torch.cat(next_q_values, dim=1)
            target_q = self.mixing_target(next_q_total, next_global_state)
            
            # Bellman target
            y = agent_reward.unsqueeze(1) + self.gamma * (1 - agent_done.unsqueeze(1)) * target_q
        
        # TD loss: MSE between current and target
        loss = nn.MSELoss()(q_total, y)
        
        return loss
    
    def update(self, batch: Dict, other_agents: List = None) -> float:
        """
        Update local Q-network and mixing network.
        
        Args:
            batch (Dict): Batch of experiences
            other_agents (List): Other agents (to get their Q-networks)
            
        Returns:
            float: Loss value for logging
        """
        # Get other agents' Q-networks
        other_q_nets = []
        if other_agents:
            for i, agent in enumerate(other_agents):
                if hasattr(agent, 'local_q_target'):
                    other_q_nets.append(agent.local_q_target)
        
        # Compute loss
        loss = self.compute_loss(batch, other_q_nets)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.local_q.parameters()) + list(self.mixing.parameters()), 1.0
        )
        self.optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.local_q, self.local_q_target)
        self._soft_update(self.mixing, self.mixing_target)
        
        return loss.item()
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def _hard_update(self, source: nn.Module, target: nn.Module):
        """Hard update target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)
    
    def save_model(self, path: str):
        """Save local Q-network and mixing network."""
        torch.save(self.local_q.state_dict(), f"{path}/local_q_{self.agent_id}.pt")
        torch.save(self.mixing.state_dict(), f"{path}/mixing_{self.agent_id}.pt")
    
    def load_model(self, path: str):
        """Load local Q-network and mixing network."""
        self.local_q.load_state_dict(torch.load(f"{path}/local_q_{self.agent_id}.pt"))
        self.mixing.load_state_dict(torch.load(f"{path}/mixing_{self.agent_id}.pt"))


def create_qmix_agents(state_dim: int, action_dim: int, num_agents: int, config: Dict) -> list:
    """
    Create multiple QMIX agents.
    
    Args:
        state_dim (int): Dimension of observation space
        action_dim (int): Number of discrete actions
        num_agents (int): Number of agents
        config (Dict): Configuration dictionary
        
    Returns:
        list: List of QMIXAgent instances
    """
    agents = []
    for agent_id in range(num_agents):
        agent = QMIXAgent(state_dim, action_dim, agent_id, num_agents, config)
        agents.append(agent)
    
    return agents
