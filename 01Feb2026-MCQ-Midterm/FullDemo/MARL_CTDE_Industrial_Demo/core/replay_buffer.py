"""
Experience Replay Buffer for Multi-Agent Learning

The replay buffer stores experiences (s, a, r, s', done) for all agents
and provides batches for training. This is critical for:

1. Breaking temporal correlations (i.i.d. assumption)
2. Sample efficiency (reuse experiences multiple times)
3. Stability (smoother gradient updates)
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Tuple, List
import random


class SharedReplayBuffer:
    """
    Shared replay buffer for all agents in MARL setting.
    
    Stores experiences for all agents and provides mini-batches
    of experiences for training. This is used in CTDE algorithms
    where we need global state information during training.
    
    Structure:
    - observations: (buffer_size, num_agents, state_dim)
    - actions: (buffer_size, num_agents, action_dim)
    - rewards: (buffer_size, num_agents)
    - next_observations: (buffer_size, num_agents, state_dim)
    - dones: (buffer_size, num_agents)
    """
    
    def __init__(self, buffer_size: int, num_agents: int, state_dim: int, 
                 action_dim: int, device: torch.device = None):
        """
        Initialize shared replay buffer.
        
        Args:
            buffer_size (int): Maximum number of transitions to store
            num_agents (int): Number of agents
            state_dim (int): Dimension of state space per agent
            action_dim (int): Dimension of action space per agent
            device (torch.device): Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device('cpu')
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate storage arrays
        self.observations = np.zeros((buffer_size, num_agents, state_dim))
        self.actions = np.zeros((buffer_size, num_agents, action_dim))
        self.rewards = np.zeros((buffer_size, num_agents))
        self.next_observations = np.zeros((buffer_size, num_agents, state_dim))
        self.dones = np.zeros((buffer_size, num_agents))
        
    def add(self, observations: np.ndarray, actions: np.ndarray, 
            rewards: np.ndarray, next_observations: np.ndarray, dones: np.ndarray):
        """
        Add a transition to the buffer.
        
        Args:
            observations (np.ndarray): Current observations (num_agents, state_dim)
            actions (np.ndarray): Actions taken (num_agents, action_dim)
            rewards (np.ndarray): Rewards received (num_agents,)
            next_observations (np.ndarray): Next observations (num_agents, state_dim)
            dones (np.ndarray): Done flags (num_agents,)
        """
        # Circular buffer: overwrite oldest experience when full
        self.observations[self.ptr] = observations
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_observations
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences.
        
        Returns batch with shapes:
        - observations: (batch_size, num_agents, state_dim)
        - actions: (batch_size, num_agents, action_dim)
        - rewards: (batch_size, num_agents)
        - next_observations: (batch_size, num_agents, state_dim)
        - dones: (batch_size, num_agents)
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            Dict: Batch of experiences as torch tensors on specified device
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch size {batch_size}")
        
        # Sample random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Get samples and convert to tensors
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
        }
        
        return batch
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.
        
        Args:
            batch_size (int): Batch size to check against
            
        Returns:
            bool: True if buffer has at least batch_size experiences
        """
        return self.size >= batch_size
    
    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return self.size
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.ptr = 0
        self.size = 0
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_observations.fill(0)
        self.dones.fill(0)


class PrioritizedReplayBuffer(SharedReplayBuffer):
    """
    Prioritized Experience Replay (PER).
    
    Samples transitions with probability proportional to their TD-error magnitude.
    This allows the agent to focus on "interesting" experiences (where it was wrong).
    
    Benefits:
    - Faster learning: focuses on important transitions
    - Better sample efficiency
    - Can learn from rare, important events
    
    Implementation uses a simple priority scheme based on TD-error.
    """
    
    def __init__(self, buffer_size: int, num_agents: int, state_dim: int,
                 action_dim: int, alpha: float = 0.6, device: torch.device = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            buffer_size (int): Maximum buffer size
            num_agents (int): Number of agents
            state_dim (int): State dimension
            action_dim (int): Action dimension
            alpha (float): Priority exponent (0=uniform, 1=full prioritization)
            device (torch.device): Device for tensors
        """
        super().__init__(buffer_size, num_agents, state_dim, action_dim, device)
        
        self.priorities = np.ones(buffer_size)
        self.alpha = alpha
        self.max_priority = 1.0
    
    def add(self, observations: np.ndarray, actions: np.ndarray,
            rewards: np.ndarray, next_observations: np.ndarray, dones: np.ndarray):
        """Add experience with maximum priority."""
        super().add(observations, actions, rewards, next_observations, dones)
        
        # New experiences get maximum priority
        idx = (self.ptr - 1) % self.buffer_size
        self.priorities[idx] = self.max_priority
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch with probability proportional to priorities.
        
        Args:
            batch_size (int): Size of batch
            
        Returns:
            Dict: Batch with additional 'indices' and 'weights' keys for importance weighting
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch size {batch_size}")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices according to probabilities
        indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        
        # Get batch
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
            'indices': indices,  # Store indices for priority updates
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices (np.ndarray): Indices of experiences in batch
            td_errors (np.ndarray): Absolute TD-errors for these experiences
            epsilon (float): Small constant to avoid zero priority
        """
        new_priorities = np.abs(td_errors) + epsilon
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class EpisodeBuffer:
    """
    Buffer that stores experiences per episode.
    
    Used for algorithms like PPO that need to process complete episodes
    before computing gradients. This helps with:
    - Computing returns and advantages
    - Policy update calculations
    - Trajectory-based learning
    """
    
    def __init__(self, num_agents: int):
        """
        Initialize episode buffer.
        
        Args:
            num_agents (int): Number of agents
        """
        self.num_agents = num_agents
        self.clear()
    
    def clear(self):
        """Clear all episode data."""
        self.observations = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.values = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.dones = [[] for _ in range(self.num_agents)]
        self.next_observations = [[] for _ in range(self.num_agents)]
    
    def add(self, agent_id: int, observation: np.ndarray, action: np.ndarray,
            reward: float, value: float, log_prob: float, done: bool,
            next_observation: np.ndarray):
        """
        Add experience for an agent.
        
        Args:
            agent_id (int): Agent index
            observation (np.ndarray): State
            action (np.ndarray): Action taken
            reward (float): Reward received
            value (float): Value estimate
            log_prob (float): Log probability of action
            done (bool): Episode done flag
            next_observation (np.ndarray): Next state
        """
        self.observations[agent_id].append(observation)
        self.actions[agent_id].append(action)
        self.rewards[agent_id].append(reward)
        self.values[agent_id].append(value)
        self.log_probs[agent_id].append(log_prob)
        self.dones[agent_id].append(done)
        self.next_observations[agent_id].append(next_observation)
    
    def get_episode(self, agent_id: int) -> Dict:
        """
        Get complete episode data for an agent.
        
        Args:
            agent_id (int): Agent index
            
        Returns:
            Dict: Episode data with all trajectories
        """
        return {
            'observations': np.array(self.observations[agent_id]),
            'actions': np.array(self.actions[agent_id]),
            'rewards': np.array(self.rewards[agent_id]),
            'values': np.array(self.values[agent_id]),
            'log_probs': np.array(self.log_probs[agent_id]),
            'dones': np.array(self.dones[agent_id]),
            'next_observations': np.array(self.next_observations[agent_id]),
        }
    
    def get_all(self) -> List[Dict]:
        """Get episodes for all agents."""
        return [self.get_episode(i) for i in range(self.num_agents)]
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float = 0.95) -> Dict:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        This is crucial for stable policy gradient updates.
        
        Returns:
            Dict: Returns and advantages for each agent
        """
        results = {}
        
        for agent_id in range(self.num_agents):
            episode = self.get_episode(agent_id)
            rewards = episode['rewards']
            values = episode['values']
            dones = episode['dones']
            
            # Initialize returns and advantages
            returns = np.zeros_like(rewards)
            advantages = np.zeros_like(rewards)
            gae = 0
            
            # Compute GAE backwards through episode
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                # TD residual
                td_residual = rewards[t] + gamma * next_value - values[t]
                
                # GAE
                gae = td_residual + gamma * gae_lambda * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
            
            results[agent_id] = {
                'returns': returns,
                'advantages': advantages,
            }
        
        return results
