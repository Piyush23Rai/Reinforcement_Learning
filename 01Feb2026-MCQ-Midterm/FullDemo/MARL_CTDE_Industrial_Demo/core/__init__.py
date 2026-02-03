"""
Core MARL algorithms and utilities.

This module provides:
- BaseAgent: Abstract base class for all agents
- MADDPG: Multi-Agent DDPG algorithm
- MAPPO: Multi-Agent PPO algorithm  
- QMIX: Q-value Mixing for cooperative MARL
- Replay buffers and utilities
"""

from .base_agent import BaseAgent, MLPNetwork, NormalizedMLP, GaussianActor, soft_update, hard_update
from .maddpg import MADDPGAgent, create_maddpg_agents
from .mappo import MAPPOAgent, create_mappo_agents
from .qmix import QMIXAgent, create_qmix_agents
from .replay_buffer import SharedReplayBuffer, PrioritizedReplayBuffer, EpisodeBuffer
from .utils import (
    StateNormalizer, RewardScaler, compute_discounted_returns, compute_gae,
    plot_training_curve, plot_multi_agent_rewards, plot_comparison, Logger, set_seed
)

__all__ = [
    'BaseAgent', 'MLPNetwork', 'NormalizedMLP', 'GaussianActor',
    'soft_update', 'hard_update',
    'MADDPGAgent', 'create_maddpg_agents',
    'MAPPOAgent', 'create_mappo_agents',
    'QMIXAgent', 'create_qmix_agents',
    'SharedReplayBuffer', 'PrioritizedReplayBuffer', 'EpisodeBuffer',
    'StateNormalizer', 'RewardScaler', 'compute_discounted_returns', 'compute_gae',
    'plot_training_curve', 'plot_multi_agent_rewards', 'plot_comparison',
    'Logger', 'set_seed'
]
