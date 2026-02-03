"""
Utility functions for MARL implementations.

Includes:
- Normalization functions
- Reward scaling
- Logging utilities
- Plotting functions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pathlib import Path


class StateNormalizer:
    """
    Normalizes states to have zero mean and unit variance.
    
    This helps with training stability by keeping inputs to neural networks
    in a reasonable range.
    
    Algorithm:
    1. Compute running mean and variance
    2. Normalize: x_norm = (x - mean) / sqrt(variance + eps)
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Initialize state normalizer.
        
        Args:
            dim (int): Dimension of state
            eps (float): Small constant for numerical stability
        """
        self.dim = dim
        self.eps = eps
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 0
    
    def update(self, state: np.ndarray):
        """
        Update running statistics.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            state (np.ndarray): State to update with
        """
        batch_mean = np.mean(state, axis=0)
        batch_var = np.var(state, axis=0)
        batch_count = state.shape[0]
        
        self.count += batch_count
        
        # Update mean
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / self.count
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / self.count
        self.var = M2 / self.count
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state.
        
        Args:
            state (np.ndarray): State to normalize
            
        Returns:
            np.ndarray: Normalized state
        """
        return (state - self.mean) / np.sqrt(self.var + self.eps)


class RewardScaler:
    """
    Scales rewards to have standard deviation 1.
    
    Helps with training stability by preventing reward explosion/vanishment.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize reward scaler.
        
        Args:
            eps (float): Small constant for stability
        """
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, rewards: np.ndarray):
        """Update running statistics."""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        self.count += batch_count
        
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / self.count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / self.count
        self.var = M2 / self.count
    
    def scale(self, reward: float) -> float:
        """Scale reward."""
        return reward / np.sqrt(self.var + self.eps)


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted cumulative returns.
    
    G_t = sum_k=0^∞ γ^k * r_t+k
    
    Args:
        rewards (np.ndarray): Array of rewards
        gamma (float): Discount factor
        
    Returns:
        np.ndarray: Discounted returns
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    
    # Compute backwards through episode
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, 
                gae_lambda: float = 0.95, dones: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE provides a good balance between bias (short TD) and variance (MC returns).
    
    Formula:
    A_t = sum_l=0^∞ (γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards (np.ndarray): Array of rewards
        values (np.ndarray): Array of value estimates
        gamma (float): Discount factor
        gae_lambda (float): GAE parameter (0=TD, 1=MC)
        dones (np.ndarray): Done flags for episode boundaries
        
    Returns:
        Tuple: (advantages, returns)
    """
    if dones is None:
        dones = np.zeros_like(rewards)
    
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_done = 1
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]
        
        # TD residual
        td_residual = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        
        # GAE accumulation
        gae = td_residual + gamma * gae_lambda * gae * (1 - next_done)
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def plot_training_curve(rewards: List[float], title: str, save_path: str = None):
    """
    Plot training rewards curve.
    
    Args:
        rewards (List[float]): List of episode rewards
        title (str): Title of plot
        save_path (str): Path to save figure (if provided)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Plot moving average
    window = min(100, len(rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Avg (window={window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_multi_agent_rewards(all_rewards: Dict[int, List[float]], title: str, save_path: str = None):
    """
    Plot rewards for multiple agents.
    
    Args:
        all_rewards (Dict): Dict mapping agent_id to list of rewards
        title (str): Title of plot
        save_path (str): Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_rewards)))
    
    for (agent_id, rewards), color in zip(all_rewards.items(), colors):
        plt.plot(rewards, alpha=0.3, color=color, label=f'Agent {agent_id}')
        
        # Moving average
        window = min(100, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, color=color, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_comparison(data: Dict[str, List[float]], title: str, save_path: str = None):
    """
    Plot comparison of multiple algorithms/runs.
    
    Args:
        data (Dict): Dict mapping names to lists of values
        title (str): Title of plot
        save_path (str): Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in data.items():
        plt.plot(values, alpha=0.5, label=name)
        
        # Moving average
        window = min(50, len(values) // 10)
        if window > 1:
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(values)), moving_avg, linewidth=2, label=f'{name} (MA)')
    
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


class Logger:
    """Simple logger for training statistics."""
    
    def __init__(self, keys: List[str]):
        """
        Initialize logger.
        
        Args:
            keys (List[str]): Names of values to log
        """
        self.keys = keys
        self.data = {key: [] for key in keys}
    
    def log(self, **kwargs):
        """
        Log values.
        
        Args:
            **kwargs: Key-value pairs to log
        """
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
    
    def get(self, key: str) -> List:
        """Get logged values for a key."""
        return self.data.get(key, [])
    
    def print_summary(self, n_last: int = None):
        """Print summary of logged values."""
        if n_last is None:
            n_last = len(list(self.data.values())[0]) if self.data else 0
        
        print("\n" + "="*60)
        for key in self.keys:
            values = self.data.get(key, [])
            if values:
                recent = values[-n_last:] if n_last > 0 else values
                avg = np.mean(recent)
                std = np.std(recent)
                print(f"{key:20s}: {avg:10.4f} ± {std:10.4f}")
        print("="*60 + "\n")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
