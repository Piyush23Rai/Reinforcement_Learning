"""
Retail Training Script: CTDE-MADDPG for Multi-Warehouse Inventory Optimization

This script demonstrates how to train multiple warehouse agents using MADDPG
with centralized training and decentralized execution (CTDE).

Training Flow:
1. Each warehouse observes its local state (inventory, demand, neighbor inventories)
2. Centralized replay buffer stores experiences from all warehouses
3. Training phase:
   - Critic sees all warehouses' observations and actions (centralized)
   - Computes Q-value of joint action
4. Execution phase:
   - Each warehouse acts independently using only local observation
   - No communication needed
5. Warehouses coordinate implicitly through learned critic feedback
"""

import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.maddpg import create_maddpg_agents
from core.replay_buffer import SharedReplayBuffer
from core.utils import Logger, set_seed, plot_training_curve, plot_multi_agent_rewards
from .environment import MultiWarehouseEnv


class RetailTrainer:
    """
    Trainer for multi-warehouse inventory optimization using CTDE-MADDPG.
    
    Key Components:
    1. Environment: Multi-warehouse inventory system
    2. Agents: MADDPG agents (one per warehouse)
    3. Replay Buffer: Shared buffer for all agents
    4. Training Loop: Standard RL training with experience replay
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        set_seed(config['seed'])
        
        # Environment
        self.env = MultiWarehouseEnv(
            num_warehouses=config['num_agents'],
            warehouse_capacity=config['warehouse_capacity'],
            demand_mean=config['demand_mean'],
            topology=config.get('topology', 'line')
        )
        
        # Agents (MADDPG)
        obs_dim = self.env.observation_space_size
        action_dim = self.env.action_space_size
        
        self.agents = create_maddpg_agents(
            state_dim=obs_dim,
            action_dim=action_dim,
            num_agents=config['num_agents'],
            config=config
        )
        
        # Shared replay buffer (CTDE: centralized training)
        self.replay_buffer = SharedReplayBuffer(
            buffer_size=config['buffer_size'],
            num_agents=config['num_agents'],
            state_dim=obs_dim,
            action_dim=action_dim,
            device=self.agents[0].device
        )
        
        # Logging
        self.logger = Logger(['episode_reward', 'episode_cost', 'avg_cost', 'exploration'])
        self.device = self.agents[0].device
    
    def train_episode(self) -> Tuple[float, float]:
        """
        Train one episode.
        
        Returns:
            Tuple: (total_reward, total_cost)
        """
        # Reset environment
        observations = self.env.reset()
        episode_reward = 0
        episode_cost = 0
        
        # Episode loop
        for step in range(self.config.get('max_steps_per_episode', 200)):
            # ===== ACT: Decentralized execution (each agent acts independently) =====
            actions = []
            for agent_id, agent in enumerate(self.agents):
                # Agent uses only its local observation
                action = agent.select_action(observations[agent_id], training=True)
                actions.append(action)
            
            actions = np.array(actions)
            
            # ===== ENVIRONMENT STEP =====
            next_observations, rewards, dones, info = self.env.step(actions)
            episode_reward += rewards.sum()
            episode_cost += info['total_cost']
            
            # ===== STORE EXPERIENCE =====
            # This is centralized: all agents' data stored together
            self.replay_buffer.add(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones
            )
            
            observations = next_observations
            
            if dones[0]:  # Episode done
                break
        
        # ===== TRAIN: Centralized training with replay buffer =====
        if self.replay_buffer.is_ready(self.config['batch_size']):
            self._train_step()
        
        # Decay exploration
        for agent in self.agents:
            agent.decay_exploration()
        
        return episode_reward, episode_cost
    
    def _train_step(self):
        """
        Perform one training step with batch from replay buffer.
        
        Key CTDE aspect:
        - Batch contains all agents' observations and actions
        - Critics see full state for training
        - Actors only use local observations
        """
        batch = self.replay_buffer.sample(self.config['batch_size'])
        
        # Train each agent
        for agent_id, agent in enumerate(self.agents):
            # Update with batch (critic sees all agents' data)
            actor_loss, critic_loss = agent.update(batch, self.agents)
    
    def train(self, num_episodes: int = 1000):
        """
        Train for specified number of episodes.
        
        Args:
            num_episodes (int): Number of training episodes
        """
        print(f"\n{'='*70}")
        print(f"Training CTDE-MADDPG for Multi-Warehouse Inventory Optimization")
        print(f"{'='*70}")
        print(f"Num agents: {self.config['num_agents']}")
        print(f"Num episodes: {num_episodes}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Buffer size: {self.config['buffer_size']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"{'='*70}\n")
        
        # Training loop
        episode_rewards = []
        episode_costs = []
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            total_reward, total_cost = self.train_episode()
            episode_rewards.append(total_reward)
            episode_costs.append(total_cost)
            
            # Log
            self.logger.log(
                episode_reward=total_reward,
                episode_cost=total_cost,
                avg_cost=total_cost / self.config['num_agents'],
                exploration=self.agents[0].epsilon
            )
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_cost = np.mean(episode_costs[-100:])
                print(f"\nEpisode {episode+1}/{num_episodes}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Avg Cost (last 100): {avg_cost:.2f}")
                print(f"  Exploration: {self.agents[0].epsilon:.4f}")
        
        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}\n")
        
        return episode_rewards, episode_costs
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate trained agents.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            
        Returns:
            Dict: Evaluation metrics
        """
        print(f"\nEvaluating trained agents for {num_episodes} episodes...")
        
        all_rewards = [[] for _ in range(self.config['num_agents'])]
        all_costs = [[] for _ in range(self.config['num_agents'])]
        all_stockouts = [[] for _ in range(self.config['num_agents'])]
        
        for episode in tqdm(range(num_episodes), desc="Evaluation"):
            observations = self.env.reset()
            
            while True:
                # Decentralized execution (no exploration, greedy)
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(observations[agent_id], training=False)
                    actions.append(action)
                
                actions = np.array(actions)
                next_observations, rewards, dones, info = self.env.step(actions)
                
                observations = next_observations
                
                if dones[0]:
                    break
            
            # Collect metrics
            for agent_id in range(self.config['num_agents']):
                all_rewards[agent_id].append(rewards[agent_id])
                all_costs[agent_id].append(info['total_cost'] / self.config['num_agents'])
                all_stockouts[agent_id].append(info['stockout'][agent_id])
        
        # Aggregate metrics
        metrics = {
            'avg_reward': np.mean([np.mean(r) for r in all_rewards]),
            'avg_cost': np.mean([np.mean(c) for c in all_costs]),
            'avg_stockout': np.mean([np.mean(s) for s in all_stockouts]),
            'total_reward': np.sum([np.sum(r) for r in all_rewards]),
            'agent_rewards': all_rewards,
            'agent_costs': all_costs,
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Avg Reward per Agent: {metrics['avg_reward']:.2f}")
        print(f"  Avg Cost per Agent: {metrics['avg_cost']:.2f}")
        print(f"  Avg Stockout per Agent: {metrics['avg_stockout']:.2f}")
        
        return metrics
    
    def save_models(self, path: str):
        """Save trained models."""
        Path(path).mkdir(parents=True, exist_ok=True)
        for agent in self.agents:
            agent.save_model(path)
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models."""
        for agent in self.agents:
            agent.load_model(path)
        print(f"Models loaded from {path}")


def main():
    """Main training script."""
    
    # Configuration
    config = {
        'num_agents': 5,
        'num_episodes': 1000,
        'max_steps_per_episode': 200,
        'batch_size': 64,
        'buffer_size': 10000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'tau': 0.001,  # Soft update rate
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'warehouse_capacity': 1000.0,
        'demand_mean': 500.0,
        'seed': 42,
    }
    
    # Create trainer
    trainer = RetailTrainer(config)
    
    # Train
    train_rewards, train_costs = trainer.train(num_episodes=config['num_episodes'])
    
    # Evaluate
    metrics = trainer.evaluate(num_episodes=100)
    
    # Save models
    results_dir = Path(__file__).parent / 'results'
    trainer.save_models(str(results_dir / 'models'))
    
    # Plot results
    plot_training_curve(
        train_rewards,
        "CTDE-MADDPG: Multi-Warehouse Training",
        str(results_dir / 'training_rewards.png')
    )
    
    plot_training_curve(
        train_costs,
        "CTDE-MADDPG: Training Costs",
        str(results_dir / 'training_costs.png')
    )
    
    # Save metrics
    import json
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump({
            'config': config,
            'metrics': {k: v for k, v in metrics.items() if k not in ['agent_rewards', 'agent_costs']}
        }, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
