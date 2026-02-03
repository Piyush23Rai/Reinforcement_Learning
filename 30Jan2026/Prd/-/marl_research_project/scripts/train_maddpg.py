#!/usr/bin/env python3
"""
MADDPG Training Script for Multi-Agent Reinforcement Learning Research
=======================================================================

This script demonstrates production-ready MADDPG training with:
- CTDE (Centralized Training, Decentralized Execution)
- Multi-Agent Replay Buffer with random sampling
- Experiment tracking with Weights & Biases
- Model checkpointing and evaluation

Usage:
    python train_maddpg.py --env simple_spread --episodes 5000
    python train_maddpg.py --env simple_tag --episodes 10000 --wandb
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import trange

# AgileRL for MADDPG
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

# PettingZoo environments
from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_adversary_v3

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ENVIRONMENTS = {
    "simple_spread": {
        "creator": lambda: simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True),
        "description": "3 cooperative agents must cover 3 landmarks",
        "type": "cooperative"
    },
    "simple_tag": {
        "creator": lambda: simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=True),
        "description": "3 predators chase 1 prey (mixed cooperative/competitive)",
        "type": "mixed"
    },
    "simple_adversary": {
        "creator": lambda: simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True),
        "description": "1 adversary vs N agents protecting a target",
        "type": "competitive"
    }
}

DEFAULT_CONFIG = {
    # Training
    "total_episodes": 5000,
    "max_steps": 25,
    "batch_size": 256,
    "buffer_size": 100000,
    "learn_start": 1000,  # Min samples before learning
    
    # MADDPG hyperparameters
    "lr_actor": 1e-4,
    "lr_critic": 1e-3,
    "gamma": 0.95,
    "tau": 0.01,
    "expl_noise": 0.1,
    
    # Network architecture
    "hidden_sizes": [64, 64],
    
    # Logging
    "log_interval": 100,
    "save_interval": 1000,
    "eval_episodes": 10,
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class MADDPGTrainer:
    """
    MADDPG Trainer implementing CTDE paradigm
    
    Key concepts demonstrated:
    1. Centralized Training: Critics see all observations and actions
    2. Decentralized Execution: Actors only use local observations
    3. Replay Buffer: Stores joint experiences, samples randomly
    """
    
    def __init__(self, env_name, config, use_wandb=False, device=None):
        self.config = {**DEFAULT_CONFIG, **config}
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment
        self.env_name = env_name
        self.env_config = ENVIRONMENTS[env_name]
        self.env = self.env_config["creator"]()
        self.env.reset()
        
        # Get environment info
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        
        self.obs_dims = [self.env.observation_space(agent).shape[0] for agent in self.agents]
        self.action_dims = [self.env.action_space(agent).shape[0] for agent in self.agents]
        
        print(f"\n{'='*70}")
        print(f"MADDPG Training Setup")
        print(f"{'='*70}")
        print(f"Environment: {env_name} ({self.env_config['type']})")
        print(f"Description: {self.env_config['description']}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Observation dimensions: {self.obs_dims}")
        print(f"Action dimensions: {self.action_dims}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Initialize MADDPG
        self._init_maddpg()
        
        # Initialize replay buffer
        self._init_buffer()
        
        # Initialize tracking
        self.episode_rewards = []
        self.best_reward = float('-inf')
        
        # Create directories
        self.run_dir = f"runs/{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(f"{self.run_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.run_dir}/plots", exist_ok=True)
    
    def _init_maddpg(self):
        """Initialize MADDPG algorithm"""
        
        net_config = {
            "arch": "mlp",
            "hidden_size": self.config["hidden_sizes"],
        }
        
        self.maddpg = MADDPG(
            state_dims=self.obs_dims,
            action_dims=self.action_dims,
            one_hot=False,
            n_agents=self.n_agents,
            agent_ids=self.agents,
            max_action=[[1.0] * ad for ad in self.action_dims],
            min_action=[[-1.0] * ad for ad in self.action_dims],
            discrete_actions=False,
            expl_noise=self.config["expl_noise"],
            lr_actor=self.config["lr_actor"],
            lr_critic=self.config["lr_critic"],
            gamma=self.config["gamma"],
            tau=self.config["tau"],
            net_config=net_config,
            device=self.device,
        )
    
    def _init_buffer(self):
        """
        Initialize Multi-Agent Replay Buffer
        
        The buffer stores JOINT experiences from all agents:
        - States: {agent_0: obs_0, agent_1: obs_1, ...}
        - Actions: {agent_0: act_0, agent_1: act_1, ...}
        - Rewards: {agent_0: r_0, agent_1: r_1, ...}
        - Next states and dones similarly
        
        Random sampling breaks temporal correlation!
        """
        self.memory = MultiAgentReplayBuffer(
            memory_size=self.config["buffer_size"],
            field_names=["state", "action", "reward", "next_state", "done"],
            agent_ids=self.agents,
            device=self.device,
        )
    
    def train(self):
        """Main training loop"""
        
        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="marl-research",
                name=f"maddpg-{self.env_name}",
                config=self.config
            )
        
        pbar = trange(self.config["total_episodes"], desc="Training")
        
        for episode in pbar:
            episode_reward = self._run_episode(training=True)
            self.episode_rewards.append(episode_reward)
            
            # Logging
            if episode % self.config["log_interval"] == 0:
                self._log_progress(episode)
            
            # Save checkpoint
            if episode % self.config["save_interval"] == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Update progress bar
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}"})
        
        # Final save
        self._save_checkpoint("final")
        self._plot_training_curve()
        
        if self.use_wandb:
            wandb.finish()
        
        return self.maddpg, self.episode_rewards
    
    def _run_episode(self, training=True):
        """
        Run a single episode
        
        Demonstrates CTDE:
        - Actions selected using only local observations (decentralized)
        - Learning uses all observations and actions (centralized)
        """
        observations, _ = self.env.reset()
        episode_reward = 0
        
        for step in range(self.config["max_steps"]):
            # ─────────────────────────────────────────────────────────────────
            # DECENTRALIZED ACTION SELECTION
            # Each agent's policy only sees its own observation
            # ─────────────────────────────────────────────────────────────────
            obs_list = [observations[agent] for agent in self.agents]
            actions_list = self.maddpg.get_action(obs_list, training=training)
            actions = {agent: actions_list[i] for i, agent in enumerate(self.agents)}
            
            # Environment step
            next_observations, rewards, terminations, truncations, _ = self.env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in self.agents}
            
            # ─────────────────────────────────────────────────────────────────
            # STORE JOINT EXPERIENCE IN REPLAY BUFFER
            # ─────────────────────────────────────────────────────────────────
            if training:
                self.memory.save_to_memory(
                    state={agent: observations[agent] for agent in self.agents},
                    action={agent: actions[agent] for agent in self.agents},
                    reward={agent: rewards[agent] for agent in self.agents},
                    next_state={agent: next_observations[agent] for agent in self.agents},
                    done={agent: dones[agent] for agent in self.agents},
                )
            
            episode_reward += sum(rewards.values())
            observations = next_observations
            
            # ─────────────────────────────────────────────────────────────────
            # CENTRALIZED TRAINING
            # Critic sees ALL observations and actions
            # Random sampling from buffer breaks temporal correlation
            # ─────────────────────────────────────────────────────────────────
            if training and len(self.memory) >= self.config["learn_start"]:
                experiences = self.memory.sample(self.config["batch_size"])
                self.maddpg.learn(experiences)
        
        return episode_reward
    
    def _log_progress(self, episode):
        """Log training progress"""
        avg_reward = np.mean(self.episode_rewards[-100:])
        
        print(f"\nEpisode {episode}")
        print(f"  Avg Reward (100 ep): {avg_reward:.2f}")
        print(f"  Buffer Size: {len(self.memory)}")
        
        if self.use_wandb:
            wandb.log({
                "episode": episode,
                "episode_reward": self.episode_rewards[-1],
                "avg_reward_100": avg_reward,
                "buffer_size": len(self.memory),
            })
        
        # Track best model
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self._save_checkpoint("best")
    
    def _save_checkpoint(self, tag):
        """Save model checkpoint"""
        path = f"{self.run_dir}/checkpoints/{tag}"
        os.makedirs(path, exist_ok=True)
        self.maddpg.save_checkpoint(path)
        print(f"Saved checkpoint: {path}")
    
    def _plot_training_curve(self):
        """Plot and save training curve"""
        plt.figure(figsize=(12, 5))
        
        # Raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, alpha=0.3, label="Episode Reward")
        
        # Moving average
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(
                self.episode_rewards, 
                np.ones(window)/window, 
                mode='valid'
            )
            plt.plot(range(window-1, len(self.episode_rewards)), 
                    moving_avg, label=f"Moving Avg ({window})")
        
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"MADDPG Training: {self.env_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reward distribution
        plt.subplot(1, 2, 2)
        plt.hist(self.episode_rewards[-500:], bins=30, edgecolor='black')
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution (Last 500 Episodes)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/plots/training_curve.png", dpi=150)
        plt.close()
        
        print(f"Saved training curve to {self.run_dir}/plots/training_curve.png")
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate trained agents
        
        DECENTRALIZED EXECUTION:
        - Each agent only uses its own observation
        - No communication or centralized coordination
        - Critics are not used!
        """
        print(f"\n{'='*70}")
        print("EVALUATION (Decentralized Execution)")
        print(f"{'='*70}")
        
        if render:
            eval_env = self.env_config["creator"]()
            # Note: For rendering, you may need render_mode="human"
        else:
            eval_env = self.env_config["creator"]()
        
        total_rewards = []
        
        for ep in range(num_episodes):
            reward = self._run_episode(training=False)
            total_rewards.append(reward)
            print(f"  Episode {ep + 1}: Reward = {reward:.2f}")
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"\nResults: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"{'='*70}\n")
        
        return total_rewards


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MADDPG Training for MARL Research")
    
    parser.add_argument("--env", type=str, default="simple_spread",
                        choices=list(ENVIRONMENTS.keys()),
                        help="Environment to train on")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr_actor", type=float, default=1e-4,
                        help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3,
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--eval_only", type=str, default=None,
                        help="Path to checkpoint for evaluation only")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "total_episodes": args.episodes,
        "batch_size": args.batch_size,
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
        "gamma": args.gamma,
    }
    
    # Create trainer
    trainer = MADDPGTrainer(
        env_name=args.env,
        config=config,
        use_wandb=args.wandb
    )
    
    if args.eval_only:
        # Load and evaluate
        trainer.maddpg.load_checkpoint(args.eval_only)
        trainer.evaluate(num_episodes=20)
    else:
        # Train
        maddpg, rewards = trainer.train()
        
        # Evaluate
        trainer.evaluate(num_episodes=10)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
