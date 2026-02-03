# Multi-Agent Reinforcement Learning: Production Deployment Guide

## Complete Open Source Research Setup for CTDE, MADDPG, QMIX

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Environment Setup](#2-environment-setup)
3. [Option A: Quick Start with PettingZoo + MADDPG](#3-option-a-quick-start-with-pettingzoo--maddpg)
4. [Option B: QMIX with PyMARL + SMAC](#4-option-b-qmix-with-pymarl--smac)
5. [Option C: Production Scale with RLlib](#5-option-c-production-scale-with-rllib)
6. [Option D: TorchRL for Research](#6-option-d-torchrl-for-research)
7. [Experiment Tracking & Visualization](#7-experiment-tracking--visualization)
8. [Model Export & Deployment](#8-model-export--deployment)
9. [Complete Example Projects](#9-complete-example-projects)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview & Architecture

### Recommended Stack for Research

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION MARL RESEARCH STACK                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENVIRONMENTS                    ALGORITHMS           INFRASTRUCTURE        │
│  ─────────────                   ──────────           ──────────────        │
│  • PettingZoo (MPE)              • MADDPG             • Ray/RLlib           │
│  • SMAC (StarCraft)              • QMIX               • Weights & Biases    │
│  • VMAS (Vectorized)             • MAPPO              • TensorBoard         │
│  • MeltingPot                    • VDN, QTRAN         • Docker              │
│                                  • IQL                • Git + DVC           │
│                                                                             │
│  DEEP LEARNING                   UTILITIES                                  │
│  ─────────────                   ─────────                                  │
│  • PyTorch                       • Gymnasium                                │
│  • TorchRL                       • SuperSuit                                │
│  • AgileRL                       • NumPy, Pandas                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Choose Your Path

| Use Case | Recommended Setup | Complexity |
|----------|-------------------|------------|
| **Learning/Teaching** | PettingZoo + AgileRL MADDPG | ⭐ Easy |
| **QMIX Research** | PyMARL + SMAC | ⭐⭐ Medium |
| **Production Scale** | Ray RLlib + PettingZoo | ⭐⭐⭐ Advanced |
| **Cutting-Edge Research** | TorchRL + VMAS | ⭐⭐⭐ Advanced |

---

## 2. Environment Setup

### 2.1 Create Conda Environment

```bash
# Create dedicated environment
conda create -n marl python=3.10 -y
conda activate marl

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn tqdm

# MARL specific
pip install pettingzoo[all]
pip install supersuit
pip install gymnasium

# Experiment tracking
pip install wandb tensorboard

# Optional: For SMAC
pip install pysc2
```

### 2.2 Verify Installation

```python
# test_installation.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

import pettingzoo
print(f"PettingZoo: {pettingzoo.__version__}")

# Test MPE environment
from pettingzoo.mpe import simple_spread_v3
env = simple_spread_v3.parallel_env(render_mode="rgb_array")
env.reset()
print("✓ PettingZoo MPE working!")

# Test action/observation spaces
for agent in env.agents:
    print(f"  Agent: {agent}")
    print(f"    Obs space: {env.observation_space(agent)}")
    print(f"    Act space: {env.action_space(agent)}")
```

---

## 3. Option A: Quick Start with PettingZoo + MADDPG

### Best for: Learning, teaching, quick experiments

### 3.1 Install AgileRL

```bash
pip install agilerl
```

### 3.2 Complete MADDPG Training Script

```python
# maddpg_simple_spread.py
"""
Complete MADDPG training on Simple Spread environment
Demonstrates: CTDE, Centralized Critics, Decentralized Actors, Replay Buffer
"""

import os
import numpy as np
import torch
from tqdm import trange

# AgileRL imports
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

# PettingZoo environment
from pettingzoo.mpe import simple_spread_v3

# Experiment tracking
import wandb


def make_env():
    """Create and configure the environment"""
    env = simple_spread_v3.parallel_env(
        N=3,                    # Number of agents
        local_ratio=0.5,        # Reward locality
        max_cycles=25,          # Episode length
        continuous_actions=True # Continuous for MADDPG
    )
    return env


def train_maddpg(
    total_episodes=5000,
    max_steps=25,
    batch_size=256,
    buffer_size=100000,
    lr_actor=1e-4,
    lr_critic=1e-3,
    gamma=0.95,
    tau=0.01,
    use_wandb=True
):
    """
    Train MADDPG agents on Simple Spread
    
    This demonstrates all key MARL concepts:
    - CTDE: Centralized training (critic sees all), decentralized execution
    - Replay Buffer: Stores joint experiences, random sampling
    - Multi-Agent: 3 cooperative agents learning together
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # SETUP
    # ═══════════════════════════════════════════════════════════════════════
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env()
    env.reset()
    
    # Get environment info
    agents = env.agents
    n_agents = len(agents)
    
    # Observation and action dimensions
    obs_dims = [env.observation_space(agent).shape[0] for agent in agents]
    action_dims = [env.action_space(agent).shape[0] for agent in agents]
    
    # State dimension (concatenated observations for centralized critic)
    state_dim = sum(obs_dims)
    
    print(f"\n{'='*60}")
    print(f"Environment: Simple Spread")
    print(f"Number of agents: {n_agents}")
    print(f"Observation dims: {obs_dims}")
    print(f"Action dims: {action_dims}")
    print(f"State dim (for critic): {state_dim}")
    print(f"{'='*60}\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZE MADDPG
    # ═══════════════════════════════════════════════════════════════════════
    
    # Network configuration
    net_config = {
        "arch": "mlp",
        "hidden_size": [64, 64],
    }
    
    # Create MADDPG agent
    maddpg = MADDPG(
        state_dims=obs_dims,           # Per-agent observation dims
        action_dims=action_dims,        # Per-agent action dims
        one_hot=False,
        n_agents=n_agents,
        agent_ids=agents,
        max_action=[[1.0] * ad for ad in action_dims],
        min_action=[[-1.0] * ad for ad in action_dims],
        discrete_actions=False,
        expl_noise=0.1,                # Exploration noise
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,                    # Discount factor
        tau=tau,                        # Soft update coefficient
        net_config=net_config,
        device=device,
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # REPLAY BUFFER (Joint Experience Storage)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Multi-agent replay buffer stores:
    # (observations_all, actions_all, rewards_all, next_observations_all, dones)
    
    memory = MultiAgentReplayBuffer(
        memory_size=buffer_size,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agents,
        device=device,
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT TRACKING
    # ═══════════════════════════════════════════════════════════════════════
    
    if use_wandb:
        wandb.init(
            project="marl-research",
            name="maddpg-simple-spread",
            config={
                "algorithm": "MADDPG",
                "environment": "simple_spread_v3",
                "n_agents": n_agents,
                "total_episodes": total_episodes,
                "batch_size": batch_size,
                "buffer_size": buffer_size,
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "gamma": gamma,
                "tau": tau,
            }
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════
    
    episode_rewards = []
    
    for episode in trange(total_episodes, desc="Training"):
        # Reset environment
        observations, _ = env.reset()
        
        episode_reward = 0
        
        for step in range(max_steps):
            # ─────────────────────────────────────────────────────────────────
            # ACTION SELECTION (Decentralized - each agent uses own observation)
            # ─────────────────────────────────────────────────────────────────
            
            # Convert observations to list format
            obs_list = [observations[agent] for agent in agents]
            
            # Get actions (with exploration noise during training)
            actions_list = maddpg.get_action(obs_list, training=True)
            
            # Convert to dict for environment
            actions = {agent: actions_list[i] for i, agent in enumerate(agents)}
            
            # ─────────────────────────────────────────────────────────────────
            # ENVIRONMENT STEP
            # ─────────────────────────────────────────────────────────────────
            
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if episode is done
            dones = {agent: terminations[agent] or truncations[agent] for agent in agents}
            
            # ─────────────────────────────────────────────────────────────────
            # STORE IN REPLAY BUFFER (Joint Experience)
            # ─────────────────────────────────────────────────────────────────
            
            # Store transition for all agents
            memory.save_to_memory(
                state={agent: observations[agent] for agent in agents},
                action={agent: actions[agent] for agent in agents},
                reward={agent: rewards[agent] for agent in agents},
                next_state={agent: next_observations[agent] for agent in agents},
                done={agent: dones[agent] for agent in agents},
            )
            
            # Track reward
            episode_reward += sum(rewards.values())
            
            # Update observations
            observations = next_observations
            
            # ─────────────────────────────────────────────────────────────────
            # TRAINING (Centralized - critic sees everything)
            # ─────────────────────────────────────────────────────────────────
            
            # Only train when buffer has enough samples
            if len(memory) >= batch_size:
                # Sample random batch (breaks temporal correlation)
                experiences = memory.sample(batch_size)
                
                # Learn from batch
                # Internally:
                #   - Critic update: Uses ALL agents' obs and actions
                #   - Actor update: Uses policy gradient through centralized critic
                losses = maddpg.learn(experiences)
        
        # Track episode stats
        episode_rewards.append(episode_reward)
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"\nEpisode {episode} | Avg Reward (100 ep): {avg_reward:.2f}")
            
            if use_wandb:
                wandb.log({
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "avg_reward_100": avg_reward,
                })
        
        # ─────────────────────────────────────────────────────────────────────
        # SAVE CHECKPOINTS
        # ─────────────────────────────────────────────────────────────────────
        
        if episode % 1000 == 0 and episode > 0:
            save_path = f"checkpoints/maddpg_episode_{episode}"
            os.makedirs(save_path, exist_ok=True)
            maddpg.save_checkpoint(save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE FINAL MODEL
    # ═══════════════════════════════════════════════════════════════════════
    
    final_path = "checkpoints/maddpg_final"
    os.makedirs(final_path, exist_ok=True)
    maddpg.save_checkpoint(final_path)
    print(f"\nFinal model saved to {final_path}")
    
    if use_wandb:
        wandb.finish()
    
    env.close()
    return maddpg, episode_rewards


def evaluate(maddpg, num_episodes=10, render=True):
    """Evaluate trained agents (Decentralized Execution)"""
    
    env = make_env()
    if render:
        env = simple_spread_v3.parallel_env(
            N=3, local_ratio=0.5, max_cycles=25,
            continuous_actions=True, render_mode="human"
        )
    
    agents = env.agents
    total_rewards = []
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        
        done = False
        while not done:
            # Decentralized execution - each agent only uses own observation
            obs_list = [observations[agent] for agent in agents]
            actions_list = maddpg.get_action(obs_list, training=False)
            actions = {agent: actions_list[i] for i, agent in enumerate(agents)}
            
            observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_reward += sum(rewards.values())
            
            done = all(terminations.values()) or all(truncations.values())
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nAverage Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    # Train
    maddpg, rewards = train_maddpg(
        total_episodes=3000,
        use_wandb=False  # Set True if you have wandb configured
    )
    
    # Plot training curve
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    
    # Moving average
    window = 100
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), moving_avg, label=f"Moving Avg ({window})")
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MADDPG Training on Simple Spread")
    plt.legend()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION (Decentralized Execution)")
    print("="*60)
    evaluate(maddpg, num_episodes=5, render=False)
```

### 3.3 Run Training

```bash
# Run with default settings
python maddpg_simple_spread.py

# With Weights & Biases tracking
wandb login  # First time only
python maddpg_simple_spread.py --use_wandb
```

---

## 4. Option B: QMIX with PyMARL + SMAC

### Best for: QMIX research, StarCraft experiments

### 4.1 Install StarCraft II and SMAC

```bash
# Clone PyMARL2 (optimized version)
git clone https://github.com/hijkzzz/pymarl2.git
cd pymarl2

# Create environment
conda create -n pymarl python=3.8 -y
conda activate pymarl

# Install dependencies
bash install_dependecies.sh

# This downloads StarCraft II and SMAC maps automatically
```

### 4.2 Alternative: Manual SMAC Installation

```bash
# For Linux
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip SC2.4.10.zip -d ~/StarCraftII/

# Set environment variable
export SC2PATH=~/StarCraftII/

# Install SMAC
pip install git+https://github.com/oxwhirl/smac.git
```

### 4.3 Run QMIX Training

```bash
# Train QMIX on 3 Marines vs 3 Marines
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m

# Train on harder scenarios
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3s5z

# With TensorBoard logging
python3 src/main.py --config=qmix --env-config=sc2 \
    with env_args.map_name=3m \
    use_tensorboard=True \
    save_model=True
```

### 4.4 SMAC Maps Reference

| Map | Agents | Difficulty | Description |
|-----|--------|------------|-------------|
| 3m | 3 | Easy | 3 Marines vs 3 Marines |
| 8m | 8 | Easy | 8 Marines vs 8 Marines |
| 2s3z | 5 | Easy | 2 Stalkers + 3 Zealots |
| 3s5z | 8 | Hard | 3 Stalkers + 5 Zealots |
| 1c3s5z | 9 | Hard | 1 Colossus + 3 Stalkers + 5 Zealots |
| 27m_vs_30m | 27 | Super Hard | Large scale battle |

### 4.5 View Results

```bash
# TensorBoard
tensorboard --logdir=results/

# Watch replay (requires Windows/Mac StarCraft II client)
# Replays saved in: ~/StarCraftII/Replays/
```

---

## 5. Option C: Production Scale with RLlib

### Best for: Distributed training, production deployment

### 5.1 Install Ray RLlib

```bash
pip install "ray[rllib]" torch
pip install pettingzoo[mpe]
```

### 5.2 Multi-Agent PPO with RLlib

```python
# rllib_mappo.py
"""
Production-ready Multi-Agent PPO using Ray RLlib
Supports distributed training across multiple GPUs/machines
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from pettingzoo.mpe import simple_spread_v3


def env_creator(config):
    """Environment factory for RLlib"""
    return simple_spread_v3.parallel_env(
        N=config.get("n_agents", 3),
        local_ratio=config.get("local_ratio", 0.5),
        max_cycles=config.get("max_cycles", 25),
        continuous_actions=True
    )


def train_mappo():
    """Train Multi-Agent PPO"""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env(
        "simple_spread",
        lambda config: ParallelPettingZooEnv(env_creator(config))
    )
    
    # Create a test env to get agent info
    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    # Configure PPO
    config = (
        PPOConfig()
        .environment(
            env="simple_spread",
            env_config={"n_agents": 3, "local_ratio": 0.5, "max_cycles": 25}
        )
        .framework("torch")
        .resources(
            num_gpus=1,  # Use GPU if available
            num_cpus_per_worker=1,
        )
        .rollouts(
            num_rollout_workers=4,  # Parallel environments
            rollout_fragment_length=128,
        )
        .training(
            train_batch_size=512,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            num_sgd_iter=10,
            sgd_minibatch_size=128,
        )
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {})
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy"
        )
    )
    
    # Train with Tune
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={
            "training_iteration": 500,
            "episode_reward_mean": -50,  # Stop when good enough
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        name="mappo_simple_spread",
        verbose=1,
    )
    
    # Get best checkpoint
    best_checkpoint = results.get_best_checkpoint(
        trial=results.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
        mode="max"
    )
    
    print(f"\nBest checkpoint: {best_checkpoint}")
    
    ray.shutdown()
    return best_checkpoint


if __name__ == "__main__":
    checkpoint = train_mappo()
```

### 5.3 Distributed Training

```bash
# Start Ray cluster head
ray start --head --port=6379

# Start workers on other machines
ray start --address='<head-ip>:6379'

# Run distributed training
python rllib_mappo.py
```

---

## 6. Option D: TorchRL for Research

### Best for: Custom research, PyTorch integration

### 6.1 Install TorchRL

```bash
pip install torchrl
pip install vmas  # Vectorized multi-agent simulator
pip install "pettingzoo[mpe]==1.24.3"
```

### 6.2 MADDPG with TorchRL

```python
# torchrl_maddpg.py
"""
Research-grade MADDPG implementation using TorchRL
Demonstrates: Centralized critic, parameter sharing, vectorized envs
"""

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.envs.libs.pettingzoo import PettingZooEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import DDPGLoss

from pettingzoo.mpe import simple_spread_v3


def create_env():
    """Create TorchRL-wrapped environment"""
    base_env = PettingZooEnv(
        task="simple_spread_v3",
        parallel=True,
        continuous_actions=True,
        device="cpu",
    )
    
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[("agents", "reward")], out_keys=[("agents", "episode_reward")])
    )
    return env


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create environment
    env = create_env()
    
    # Get specs
    n_agents = env.n_agents
    obs_spec = env.observation_spec
    action_spec = env.action_spec
    
    print(f"N Agents: {n_agents}")
    print(f"Observation spec: {obs_spec}")
    print(f"Action spec: {action_spec}")
    
    # Create actor network (decentralized)
    actor_net = MultiAgentMLP(
        n_agent_inputs=obs_spec["agents", "observation"].shape[-1],
        n_agent_outputs=action_spec["agents", "action"].shape[-1],
        n_agents=n_agents,
        centralised=False,  # Decentralized actors
        share_params=True,  # Parameter sharing
        depth=2,
        num_cells=64,
    )
    
    # Create critic network (centralized)
    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=n_agents,
        centralised=True,  # Centralized critic (sees all observations)
        share_params=True,
        depth=2,
        num_cells=64,
    )
    
    # Wrap in TensorDict modules
    policy = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    
    value = ValueOperator(
        critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_action_value")],
    )
    
    # Create replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=100000),
        batch_size=256,
    )
    
    # Data collector
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=1000,
        total_frames=100000,
        device=device,
    )
    
    # Training loop
    for i, data in enumerate(collector):
        # Add to replay buffer
        replay_buffer.extend(data)
        
        if len(replay_buffer) > 1000:
            # Sample and train
            batch = replay_buffer.sample()
            
            # Your training logic here
            # ...
        
        if i % 100 == 0:
            print(f"Collected {(i+1) * 1000} frames")
    
    return policy


if __name__ == "__main__":
    policy = train()
```

---

## 7. Experiment Tracking & Visualization

### 7.1 Weights & Biases Setup

```bash
pip install wandb
wandb login
```

```python
# Add to your training script
import wandb

wandb.init(
    project="marl-research",
    name="experiment-name",
    config={
        "algorithm": "MADDPG",
        "n_agents": 3,
        "learning_rate": 1e-4,
        # ... other hyperparameters
    }
)

# Log metrics during training
wandb.log({
    "episode": episode,
    "reward": reward,
    "actor_loss": actor_loss,
    "critic_loss": critic_loss,
})

# Log model
wandb.save("model.pt")
wandb.finish()
```

### 7.2 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/maddpg_experiment")

# During training
writer.add_scalar("Reward/episode", reward, episode)
writer.add_scalar("Loss/actor", actor_loss, step)
writer.add_scalar("Loss/critic", critic_loss, step)

# View with: tensorboard --logdir=runs/
```

---

## 8. Model Export & Deployment

### 8.1 Save Trained Policies

```python
# Save PyTorch model
def save_model(policy, path):
    """Save only the policy (actors) for deployment"""
    torch.save({
        'actor_state_dict': policy.actor.state_dict(),
        'config': policy.config,
    }, path)

def load_model(path, device='cpu'):
    """Load policy for inference"""
    checkpoint = torch.load(path, map_location=device)
    # Reconstruct policy and load weights
    # ...
    return policy
```

### 8.2 ONNX Export for Production

```python
import torch.onnx

def export_to_onnx(policy, obs_dim, path):
    """Export policy to ONNX for cross-platform deployment"""
    dummy_input = torch.randn(1, obs_dim)
    
    torch.onnx.export(
        policy.actor,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print(f"Exported to {path}")

# Usage
export_to_onnx(maddpg.actors[0], obs_dim=18, path="agent_0.onnx")
```

### 8.3 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY inference.py .

CMD ["python", "inference.py"]
```

---

## 9. Complete Example Projects

### Project 1: Multi-Agent Pricing (MADDPG)

```python
# pricing_env.py
"""Custom multi-agent pricing environment"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class PricingEnv(ParallelEnv):
    """
    3 competing stores set prices for identical products.
    Demonstrates: Competitive MARL, CTDE, Economic modeling
    """
    
    metadata = {"render_modes": ["human"], "name": "pricing_v0"}
    
    def __init__(self, n_stores=3, max_price=15, min_price=5, cost=4):
        self.n_stores = n_stores
        self.max_price = max_price
        self.min_price = min_price
        self.cost = cost
        
        self.possible_agents = [f"store_{i}" for i in range(n_stores)]
        self.agents = self.possible_agents[:]
        
        # Observation: own inventory, local demand, competitor prices
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=100, shape=(n_stores + 2,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Action: price to set (continuous)
        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([min_price]),
                high=np.array([max_price]),
                dtype=np.float32
            )
            for agent in self.agents
        }
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        # Random initial inventories and demands
        self.inventories = np.random.uniform(30, 70, self.n_stores)
        self.demands = np.random.uniform(0.5, 1.0, self.n_stores)
        self.prices = np.ones(self.n_stores) * 10  # Initial prices
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        # Update prices from actions
        for i, agent in enumerate(self.agents):
            self.prices[i] = np.clip(actions[agent][0], self.min_price, self.max_price)
        
        # Calculate demands based on all prices
        rewards = {}
        for i, agent in enumerate(self.agents):
            # Demand increases if price is lower than competitors
            base_demand = 100 * self.demands[i]
            price_effect = -3 * self.prices[i]
            competitor_effect = 1.5 * np.mean(np.delete(self.prices, i))
            
            demand = max(0, base_demand + price_effect + competitor_effect)
            sold = min(demand, self.inventories[i])
            
            # Profit = (price - cost) * units_sold
            profit = (self.prices[i] - self.cost) * sold
            rewards[agent] = profit / 100  # Normalize
            
            # Update inventory
            self.inventories[i] -= sold
            self.inventories[i] += np.random.uniform(10, 30)  # Restock
        
        self.timestep += 1
        
        # Get new observations
        observations = self._get_observations()
        
        # Episode ends after 50 timesteps
        terminated = {agent: self.timestep >= 50 for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {"price": self.prices[i]} for i, agent in enumerate(self.agents)}
        
        return observations, rewards, terminated, truncated, infos
    
    def _get_observations(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            obs = np.concatenate([
                [self.inventories[i] / 100],  # Own inventory (normalized)
                [self.demands[i]],             # Local demand
                self.prices / self.max_price   # All prices (normalized)
            ])
            observations[agent] = obs.astype(np.float32)
        return observations
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]


# Usage
if __name__ == "__main__":
    env = PricingEnv()
    obs, _ = env.reset()
    
    for step in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, term, trunc, info = env.step(actions)
        
        if step % 10 == 0:
            print(f"Step {step}: Prices={[info[a]['price'] for a in env.agents]}, "
                  f"Rewards={[rewards[a] for a in env.agents]}")
```

---

## 10. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size, use gradient accumulation |
| Training doesn't converge | Lower learning rate, increase exploration noise |
| Agents don't coordinate | Check reward shaping, try parameter sharing |
| SMAC installation fails | Check SC2PATH env variable, use Docker |
| RLlib policy mapping errors | Verify agent IDs match environment |

### Debug Commands

```python
# Check environment setup
env = make_env()
env.reset()
print(f"Agents: {env.agents}")
print(f"Obs spaces: {[env.observation_space(a) for a in env.agents]}")
print(f"Act spaces: {[env.action_space(a) for a in env.agents]}")

# Check replay buffer
print(f"Buffer size: {len(memory)}")
sample = memory.sample(1)
print(f"Sample keys: {sample.keys()}")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

---

## Quick Start Commands Summary

```bash
# Option A: PettingZoo + AgileRL (Easiest)
pip install agilerl pettingzoo[mpe] wandb
python maddpg_simple_spread.py

# Option B: PyMARL + SMAC (QMIX)
git clone https://github.com/hijkzzz/pymarl2.git
cd pymarl2 && bash install_dependecies.sh
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m

# Option C: RLlib (Production)
pip install "ray[rllib]" torch pettingzoo[mpe]
python rllib_mappo.py

# Option D: TorchRL (Research)
pip install torchrl vmas pettingzoo[mpe]
python torchrl_maddpg.py
```

---

## Resources

- **MARL Book (FREE)**: https://www.marl-book.com/
- **PettingZoo Docs**: https://pettingzoo.farama.org/
- **PyMARL GitHub**: https://github.com/oxwhirl/pymarl
- **RLlib Docs**: https://docs.ray.io/en/latest/rllib/
- **TorchRL MARL**: https://pytorch.org/rl/tutorials/multiagent_competitive_ddpg.html

---

*Happy researching! 🚀*
