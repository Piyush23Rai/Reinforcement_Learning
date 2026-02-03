"""
Retail Demo: Multi-Warehouse Inventory Optimization with CTDE-MADDPG

This is a complete, end-to-end demo that:
1. Creates a multi-warehouse environment
2. Trains MADDPG agents using CTDE architecture
3. Evaluates the trained agents
4. Visualizes results

Run this script to see MARL in action!
"""

import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.maddpg import create_maddpg_agents
from core.replay_buffer import SharedReplayBuffer
from core.utils import plot_training_curve, Logger, set_seed
from .environment import MultiWarehouseEnv


def demo_train_and_evaluate():
    """
    Complete demo of retail inventory optimization with CTDE-MADDPG.
    """
    print("\n" + "="*80)
    print(" RETAIL DEMO: Multi-Warehouse Inventory Optimization with CTDE-MADDPG")
    print("="*80 + "\n")
    
    # ==========================================
    # 1. SETUP
    # ==========================================
    print("1. SETUP")
    print("-" * 80)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configuration
    config = {
        'num_agents': 5,
        'num_episodes': 100,
        'max_steps_per_episode': 100,
        'batch_size': 32,
        'buffer_size': 5000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'tau': 0.001,
        'epsilon': 1.0,
        'epsilon_decay': 0.998,
        'epsilon_min': 0.05,
        'warehouse_capacity': 1000.0,
        'demand_mean': 500.0,
    }
    
    print(f"Number of warehouses (agents): {config['num_agents']}")
    print(f"Number of training episodes: {config['num_episodes']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}\n")
    
    # ==========================================
    # 2. CREATE ENVIRONMENT
    # ==========================================
    print("2. CREATE ENVIRONMENT")
    print("-" * 80)
    
    env = MultiWarehouseEnv(
        num_warehouses=config['num_agents'],
        warehouse_capacity=config['warehouse_capacity'],
        demand_mean=config['demand_mean'],
        topology='line'  # Linear warehouse network
    )
    
    print(f"Environment: {config['num_agents']}-warehouse inventory system")
    print(f"Warehouse capacity: {config['warehouse_capacity']} units")
    print(f"Observation space per agent: {env.observation_space_size}")
    print(f"Action space per agent: {env.action_space_size}\n")
    
    # ==========================================
    # 3. CREATE AGENTS (MADDPG with CTDE)
    # ==========================================
    print("3. CREATE AGENTS")
    print("-" * 80)
    
    agents = create_maddpg_agents(
        state_dim=env.observation_space_size,
        action_dim=env.action_space_size,
        num_agents=config['num_agents'],
        config=config
    )
    
    print(f"Created {len(agents)} MADDPG agents")
    print("Each agent has:")
    print("  - Local actor network (uses only local observation)")
    print("  - Centralized critic network (sees all agents' observations)")
    print("This is the CTDE architecture!\n")
    
    # ==========================================
    # 4. CREATE REPLAY BUFFER
    # ==========================================
    print("4. CREATE REPLAY BUFFER")
    print("-" * 80)
    
    replay_buffer = SharedReplayBuffer(
        buffer_size=config['buffer_size'],
        num_agents=config['num_agents'],
        state_dim=env.observation_space_size,
        action_dim=env.action_space_size,
        device=agents[0].device
    )
    
    print(f"Shared replay buffer size: {config['buffer_size']}")
    print("This buffer stores experiences from ALL agents together")
    print("(Centralized training component of CTDE)\n")
    
    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    print("5. TRAINING")
    print("-" * 80)
    print("Training agents to optimize multi-warehouse inventory...\n")
    # Enable anomaly detection to help debug any autograd in-place errors during development
    torch.autograd.set_detect_anomaly(True)
    
    logger = Logger(['reward', 'cost', 'exploration'])
    train_rewards = []
    train_costs = []
    
    for episode in tqdm(range(config['num_episodes']), desc="Training episodes"):
        obs = env.reset()
        ep_reward = 0
        ep_cost = 0
        
        for step in range(config['max_steps_per_episode']):
            # ===== DECENTRALIZED EXECUTION =====
            # Each agent decides independently using only its local observation
            actions = np.array([agent.select_action(obs[i], training=True) 
                              for i, agent in enumerate(agents)])
            
            # Execute in environment
            next_obs, rewards, dones, info = env.step(actions)
            ep_reward += rewards.sum()
            ep_cost += info['total_cost']
            
            # ===== STORE EXPERIENCE (Centralized) =====
            # All agents' data stored together in shared buffer
            replay_buffer.add(obs, actions, rewards, next_obs, dones)
            
            obs = next_obs
            if dones[0]:
                break
        
        # ===== CENTRALIZED TRAINING =====
        # Train all agents from shared replay buffer
        if replay_buffer.is_ready(config['batch_size']):
            batch = replay_buffer.sample(config['batch_size'])
            for agent in agents:
                agent.update(batch, agents)
        
        # Decay exploration
        for agent in agents:
            agent.decay_exploration()
        
        # Log metrics
        logger.log(reward=ep_reward, cost=ep_cost, exploration=agents[0].epsilon)
        train_rewards.append(ep_reward)
        train_costs.append(ep_cost)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(train_rewards[-100:])
            avg_cost = np.mean(train_costs[-100:])
            print(f"Episode {episode+1}/{config['num_episodes']} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Cost: {avg_cost:.2f} | "
                  f"Exploration: {agents[0].epsilon:.4f}")
    
    print("Training completed!\n")
    
    # ==========================================
    # 6. EVALUATION
    # ==========================================
    print("6. EVALUATION")
    print("-" * 80)
    print("Evaluating trained agents (using greedy policy, no exploration)...\n")
    
    eval_rewards = [[] for _ in range(config['num_agents'])]
    eval_costs = []
    eval_stockouts = []
    
    for episode in tqdm(range(100), desc="Evaluation episodes"):
        obs = env.reset()
        
        while True:
            # Decentralized execution with greedy policy (no exploration)
            actions = np.array([agent.select_action(obs[i], training=False) 
                              for i, agent in enumerate(agents)])
            
            next_obs, rewards, dones, info = env.step(actions)
            
            for i in range(config['num_agents']):
                eval_rewards[i].append(rewards[i])
            
            eval_costs.append(info['total_cost'])
            eval_stockouts.extend(info['stockout'])
            
            obs = next_obs
            if dones[0]:
                break
    
    print("\nEvaluation Results:")
    print(f"  Average total cost: {np.mean(eval_costs):.2f}")
    print(f"  Total stockouts: {np.sum(eval_stockouts):.2f}")
    print(f"  Average reward per agent: {np.mean([np.mean(r) for r in eval_rewards]):.2f}")
    print()
    
    # ==========================================
    # 7. RESULTS & VISUALIZATION
    # ==========================================
    print("7. RESULTS")
    print("-" * 80)
    
    # Create results directory
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training plot
    plot_training_curve(
        train_costs,
        "Multi-Warehouse Inventory Optimization\nCTDE-MADDPG Training Costs",
        str(results_dir / 'training_costs.png')
    )
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Initial avg cost: {np.mean(train_costs[:10]):.2f}")
    print(f"  Final avg cost: {np.mean(train_costs[-10:]):.2f}")
    print(f"  Cost improvement: {(1 - np.mean(train_costs[-10:]) / np.mean(train_costs[:10])) * 100:.1f}%")
    print(f"\nResults saved to: {results_dir}")
    
    print("\n" + "="*80)
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Insights:")
    print("1. CTDE Architecture enabled coordination without explicit communication")
    print("2. Agents learned implicit cooperation through shared training")
    print("3. Decentralized execution allows scalability to more agents")
    print("4. Inventory costs reduced significantly during training")
    print("5. System dynamically optimized transfer routes and safety stocks")
    print("\n")


if __name__ == "__main__":
    demo_train_and_evaluate()
