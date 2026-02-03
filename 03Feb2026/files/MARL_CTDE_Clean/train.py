"""
Simple Training Script - WORKING VERSION
No infinite loops, straightforward training loop
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from maddpg import MADDPGAgent, ReplayBuffer
from warehouse_env import WarehouseEnv


def train():
    """Main training function"""
    
    print("\n" + "="*70)
    print("MARL CTDE - Multi-Warehouse Inventory Optimization")
    print("="*70 + "\n")
    
    # Configuration
    num_agents = 3
    num_episodes = 200
    batch_size = 32
    update_frequency = 4
    
    print(f"Configuration:")
    print(f"  Agents: {num_agents}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Environment
    env = WarehouseEnv(num_warehouses=num_agents, max_steps=50)
    
    # Agents
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agents = [MADDPGAgent(i, state_dim=2, action_dim=1, 
                         num_agents=num_agents, device=device, lr=0.001)
              for i in range(num_agents)]
    
    # Replay buffer
    buffer = ReplayBuffer(max_size=5000)
    
    # Training loop
    episode_rewards = []
    episode_costs = []
    
    print("Starting training...")
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        ep_reward = 0
        ep_cost = 0
        
        # Episode loop
        for step in range(env.max_steps):
            # Select actions (decentralized)
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(obs[i], training=True)
                actions.append(action)
            
            actions = np.array(actions)
            
            # Environment step
            next_obs, rewards, dones, _ = env.step(actions)
            ep_reward += rewards.sum()
            ep_cost += (-rewards).sum()  # Convert rewards back to costs
            
            # Store in buffer
            buffer.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            
            if dones[0]:
                break
        
        # Update from buffer (after episode)
        if buffer.is_ready(batch_size):
            batch = buffer.sample(batch_size)
            for agent in agents:
                agent.update(batch, agents)
        
        # Decay exploration
        for agent in agents:
            agent.epsilon = max(0.01, agent.epsilon * 0.99)
        
        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_cost = np.mean(episode_costs[-50:])
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  Avg Reward (last 50): {avg_reward:.2f}")
            print(f"  Avg Cost (last 50): {avg_cost:.2f}")
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70 + "\n")
    
    # Results
    print("RESULTS:")
    print(f"  Initial avg cost: {np.mean(episode_costs[:10]):.2f}")
    print(f"  Final avg cost: {np.mean(episode_costs[-10:]):.2f}")
    if np.mean(episode_costs[:10]) > 0:
        improvement = (1 - np.mean(episode_costs[-10:]) / np.mean(episode_costs[:10])) * 100
        print(f"  Improvement: {improvement:.1f}%")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_costs, alpha=0.3, label='Episode Cost')
    
    # Moving average
    window = 20
    moving_avg = np.convolve(episode_costs, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_costs)), moving_avg, label=f'Moving Average (window={window})', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.title('Multi-Warehouse Inventory Optimization Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=100)
    print("\nPlot saved to: training_results.png")
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation (greedy policy, no exploration)")
    print("="*70 + "\n")
    
    eval_costs = []
    for _ in range(100):
        obs = env.reset()
        ep_cost = 0
        
        for step in range(env.max_steps):
            # Greedy actions (no noise)
            actions = np.array([agent.select_action(obs[i], training=False) 
                              for i, agent in enumerate(agents)])
            next_obs, rewards, dones, _ = env.step(actions)
            ep_cost += (-rewards).sum()
            obs = next_obs
            
            if dones[0]:
                break
        
        eval_costs.append(ep_cost)
    
    print(f"Evaluation Results:")
    print(f"  Average cost: {np.mean(eval_costs):.2f}")
    print(f"  Std dev: {np.std(eval_costs):.2f}")
    print(f"  Min: {np.min(eval_costs):.2f}")
    print(f"  Max: {np.max(eval_costs):.2f}")
    print()


if __name__ == "__main__":
    train()
