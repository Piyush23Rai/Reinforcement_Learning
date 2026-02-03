"""
Banking Demo - Transaction Routing with MADDPG
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from maddpg import MADDPGAgent, ReplayBuffer
from transaction_env import TransactionEnv


def train_banking():
    """Train transaction routing agents"""
    
    print("\n" + "="*70)
    print("MARL CTDE - Transaction Routing Optimization")
    print("="*70 + "\n")
    
    # Config
    num_agents = 3
    num_episodes = 200
    batch_size = 32
    
    print(f"Configuration:")
    print(f"  Agents: {num_agents}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Environment
    env = TransactionEnv(num_agents=num_agents, max_steps=100)
    
    # Agents
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agents = [MADDPGAgent(i, state_dim=2, action_dim=1,
                         num_agents=num_agents, device=device, lr=0.001)
              for i in range(num_agents)]
    
    # Buffer
    buffer = ReplayBuffer(max_size=5000)
    
    # Training
    episode_rewards = []
    episode_risks = []
    
    print("Starting training...")
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        ep_reward = 0
        max_risk = 0
        
        for step in range(env.max_steps):
            # Select actions
            actions = []
            for i, agent in enumerate(agents):
                # Convert to discrete action by rounding
                action = agent.select_action(obs[i], training=True)
                # Round to nearest integer (0, 1, or 2)
                action = np.round(action * 2).astype(int)
                actions.append(action)
            
            actions = np.array(actions)
            
            # Environment step
            next_obs, rewards, dones, info = env.step(actions)
            ep_reward += rewards.sum()
            max_risk = max(max_risk, info['total_risk'])
            
            # Store
            buffer.add(obs, actions.reshape(-1, 1), rewards, next_obs, dones)
            obs = next_obs
            
            if dones[0]:
                break
        
        # Update
        if buffer.is_ready(batch_size):
            batch = buffer.sample(batch_size)
            for agent in agents:
                agent.update(batch, agents)
        
        # Decay
        for agent in agents:
            agent.epsilon = max(0.01, agent.epsilon * 0.99)
        
        episode_rewards.append(ep_reward)
        episode_risks.append(max_risk)
        
        if (episode + 1) % 50 == 0:
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {np.mean(episode_rewards[-50:]):.2f}")
            print(f"  Max Risk: {max(episode_risks[-50:]):.4f}")
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70 + "\n")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    window = 20
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(episode_rewards)), moving_avg, label='Moving Average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Transaction Routing - Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Risk
    ax2.plot(episode_risks, alpha=0.3, label='Max Risk')
    ax2.axhline(y=0.15, color='r', linestyle='--', label='Risk Threshold')
    moving_avg_risk = np.convolve(episode_risks, np.ones(window)/window, mode='valid')
    ax2.plot(range(window-1, len(episode_risks)), moving_avg_risk, label='Moving Average', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Risk')
    ax2.set_title('Portfolio Risk During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('banking_results.png', dpi=100)
    print("Plot saved to: banking_results.png")
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation")
    print("="*70 + "\n")
    
    eval_rewards = []
    eval_risks = []
    
    for _ in range(50):
        obs = env.reset()
        ep_reward = 0
        max_risk = 0
        
        for step in range(env.max_steps):
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(obs[i], training=False)
                action = np.round(action * 2).astype(int)
                actions.append(action)
            
            actions = np.array(actions)
            next_obs, rewards, dones, info = env.step(actions)
            ep_reward += rewards.sum()
            max_risk = max(max_risk, info['total_risk'])
            obs = next_obs
            
            if dones[0]:
                break
        
        eval_rewards.append(ep_reward)
        eval_risks.append(max_risk)
    
    print(f"Evaluation Results:")
    print(f"  Avg Reward: {np.mean(eval_rewards):.2f}")
    print(f"  Max Risk: {max(eval_risks):.4f}")
    print(f"  Risk Violations: {sum(1 for r in eval_risks if r > 0.15)}/{len(eval_risks)}")
    print()


if __name__ == "__main__":
    train_banking()
