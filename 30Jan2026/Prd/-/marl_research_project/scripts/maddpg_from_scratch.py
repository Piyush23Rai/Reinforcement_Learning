#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPLETE MADDPG IMPLEMENTATION FROM SCRATCH
Demonstrates: CTDE, Centralized Critic, Decentralized Actor, Replay Buffer
═══════════════════════════════════════════════════════════════════════════════

This is a self-contained implementation showing ALL the concepts we discussed:
1. CTDE Paradigm - Centralized Training, Decentralized Execution
2. MADDPG Algorithm - Actor-Critic for continuous multi-agent
3. Replay Buffer - Joint experience storage with random sampling
4. Multi-Agent Coordination - 3 agents learning together

Run: python maddpg_from_scratch.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import trange

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """
    DECENTRALIZED ACTOR (Policy Network)
    
    Input: Local observation for ONE agent only
    Output: Continuous action
    
    Key: During execution, each agent only uses its OWN observation!
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # Tanh ensures actions in [-1, 1]
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    CENTRALIZED CRITIC (Q-Network)
    
    Input: ALL observations from ALL agents + ALL actions from ALL agents
    Output: Q-value for the specific agent
    
    Key: Sees EVERYTHING during training for stable learning!
    
    This is the core of CTDE - the critic has global view but is only
    used during training, not execution.
    """
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # Input is concatenation of ALL obs and ALL actions
        input_dim = total_obs_dim + total_action_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single Q-value output
        
    def forward(self, all_obs, all_actions):
        """
        Args:
            all_obs: Concatenated observations from ALL agents [batch, total_obs_dim]
            all_actions: Concatenated actions from ALL agents [batch, total_action_dim]
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER - Joint Experience Storage
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentReplayBuffer:
    """
    MULTI-AGENT REPLAY BUFFER
    
    Stores JOINT experiences from all agents together:
    - All observations at time t
    - All actions taken at time t  
    - All rewards received
    - All next observations at time t+1
    - Done flag
    
    Key insight: Random sampling breaks temporal correlation!
    Without this, learning would be unstable due to correlated samples.
    """
    
    def __init__(self, capacity, n_agents, obs_dims, action_dims):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        
        # Use deque for efficient FIFO when at capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, observations, actions, rewards, next_observations, dones):
        """
        Store a joint experience tuple
        
        Args:
            observations: List of obs for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_observations: List of next obs for each agent
            dones: List of done flags for each agent
        """
        experience = (
            [np.array(o, dtype=np.float32) for o in observations],
            [np.array(a, dtype=np.float32) for a in actions],
            [np.array([r], dtype=np.float32) for r in rewards],
            [np.array(o, dtype=np.float32) for o in next_observations],
            [np.array([d], dtype=np.float32) for d in dones]
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        RANDOM SAMPLING - This is crucial!
        
        Random sampling ensures:
        1. IID data assumption for SGD
        2. Breaks temporal correlation
        3. Diverse training batches
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Organize by agent
        obs_batch = [[] for _ in range(self.n_agents)]
        action_batch = [[] for _ in range(self.n_agents)]
        reward_batch = [[] for _ in range(self.n_agents)]
        next_obs_batch = [[] for _ in range(self.n_agents)]
        done_batch = [[] for _ in range(self.n_agents)]
        
        for experience in batch:
            obs, actions, rewards, next_obs, dones = experience
            for i in range(self.n_agents):
                obs_batch[i].append(obs[i])
                action_batch[i].append(actions[i])
                reward_batch[i].append(rewards[i])
                next_obs_batch[i].append(next_obs[i])
                done_batch[i].append(dones[i])
        
        # Convert to tensors
        obs_batch = [torch.FloatTensor(np.array(obs_batch[i])) for i in range(self.n_agents)]
        action_batch = [torch.FloatTensor(np.array(action_batch[i])) for i in range(self.n_agents)]
        reward_batch = [torch.FloatTensor(np.array(reward_batch[i])) for i in range(self.n_agents)]
        next_obs_batch = [torch.FloatTensor(np.array(next_obs_batch[i])) for i in range(self.n_agents)]
        done_batch = [torch.FloatTensor(np.array(done_batch[i])) for i in range(self.n_agents)]
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# MADDPG AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class MADDPGAgent:
    """
    MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
    
    Implements CTDE:
    - Each agent has its own Actor (decentralized policy)
    - Each agent has its own Critic (centralized Q-function)
    - Training: Critic sees all obs/actions (centralized)
    - Execution: Actor only uses local obs (decentralized)
    """
    
    def __init__(self, n_agents, obs_dims, action_dims, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01):
        
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update coefficient
        
        # Total dimensions for centralized critic
        self.total_obs_dim = sum(obs_dims)
        self.total_action_dim = sum(action_dims)
        
        # Create networks for each agent
        self.actors = []
        self.critics = []
        self.actors_target = []
        self.critics_target = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(n_agents):
            # Actor: Only takes local observation
            actor = Actor(obs_dims[i], action_dims[i])
            actor_target = Actor(obs_dims[i], action_dims[i])
            actor_target.load_state_dict(actor.state_dict())
            
            # Critic: Takes ALL observations and ALL actions (CENTRALIZED!)
            critic = Critic(self.total_obs_dim, self.total_action_dim)
            critic_target = Critic(self.total_obs_dim, self.total_action_dim)
            critic_target.load_state_dict(critic.state_dict())
            
            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic))
        
        # Exploration noise
        self.noise_scale = 0.1
    
    def select_actions(self, observations, explore=True):
        """
        DECENTRALIZED ACTION SELECTION
        
        Each agent selects action based on its OWN observation only.
        This is what happens during execution!
        """
        actions = []
        for i, (actor, obs) in enumerate(zip(self.actors, observations)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = actor(obs_tensor).squeeze(0).numpy()
            
            # Add exploration noise during training
            if explore:
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            actions.append(action)
        
        return actions
    
    def update(self, batch):
        """
        CENTRALIZED TRAINING
        
        This is where the magic of CTDE happens:
        1. Critic update uses ALL observations and actions (global view)
        2. Actor update uses policy gradient through centralized critic
        3. Both happen during training only
        """
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        # Concatenate all observations and actions for centralized critic
        all_obs = torch.cat(obs_batch, dim=-1)
        all_actions = torch.cat(action_batch, dim=-1)
        all_next_obs = torch.cat(next_obs_batch, dim=-1)
        
        # Get target actions from all agents' target policies
        all_next_actions = []
        for i in range(self.n_agents):
            with torch.no_grad():
                next_action = self.actors_target[i](next_obs_batch[i])
            all_next_actions.append(next_action)
        all_next_actions = torch.cat(all_next_actions, dim=-1)
        
        # Update each agent
        for i in range(self.n_agents):
            # ─────────────────────────────────────────────────────────────────
            # CRITIC UPDATE (Centralized - sees everything!)
            # ─────────────────────────────────────────────────────────────────
            
            # Current Q-value
            current_q = self.critics[i](all_obs, all_actions)
            
            # Target Q-value using target networks
            with torch.no_grad():
                target_q = self.critics_target[i](all_next_obs, all_next_actions)
                # TD Target: r + γ * Q'(s', a')
                td_target = reward_batch[i] + self.gamma * target_q * (1 - done_batch[i])
            
            # Critic loss: MSE between current Q and TD target
            critic_loss = F.mse_loss(current_q, td_target)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # ─────────────────────────────────────────────────────────────────
            # ACTOR UPDATE (Policy gradient through centralized critic)
            # ─────────────────────────────────────────────────────────────────
            
            # Get current agent's action from its policy
            current_action = self.actors[i](obs_batch[i])
            
            # Replace agent i's action in the joint action
            all_actions_for_actor = []
            for j in range(self.n_agents):
                if j == i:
                    all_actions_for_actor.append(current_action)
                else:
                    all_actions_for_actor.append(action_batch[j].detach())
            all_actions_for_actor = torch.cat(all_actions_for_actor, dim=-1)
            
            # Actor loss: Negative Q-value (we want to maximize Q)
            actor_loss = -self.critics[i](all_obs.detach(), all_actions_for_actor).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        
        # ─────────────────────────────────────────────────────────────────────
        # SOFT UPDATE of target networks
        # ─────────────────────────────────────────────────────────────────────
        self._soft_update()
    
    def _soft_update(self):
        """
        Soft update: θ_target = τ * θ + (1 - τ) * θ_target
        
        This provides stability by slowly updating target networks.
        """
        for i in range(self.n_agents):
            for target_param, param in zip(self.actors_target[i].parameters(), 
                                          self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.critics_target[i].parameters(),
                                          self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE MULTI-AGENT ENVIRONMENT (No external dependencies!)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleMultiAgentEnv:
    """
    Simple Multi-Agent Environment for demonstration
    
    3 agents try to reach 3 target landmarks while avoiding collisions.
    - Cooperative: All agents share the goal
    - Continuous actions: 2D velocity
    - Observations: Own position + landmark positions
    """
    
    def __init__(self, n_agents=3, n_landmarks=3):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.world_size = 2.0
        
        # Observation: own pos (2) + own vel (2) + landmark positions (n_landmarks * 2)
        self.obs_dim = 4 + n_landmarks * 2
        self.action_dim = 2  # 2D velocity
        
        self.reset()
    
    def reset(self):
        # Random agent positions
        self.agent_positions = np.random.uniform(-1, 1, (self.n_agents, 2))
        self.agent_velocities = np.zeros((self.n_agents, 2))
        
        # Random landmark positions
        self.landmark_positions = np.random.uniform(-1, 1, (self.n_landmarks, 2))
        
        self.steps = 0
        return self._get_observations()
    
    def _get_observations(self):
        """Each agent observes: own pos, own vel, all landmark positions"""
        observations = []
        for i in range(self.n_agents):
            obs = np.concatenate([
                self.agent_positions[i],
                self.agent_velocities[i],
                self.landmark_positions.flatten()
            ])
            observations.append(obs)
        return observations
    
    def step(self, actions):
        """
        Execute actions for all agents
        
        Returns:
            observations: List of obs for each agent
            rewards: List of rewards for each agent
            dones: List of done flags
            info: Additional info
        """
        # Apply actions (velocities)
        for i, action in enumerate(actions):
            self.agent_velocities[i] = np.clip(action, -1, 1) * 0.1
            self.agent_positions[i] += self.agent_velocities[i]
            # Keep in bounds
            self.agent_positions[i] = np.clip(self.agent_positions[i], 
                                               -self.world_size, self.world_size)
        
        # Calculate rewards
        rewards = []
        for i in range(self.n_agents):
            # Distance to closest landmark
            distances = np.linalg.norm(
                self.landmark_positions - self.agent_positions[i], axis=1
            )
            min_dist = np.min(distances)
            
            # Reward: negative distance (closer is better)
            reward = -min_dist
            
            # Collision penalty
            for j in range(self.n_agents):
                if i != j:
                    agent_dist = np.linalg.norm(
                        self.agent_positions[i] - self.agent_positions[j]
                    )
                    if agent_dist < 0.1:
                        reward -= 1.0
            
            rewards.append(reward)
        
        self.steps += 1
        dones = [self.steps >= 25] * self.n_agents
        
        return self._get_observations(), rewards, dones, {}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_maddpg(n_episodes=2000, batch_size=256, buffer_size=100000):
    """
    Complete MADDPG Training Loop
    
    Demonstrates all key concepts:
    1. Environment interaction (decentralized execution)
    2. Experience storage (joint replay buffer)
    3. Learning (centralized training)
    """
    
    print("="*70)
    print("MADDPG TRAINING - Demonstrating CTDE Paradigm")
    print("="*70)
    
    # Create environment
    env = SimpleMultiAgentEnv(n_agents=3, n_landmarks=3)
    n_agents = env.n_agents
    obs_dims = [env.obs_dim] * n_agents
    action_dims = [env.action_dim] * n_agents
    
    print(f"\nEnvironment Setup:")
    print(f"  Agents: {n_agents}")
    print(f"  Observation dim per agent: {env.obs_dim}")
    print(f"  Action dim per agent: {env.action_dim}")
    print(f"  Total obs dim (for critic): {sum(obs_dims)}")
    print(f"  Total action dim (for critic): {sum(action_dims)}")
    
    # Create MADDPG agent
    maddpg = MADDPGAgent(
        n_agents=n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01
    )
    
    # Create replay buffer
    buffer = MultiAgentReplayBuffer(
        capacity=buffer_size,
        n_agents=n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Learning starts at: {batch_size} samples")
    
    # Training metrics
    episode_rewards = []
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70 + "\n")
    
    for episode in trange(n_episodes, desc="Training"):
        observations = env.reset()
        episode_reward = 0
        
        for step in range(25):
            # ─────────────────────────────────────────────────────────────────
            # DECENTRALIZED EXECUTION
            # Each agent selects action using only its own observation
            # ─────────────────────────────────────────────────────────────────
            actions = maddpg.select_actions(observations, explore=True)
            
            # Environment step
            next_observations, rewards, dones, _ = env.step(actions)
            
            # ─────────────────────────────────────────────────────────────────
            # STORE JOINT EXPERIENCE
            # Buffer stores transitions from ALL agents together
            # ─────────────────────────────────────────────────────────────────
            buffer.push(observations, actions, rewards, next_observations, dones)
            
            episode_reward += sum(rewards)
            observations = next_observations
            
            # ─────────────────────────────────────────────────────────────────
            # CENTRALIZED TRAINING
            # Random sampling from buffer + update with global information
            # ─────────────────────────────────────────────────────────────────
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                maddpg.update(batch)
        
        episode_rewards.append(episode_reward)
        
        # Progress logging
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            print(f"\nEpisode {episode + 1}")
            print(f"  Avg Reward (200 ep): {avg_reward:.2f}")
            print(f"  Buffer Size: {len(buffer)}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    return maddpg, episode_rewards


def evaluate_agents(maddpg, n_episodes=10):
    """
    DECENTRALIZED EXECUTION (Evaluation)
    
    During evaluation:
    - Each agent uses ONLY its local policy (actor)
    - No centralized critic needed!
    - No communication between agents!
    """
    print("\n" + "="*70)
    print("EVALUATION - Decentralized Execution Only")
    print("="*70)
    print("\nNote: Critics are NOT used during execution!")
    print("Each agent acts based solely on its local observation.\n")
    
    env = SimpleMultiAgentEnv(n_agents=3, n_landmarks=3)
    
    total_rewards = []
    
    for episode in range(n_episodes):
        observations = env.reset()
        episode_reward = 0
        
        for step in range(25):
            # Decentralized action selection (no exploration)
            actions = maddpg.select_actions(observations, explore=False)
            observations, rewards, dones, _ = env.step(actions)
            episode_reward += sum(rewards)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")
    
    print(f"\nAverage Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    return total_rewards


def plot_results(episode_rewards):
    """Plot training curve"""
    plt.figure(figsize=(12, 5))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Moving average
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('MADDPG Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards[-500:], bins=30, color='green', edgecolor='black', alpha=0.7)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution (Last 500 Episodes)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('maddpg_training_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved training plot to 'maddpg_training_results.png'")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MADDPG FROM SCRATCH DEMONSTRATION                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This script demonstrates all key MARL concepts:                            ║
║                                                                              ║
║  1. CTDE (Centralized Training, Decentralized Execution)                    ║
║     - Critics see ALL observations and actions during training              ║
║     - Actors only use LOCAL observations during execution                   ║
║                                                                              ║
║  2. Replay Buffer                                                           ║
║     - Stores JOINT experiences from all agents                              ║
║     - RANDOM sampling breaks temporal correlation                           ║
║                                                                              ║
║  3. Actor-Critic Architecture                                               ║
║     - Each agent has its own actor (policy) and critic (Q-function)         ║
║     - Soft target updates for stability                                     ║
║                                                                              ║
║  No external dependencies required! Just PyTorch and NumPy.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Train
    maddpg, rewards = train_maddpg(n_episodes=2000, batch_size=256)
    
    # Evaluate
    evaluate_agents(maddpg, n_episodes=10)
    
    # Plot results
    plot_results(rewards)
    
    print("\n✓ Demonstration complete!")
    print("\nKey Takeaways:")
    print("  • Centralized critics stabilize learning in multi-agent settings")
    print("  • Decentralized actors enable practical deployment")
    print("  • Random sampling from replay buffer is essential for stability")
    print("  • Soft updates prevent target network oscillation")
