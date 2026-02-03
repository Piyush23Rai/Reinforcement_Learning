"""
Simple, Clean MADDPG Implementation
No infinite loops, no complex abstractions, just working code
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class SimpleActor(nn.Module):
    """Simple actor network for local policy"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)


class SimpleCritic(nn.Module):
    """Simple critic network - sees all agents"""
    def __init__(self, total_state_dim, total_action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


class MADDPGAgent:
    """Single MADDPG agent"""
    
    def __init__(self, agent_id, state_dim, action_dim, num_agents, device='cpu', lr=0.001):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        self.lr = lr
        
        # Networks
        self.actor = SimpleActor(state_dim, action_dim).to(device)
        self.actor_target = SimpleActor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = SimpleCritic(state_dim * num_agents, action_dim * num_agents).to(device)
        self.critic_target = SimpleCritic(state_dim * num_agents, action_dim * num_agents).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.001
        self.epsilon = 1.0
        
    def select_action(self, state, training=True):
        """Select action (decentralized - only uses local state)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().item()

        # Add noise during training
        if training:
            noise = np.random.normal(0, 0.1)
            action = np.clip(action + noise, -1, 1)

        return action
    
    def update(self, batch, all_agents):
        """Update networks with batch"""
        states, actions, rewards, next_states, dones = batch
        
        batch_size = states.shape[0]
        
        # Get own data
        own_states = states[:, self.agent_id]
        own_actions = actions[:, self.agent_id]
        own_rewards = rewards[:, self.agent_id]
        own_next_states = next_states[:, self.agent_id]
        own_dones = dones[:, self.agent_id]
        
        # Flatten for critic
        states_flat = states.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        next_states_flat = next_states.reshape(batch_size, -1)
        
        # === CRITIC UPDATE ===
        with torch.no_grad():
            # Get target actions from all agents
            target_actions = []
            for i, agent in enumerate(all_agents):
                next_state_i = next_states[:, i]
                next_state_tensor = torch.FloatTensor(next_state_i).to(self.device)
                action_i = agent.actor_target(next_state_tensor)
                target_actions.append(action_i)
            
            target_actions_flat = torch.cat(target_actions, dim=1)
            next_states_tensor = torch.FloatTensor(next_states_flat).to(self.device)
            
            # Target Q
            target_q = self.critic_target(next_states_tensor, target_actions_flat)
            target_q = torch.FloatTensor(own_rewards).unsqueeze(1).to(self.device) + \
                       self.gamma * target_q * (1 - torch.FloatTensor(own_dones).unsqueeze(1).to(self.device))
        
        # Current Q
        states_tensor = torch.FloatTensor(states_flat).to(self.device)
        actions_tensor = torch.FloatTensor(actions_flat).to(self.device)
        current_q = self.critic(states_tensor, actions_tensor)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # === ACTOR UPDATE ===
        own_states_tensor = torch.FloatTensor(own_states).to(self.device)
        current_actions = []
        for i, agent in enumerate(all_agents):
            state_i = states[:, i]
            state_tensor = torch.FloatTensor(state_i).to(self.device)
            if i == self.agent_id:
                action_i = self.actor(state_tensor)
            else:
                action_i = agent.actor(state_tensor)
            current_actions.append(action_i)
        
        current_actions_flat = torch.cat(current_actions, dim=1)
        actor_q = self.critic(states_tensor, current_actions_flat)
        
        actor_loss = -actor_q.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()


class ReplayBuffer:
    """Simple replay buffer"""
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, states, actions, rewards, next_states, dones):
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size):
        """Sample batch"""
        if len(self.buffer) < batch_size:
            indices = range(len(self.buffer))
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones
    
    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size
