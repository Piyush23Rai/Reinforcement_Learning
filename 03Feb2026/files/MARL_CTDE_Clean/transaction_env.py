"""
Simple Transaction Routing Environment
"""

import numpy as np


class TransactionEnv:
    """Simple transaction routing environment"""
    
    def __init__(self, num_agents=3, max_steps=100):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.step_count = 0
        
        # Channel characteristics
        # latency (ms), cost, risk per transaction
        self.channels = {
            0: {'latency': 20, 'cost': 0.5, 'risk': 0.03},     # Fast
            1: {'latency': 100, 'cost': 1.0, 'risk': 0.02},    # Medium
            2: {'latency': 300, 'cost': 2.0, 'risk': 0.01},    # Slow but safe
        }
        
        self.pending_transactions = None
        self.total_risk = 0.0
    
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        self.total_risk = 0.0
        self._sample_transactions()
        return self._get_observations()
    
    def _sample_transactions(self):
        """Sample random pending transactions"""
        self.pending_transactions = np.random.randint(5, 20, self.num_agents)
    
    def _get_observations(self):
        """Get observations"""
        obs = []
        for i in range(self.num_agents):
            o = np.array([
                self.pending_transactions[i] / 30.0,      # Pending
                self.total_risk / 0.1,                    # Total risk
            ])
            obs.append(o)
        return np.array(obs)
    
    def step(self, actions):
        """Execute one step"""
        # Actions: discrete channel choice (0, 1, 2)
        actions = np.clip(actions, 0, 2).astype(int).flatten()
        
        # Process transactions
        rewards = np.zeros(self.num_agents)
        total_latency = 0
        total_cost = 0
        risk_penalty = 0
        
        for i in range(self.num_agents):
            num_trans = self.pending_transactions[i]
            channel = actions[i]
            
            # Get channel metrics
            latency = self.channels[channel]['latency'] * num_trans
            cost = self.channels[channel]['cost'] * num_trans
            risk = self.channels[channel]['risk'] * num_trans
            
            total_latency += latency
            total_cost += cost
            self.total_risk += risk
            
            # Reward: minimize latency and cost, penalize high risk
            latency_penalty = latency / 1000.0
            cost_penalty = cost / 5.0
            risk_penalty_i = max(0, (self.total_risk - 0.15) ** 2) * 10
            
            rewards[i] = -(latency_penalty + cost_penalty + risk_penalty_i)
        
        # Update
        self.step_count += 1
        done = self.step_count >= self.max_steps
        dones = np.array([done] * self.num_agents)
        
        # Sample new transactions
        self._sample_transactions()
        obs = self._get_observations()
        
        return obs, rewards, dones, {'total_risk': self.total_risk, 'total_cost': total_cost}
