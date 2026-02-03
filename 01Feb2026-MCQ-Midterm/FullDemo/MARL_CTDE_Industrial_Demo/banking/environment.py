"""
Banking Transaction Routing Environment

Problem:
--------
Route transactions across multiple channels optimally.

Channels:
- Internal (fast, lower cost, higher risk)
- External (moderate latency, moderate cost/risk)
- Blockchain (slower, higher cost, lower risk)

State:
------
Per agent: [pending_transactions, channel_load, channel_risk, avg_latency]

Actions:
--------
Per agent: discrete {0, 1, 2}
- 0 = internal channel
- 1 = external channel
- 2 = blockchain channel

Rewards:
--------
R = -latency_cost * latency - risk_penalty * (risk - threshold)² - cost

The system must:
1. Minimize average transaction latency
2. Keep portfolio risk below threshold (5%)
3. Minimize transaction costs
4. Avoid channel overload
"""

import numpy as np
from typing import Tuple, Dict, List


class TransactionRoutingEnv:
    """
    Multi-agent transaction routing environment.
    
    Each agent (transaction router) decides which channel to use
    for its transactions based on current system state.
    """
    
    def __init__(self, 
                 num_agents: int = 3,
                 num_channels: int = 3,
                 num_transactions_per_episode: int = 100,
                 latency_target: float = 100.0,
                 risk_threshold: float = 0.05):
        """
        Initialize environment.
        
        Args:
            num_agents (int): Number of transaction routers
            num_channels (int): Number of available channels
            num_transactions_per_episode (int): Transactions to process
            latency_target (float): Target latency in ms
            risk_threshold (float): Maximum allowed portfolio risk
        """
        self.num_agents = num_agents
        self.num_channels = num_channels
        self.num_transactions = num_transactions_per_episode
        self.latency_target = latency_target
        self.risk_threshold = risk_threshold
        
        # Channel characteristics
        self.channel_latencies = np.array([20.0, 80.0, 200.0])  # ms
        self.channel_risks = np.array([0.03, 0.02, 0.01])       # Risk per transaction
        self.channel_costs = np.array([0.1, 0.5, 2.0])          # Cost per transaction
        self.channel_capacities = np.array([50.0, 100.0, 200.0])  # Max transactions
        
        # State
        self.pending_transactions = np.zeros(num_agents)
        self.channel_loads = np.zeros(num_channels)
        self.cumulative_risk = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0
        
        self.step_count = 0
        self.max_steps = num_transactions_per_episode
        
        # Metrics
        self.episode_latencies = []
        self.episode_risks = []
        self.episode_costs = []
        self.episode_risk_violations = 0
    
    def reset(self) -> np.ndarray:
        """
        Reset environment.
        
        Returns:
            np.ndarray: Initial observations for all agents
        """
        # Initialize random pending transactions
        self.pending_transactions = np.random.poisson(5, self.num_agents).astype(float)
        self.channel_loads = np.zeros(self.num_channels)
        self.cumulative_risk = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0
        
        self.step_count = 0
        self.episode_latencies = []
        self.episode_risks = []
        self.episode_costs = []
        self.episode_risk_violations = 0
        
        return self._get_observations()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute one step.
        
        Args:
            actions (np.ndarray): Channel selections for each agent (0, 1, or 2)
            
        Returns:
            Tuple: (observations, rewards, dones, info)
        """
        actions = np.clip(actions, 0, 2).astype(int)
        
        # Track transactions routed and metrics
        rewards = np.zeros(self.num_agents)
        latencies = np.zeros(self.num_agents)
        risks = np.zeros(self.num_agents)
        costs = np.zeros(self.num_agents)
        
        # Process transactions
        total_transactions = self.pending_transactions.sum()
        
        if total_transactions > 0:
            for agent_id in range(self.num_agents):
                num_trans = int(self.pending_transactions[agent_id])
                if num_trans > 0:
                    channel = actions[agent_id]
                    
                    # Get channel metrics
                    latency = self.channel_latencies[channel]
                    risk = self.channel_risks[channel] * num_trans
                    cost = self.channel_costs[channel] * num_trans
                    
                    latencies[agent_id] = latency
                    risks[agent_id] = risk
                    costs[agent_id] = cost
                    
                    self.channel_loads[channel] += num_trans
                    self.cumulative_latency += latency * num_trans
                    self.cumulative_risk += risk
                    self.cumulative_cost += cost
                    
                    self.episode_latencies.append(latency)
                    self.episode_risks.append(risk)
                    self.episode_costs.append(cost)
            
            # Normalize cumulative metrics
            avg_latency = self.cumulative_latency / total_transactions if total_transactions > 0 else 0
            
            # Compute rewards
            # Goal: minimize latency, keep risk below threshold, minimize cost
            latency_penalty = (avg_latency - self.latency_target) / 100.0
            
            risk_excess = max(0, self.cumulative_risk - self.risk_threshold)
            risk_penalty = (risk_excess ** 2) * 100
            
            cost_penalty = self.cumulative_cost / 100.0
            
            # Individual rewards (collaborative)
            for agent_id in range(self.num_agents):
                num_trans = int(self.pending_transactions[agent_id])
                if num_trans > 0:
                    # Penalty shared across agents
                    shared_penalty = (latency_penalty + risk_penalty + cost_penalty) / self.num_agents
                    rewards[agent_id] = -shared_penalty
                else:
                    rewards[agent_id] = 0
            
            # Track risk violations
            if self.cumulative_risk > self.risk_threshold:
                self.episode_risk_violations += 1
        
        # Sample new transactions
        self.pending_transactions = np.random.poisson(5, self.num_agents).astype(float)
        self.channel_loads = np.zeros(self.num_channels)  # Reset channel loads each step
        
        # Update step
        self.step_count += 1
        done = self.step_count >= self.max_steps
        dones = np.array([done] * self.num_agents)
        
        # Next observations
        observations = self._get_observations()
        
        # Info
        info = {
            'avg_latency': np.mean(latencies) if total_transactions > 0 else 0,
            'total_risk': self.cumulative_risk,
            'total_cost': self.cumulative_cost,
            'risk_violations': self.episode_risk_violations,
            'step': self.step_count
        }
        
        return observations, rewards, dones, info
    
    def _get_observations(self) -> np.ndarray:
        """
        Get observations for all agents.
        
        Each agent observes:
        - Its own pending transactions
        - Channel loads (system state)
        - Current risk level
        - Average latency
        
        Returns:
            np.ndarray: Observations (num_agents, obs_dim)
        """
        observations = []
        
        for agent_id in range(self.num_agents):
            # Normalize values
            obs = [
                self.pending_transactions[agent_id] / 20.0,  # Pending transactions
                self.cumulative_risk / 0.1,                  # Current risk
                self.cumulative_cost / 100.0,                # Cumulative cost
                np.mean(self.channel_loads) / 50.0,          # Average channel load
            ]
            observations.append(obs)
        
        return np.array(observations)
    
    @property
    def observation_space_size(self) -> int:
        """Size of observation space per agent."""
        return 4
    
    @property
    def action_space_size(self) -> int:
        """Size of action space per agent (number of channels)."""
        return self.num_channels


# Test
if __name__ == "__main__":
    env = TransactionRoutingEnv(num_agents=3, num_channels=3)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Obs: {obs}")
    
    for step in range(10):
        actions = np.random.randint(0, 3, 3)  # Random channel selection
        obs, rewards, dones, info = env.step(actions)
        print(f"Step {info['step']}: Reward={rewards.sum():.3f}, "
              f"Latency={info['avg_latency']:.1f}ms, Risk={info['total_risk']:.4f}")
    
    print(f"\nFinal metrics:")
    print(f"Total cost: {info['total_cost']:.2f}")
    print(f"Risk violations: {info['risk_violations']}")
