"""
Retail Multi-Warehouse Inventory Optimization Environment

Problem:
--------
Multiple warehouses (agents) must collaborate to minimize total inventory costs.

Each warehouse:
- Receives stochastic customer demand
- Can order inventory
- Can transfer inventory to neighboring warehouses
- Incurs costs for stockouts, transfers, and holding excess inventory

State:
------
Per warehouse: [current_inventory, customer_demand, neighbor_inventory_levels]

Actions:
--------
Per warehouse: continuous [0, 1]
- 0 = no restocking
- 1 = maximum inventory ordering

Rewards:
--------
Negative cost: -(stockout_cost + transfer_cost + holding_cost)

The goal is for warehouses to cooperate (transfer inventory efficiently)
to minimize total system cost.
"""

import numpy as np
from typing import Tuple, List, Dict
import networkx as nx


class MultiWarehouseEnv:
    """
    Multi-warehouse inventory management environment.
    
    Warehouses are arranged in a network topology where transfers can occur
    between adjacent warehouses.
    """
    
    def __init__(self, 
                 num_warehouses: int = 5,
                 warehouse_capacity: float = 1000.0,
                 demand_mean: float = 500.0,
                 demand_std: float = 200.0,
                 transfer_capacity: float = 200.0,
                 stockout_cost: float = 10.0,
                 transfer_cost: float = 0.5,
                 holding_cost: float = 0.1,
                 topology: str = 'line'):
        """
        Initialize environment.
        
        Args:
            num_warehouses (int): Number of warehouses
            warehouse_capacity (float): Max inventory per warehouse
            demand_mean (float): Mean customer demand per warehouse
            demand_std (float): Std dev of demand
            transfer_capacity (float): Max inventory to transfer per step
            stockout_cost (float): Cost per unit of unmet demand
            transfer_cost (float): Cost per unit transferred
            holding_cost (float): Cost per unit of excess inventory
            topology (str): Network topology ('line', 'ring', 'star', 'complete')
        """
        self.num_warehouses = num_warehouses
        self.warehouse_capacity = warehouse_capacity
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.transfer_capacity = transfer_capacity
        self.stockout_cost = stockout_cost
        self.transfer_cost = transfer_cost
        self.holding_cost = holding_cost
        
        # Create network topology
        self.graph = self._create_topology(topology)
        self.neighbors = {i: list(self.graph.neighbors(i)) for i in range(num_warehouses)}
        
        # State
        self.inventory = np.zeros(num_warehouses)
        self.demand = np.zeros(num_warehouses)
        self.step_count = 0
        self.max_steps = 100
        
        # Metrics
        self.episode_costs = np.zeros(num_warehouses)
        self.episode_stockouts = np.zeros(num_warehouses)
        self.episode_transfers = np.zeros(num_warehouses)
    
    def _create_topology(self, topology: str) -> nx.Graph:
        """
        Create warehouse network topology.
        
        Args:
            topology (str): Type of topology
            
        Returns:
            nx.Graph: Network graph
        """
        if topology == 'line':
            # Linear chain: 0-1-2-3-4
            graph = nx.path_graph(self.num_warehouses)
        elif topology == 'ring':
            # Ring: 0-1-2-3-4-0
            graph = nx.cycle_graph(self.num_warehouses)
        elif topology == 'star':
            # Star: all connected to central node
            graph = nx.star_graph(self.num_warehouses - 1)
        elif topology == 'complete':
            # Complete graph: all connected to all
            graph = nx.complete_graph(self.num_warehouses)
        else:
            graph = nx.path_graph(self.num_warehouses)
        
        return graph
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            np.ndarray: Initial observations for all warehouses
        """
        # Initialize inventory
        self.inventory = np.random.uniform(
            self.warehouse_capacity * 0.3,
            self.warehouse_capacity * 0.7,
            self.num_warehouses
        )
        
        # Sample initial demand
        self._sample_demand()
        
        # Reset metrics
        self.step_count = 0
        self.episode_costs = np.zeros(self.num_warehouses)
        self.episode_stockouts = np.zeros(self.num_warehouses)
        self.episode_transfers = np.zeros(self.num_warehouses)
        
        return self._get_observations()
    
    def _sample_demand(self):
        """Sample customer demand for this step."""
        self.demand = np.random.normal(self.demand_mean, self.demand_std, self.num_warehouses)
        self.demand = np.clip(self.demand, 0, self.warehouse_capacity * 2)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute one environment step.
        
        Args:
            actions (np.ndarray): Actions for each warehouse (num_warehouses,)
                                 Value in [0, 1]: restock amount
        
        Returns:
            Tuple: (observations, rewards, dones, info)
        """
        # Validate and normalize actions to shape (num_warehouses,)
        actions = np.asarray(actions)
        # Remove trailing singleton dimensions (e.g., shape (N,1) -> (N,))
        actions = np.squeeze(actions)
        # If a single scalar was provided, expand to all warehouses
        if actions.ndim == 0:
            actions = np.full(self.num_warehouses, float(actions))
        else:
            actions = actions.flatten()

        if actions.shape[0] != self.num_warehouses:
            raise ValueError(f"Expected actions for {self.num_warehouses} warehouses, got shape {actions.shape}")

        actions = np.clip(actions, 0, 1)
        
        # Convert actions to order quantities
        # action=0 → order 0, action=1 → order max_order_quantity
        order_amounts = actions * (self.warehouse_capacity * 0.5)
        
        # ===== Step 1: Fulfill demand from current inventory =====
        fulfilled = np.minimum(self.inventory, self.demand)
        stockout = self.demand - fulfilled
        
        self.inventory -= fulfilled
        self.episode_stockouts += stockout
        
        # ===== Step 2: Coordinate inventory transfers between neighbors =====
        # Each warehouse can transfer inventory to neighbors
        transfers_sent = np.zeros(self.num_warehouses)
        transfers_received = np.zeros(self.num_warehouses)
        
        for i in range(self.num_warehouses):
            # Get neighbors
            neighbor_list = self.neighbors[i]
            
            if len(neighbor_list) > 0 and self.inventory[i] > self.warehouse_capacity * 0.5:
                # If inventory is high, offer to transfer to neighbors with low inventory
                for j in neighbor_list:
                    if self.inventory[j] < self.warehouse_capacity * 0.3:
                        # Transfer inventory
                        transfer_amount = min(
                            self.transfer_capacity,
                            self.inventory[i] - self.warehouse_capacity * 0.4
                        )
                        transfer_amount = max(0, transfer_amount)
                        
                        self.inventory[i] -= transfer_amount
                        self.inventory[j] += transfer_amount
                        transfers_sent[i] += transfer_amount
                        transfers_received[j] += transfer_amount
        
        self.episode_transfers += transfers_sent
        
        # ===== Step 3: Receive new orders =====
        # Orders arrive after some lead time (simplified: arrive immediately)
        self.inventory += order_amounts
        self.inventory = np.clip(self.inventory, 0, self.warehouse_capacity)
        
        # ===== Step 4: Compute costs =====
        stockout_costs = stockout * self.stockout_cost
        transfer_costs = transfers_sent * self.transfer_cost
        holding_costs = self.inventory * self.holding_cost
        
        total_costs = stockout_costs + transfer_costs + holding_costs
        self.episode_costs += total_costs
        
        # Rewards: negative costs (we want to minimize cost)
        rewards = -total_costs
        
        # Update step
        self.step_count += 1
        done = self.step_count >= self.max_steps
        dones = np.array([done] * self.num_warehouses)
        
        # Next observations
        self._sample_demand()
        observations = self._get_observations()
        
        # Info
        info = {
            'stockout': stockout,
            'transfers_sent': transfers_sent,
            'total_cost': total_costs.sum(),
            'inventory': self.inventory.copy(),
            'step': self.step_count
        }
        
        return observations, rewards, dones, info
    
    def _get_observations(self) -> np.ndarray:
        """
        Get observations for all warehouses.
        
        Each warehouse observes:
        - Its own inventory level
        - Its own demand
        - Neighbor inventory levels
        
        Returns:
            np.ndarray: Observations (num_warehouses, obs_dim)
        """
        observations = []
        
        for i in range(self.num_warehouses):
            # Normalize values to [0, 1]
            obs = [
                self.inventory[i] / self.warehouse_capacity,  # Own inventory
                self.demand[i] / (self.demand_mean * 2),      # Own demand
            ]
            
            # Add neighbor inventory
            for j in self.neighbors[i]:
                obs.append(self.inventory[j] / self.warehouse_capacity)
            
            # Pad to fixed size if fewer neighbors than max
            # max_neighbors should be the maximum number of neighbors any node has
            max_neighbors = max((len(neighbors) for neighbors in self.neighbors.values()), default=0)
            # Ensure at least 1 to match observation_space_size logic
            max_neighbors = max(max_neighbors, 1)
            while len(obs) < 2 + max_neighbors:
                obs.append(0.0)
            
            observations.append(obs)
        
        return np.array(observations)
    
    def render(self, mode: str = 'human'):
        """
        Render environment state.
        
        Args:
            mode (str): Render mode ('human', etc.)
        """
        if mode == 'human':
            print(f"\n--- Step {self.step_count} ---")
            print(f"Inventory: {self.inventory.round(1)}")
            print(f"Demand: {self.demand.round(1)}")
            print(f"Episode Cost: {self.episode_costs.sum():.2f}")
            print(f"Episode Stockouts: {self.episode_stockouts.sum():.2f}")
    
    def get_state_dict(self) -> Dict:
        """Get current environment state as dictionary."""
        return {
            'inventory': self.inventory.copy(),
            'demand': self.demand.copy(),
            'step': self.step_count,
            'costs': self.episode_costs.copy(),
            'stockouts': self.episode_stockouts.copy(),
        }
    
    @property
    def observation_space_size(self) -> int:
        """Size of observation space per agent."""
        # 2 (own inv + own demand) + max_neighbors (neighbor inventories)
        max_neighbors = max(len(neighbors) for neighbors in self.neighbors.values())
        return 2 + max_neighbors
    
    @property
    def action_space_size(self) -> int:
        """Size of action space per agent."""
        return 1  # Continuous action


# Test the environment
if __name__ == "__main__":
    env = MultiWarehouseEnv(num_warehouses=5)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Obs: {obs}")
    
    for _ in range(10):
        actions = np.random.uniform(0, 1, 5)
        obs, rewards, dones, info = env.step(actions)
        print(f"Step {info['step']}: Reward={rewards.sum():.2f}, Cost={info['total_cost']:.2f}")
    
    print(f"\nFinal inventory: {env.inventory}")
    print(f"Total episode cost: {env.episode_costs.sum():.2f}")
