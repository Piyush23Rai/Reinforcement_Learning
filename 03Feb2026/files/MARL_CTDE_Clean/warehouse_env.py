"""
Simple Multi-Warehouse Environment
No infinite loops, straightforward logic
"""

import numpy as np


class WarehouseEnv:
    """Simple multi-warehouse inventory environment"""
    
    def __init__(self, num_warehouses=3, max_steps=50):
        self.num_warehouses = num_warehouses
        self.max_steps = max_steps
        self.step_count = 0
        
        # Inventory levels
        self.inventory = np.random.rand(num_warehouses) * 100
        self.max_inventory = 100
        self.demand = None
        
        # Hyperparameters
        self.stockout_cost = 10.0
        self.transfer_cost = 0.5
        self.holding_cost = 0.1
    
    def reset(self):
        """Reset environment"""
        self.inventory = np.random.rand(self.num_warehouses) * 50 + 25
        self.step_count = 0
        self._sample_demand()
        return self._get_observations()
    
    def _sample_demand(self):
        """Sample random demand"""
        self.demand = np.random.normal(30, 10, self.num_warehouses)
        self.demand = np.clip(self.demand, 0, 50)
    
    def _get_observations(self):
        """Get observations for all warehouses"""
        obs = []
        for i in range(self.num_warehouses):
            # Normalize to [0, 1]
            o = np.array([
                self.inventory[i] / self.max_inventory,
                self.demand[i] / 50.0,
            ])
            obs.append(o)
        return np.array(obs)
    
    def step(self, actions):
        """Execute one step"""
        # Clip actions to [0, 1]
        actions = np.clip(actions, 0, 1)
        
        # Order amounts
        order_amounts = actions * 30  # Max order 30 units
        
        # Add inventory from orders
        self.inventory += order_amounts
        self.inventory = np.clip(self.inventory, 0, self.max_inventory)
        
        # Fulfill demand
        fulfilled = np.minimum(self.inventory, self.demand)
        stockouts = self.demand - fulfilled
        self.inventory -= fulfilled
        
        # Compute costs
        stockout_costs = stockouts * self.stockout_cost
        holding_costs = self.inventory * self.holding_cost
        total_costs = stockout_costs + holding_costs
        
        # Rewards (negative costs)
        rewards = -total_costs
        
        # Update step
        self.step_count += 1
        done = self.step_count >= self.max_steps
        dones = np.array([done] * self.num_warehouses)
        
        # Sample new demand
        self._sample_demand()
        obs = self._get_observations()
        
        return obs, rewards, dones, {}
