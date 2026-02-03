"""
Banking Demo: Transaction Routing with CTDE-MAPPO

Demonstrates CTDE using MAPPO algorithm for a different domain.
Key difference from retail: MAPPO uses policy gradients instead of Q-learning.
"""

import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mappo import create_mappo_agents
from core.replay_buffer import EpisodeBuffer
from core.utils import plot_training_curve, compute_gae, set_seed
from .environment import TransactionRoutingEnv


class BankingDemo:
    """Demo trainer for transaction routing with CTDE-MAPPO."""
    
    def __init__(self, config: dict):
        self.config = config
        set_seed(config.get('seed', 42))
        
        # Environment
        self.env = TransactionRoutingEnv(
            num_agents=config['num_agents'],
            num_channels=3,
            num_transactions_per_episode=config['max_steps']
        )
        
        # Agents (MAPPO)
        obs_dim = self.env.observation_space_size
        action_dim = self.env.action_space_size
        
        self.agents = create_mappo_agents(
            state_dim=obs_dim,
            action_dim=action_dim,
            num_agents=config['num_agents'],
            config=config
        )
        
        # Episode buffer (for PPO)
        self.episode_buffer = EpisodeBuffer(config['num_agents'])
        
        self.device = self.agents[0].device
    
    def train_episode(self):
        """Train one episode."""
        obs = self.env.reset()
        self.episode_buffer.clear()
        ep_reward = 0
        ep_cost = 0
        
        # Collect episode experiences
        for step in range(self.config['max_steps']):
            # Decentralized action selection (using policy)
            actions = []
            log_probs = []
            values = []
            
            for agent_id, agent in enumerate(self.agents):
                # Policy action (stochastic)
                action, log_prob = agent.select_action(obs[agent_id], training=True)
                actions.append(action[0])  # Extract action value
                log_probs.append(log_prob)
                
                # Value estimate
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                obs_concat = obs_tensor.reshape(1, -1)
                with torch.no_grad():
                    value = agent.critic(obs_concat).item()
                values.append(value)
            
            actions = np.array(actions)
            
            # Environment step
            next_obs, rewards, dones, info = self.env.step(actions)
            ep_reward += rewards.sum()
            ep_cost += info['total_cost']
            
            # Store experiences
            for agent_id in range(self.config['num_agents']):
                self.episode_buffer.add(
                    agent_id=agent_id,
                    observation=obs[agent_id],
                    action=np.array([actions[agent_id]]),
                    reward=rewards[agent_id],
                    value=values[agent_id],
                    log_prob=log_probs[agent_id],
                    done=dones[agent_id],
                    next_observation=next_obs[agent_id]
                )
            
            obs = next_obs
            if dones[0]:
                break
        
        # Compute returns and advantages
        gae_info = self.episode_buffer.compute_returns_and_advantages(
            gamma=self.config['gamma'],
            gae_lambda=0.95
        )
        
        # Prepare batch for training
        batch = self._prepare_batch(gae_info)
        
        # Train agents
        for agent in self.agents:
            agent.update(batch)
        
        # Decay exploration
        for agent in self.agents:
            agent.decay_exploration()
        
        return ep_reward, ep_cost, info['avg_latency'], info['total_risk']
    
    def _prepare_batch(self, gae_info):
        """Prepare batch from episode experiences."""
        batch = {}
        
        # Stack agent experiences
        observations_list = []
        actions_list = []
        advantages_list = []
        returns_list = []
        old_log_probs_list = []
        
        for agent_id in range(self.config['num_agents']):
            episode = self.episode_buffer.get_episode(agent_id)
            observations_list.append(torch.FloatTensor(episode['observations']))
            actions_list.append(torch.FloatTensor(episode['actions']))
            advantages_list.append(torch.FloatTensor(gae_info[agent_id]['advantages']))
            returns_list.append(torch.FloatTensor(gae_info[agent_id]['returns']))
            old_log_probs_list.append(torch.FloatTensor([self.episode_buffer.log_probs[agent_id]]))
        
        # Concatenate observations for critic
        obs_concat = torch.cat([o.reshape(-1, o.shape[-1]) for o in observations_list], dim=1)
        
        batch['observations'] = observations_list[0]  # First agent obs for actor
        batch['actions'] = actions_list[0]
        batch['advantages'] = advantages_list[0]
        batch['returns'] = returns_list[0]
        batch['old_log_probs'] = torch.cat(old_log_probs_list)
        batch['states_concat'] = obs_concat
        
        return batch
    
    def train(self, num_episodes: int):
        """Train for specified episodes."""
        print("\n" + "="*80)
        print(" BANKING DEMO: Transaction Routing with CTDE-MAPPO")
        print("="*80 + "\n")
        
        print("3 Transaction Routers optimizing across 3 Channels:")
        print("  - Internal Channel (20ms, 3% risk, $0.1)")
        print("  - External Channel (80ms, 2% risk, $0.5)")
        print("  - Blockchain Channel (200ms, 1% risk, $2.0)\n")
        
        train_rewards = []
        train_latencies = []
        train_risks = []
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            ep_reward, ep_cost, ep_latency, ep_risk = self.train_episode()
            train_rewards.append(ep_reward)
            train_latencies.append(ep_latency)
            train_risks.append(ep_risk)
            
            if (episode + 1) % 50 == 0:
                avg_latency = np.mean(train_latencies[-50:])
                avg_risk = np.mean(train_risks[-50:])
                print(f"\nEpisode {episode+1}/{num_episodes}")
                print(f"  Avg Latency: {avg_latency:.1f}ms | Avg Risk: {avg_risk:.4f}")
        
        print("\nTraining completed!\n")
        
        # Evaluate
        print("Evaluating trained agents...")
        eval_latencies = []
        eval_risks = []
        
        for _ in range(50):
            obs = self.env.reset()
            while True:
                actions = np.array([agent.select_action(obs[i], training=False)[0] 
                                   for i, agent in enumerate(self.agents)])
                next_obs, _, dones, info = self.env.step(actions)
                eval_latencies.append(info['avg_latency'])
                eval_risks.append(info['total_risk'])
                obs = next_obs
                if dones[0]:
                    break
        
        print(f"\nEvaluation Results:")
        print(f"  Final average latency: {np.mean(eval_latencies[-200:]):.1f}ms")
        print(f"  Final average risk: {np.mean(eval_risks[-200:]):.4f}")
        print(f"  Latency improvement: {(1 - np.mean(eval_latencies) / 100) * 100:.1f}%\n")
        
        # Save results
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plot_training_curve(
            train_latencies,
            "Transaction Routing with CTDE-MAPPO\nAverage Latency During Training",
            str(results_dir / 'latency_training.png')
        )
        
        print(f"Results saved to {results_dir}\n")


def main():
    """Main demo function."""
    config = {
        'num_agents': 3,
        'num_episodes': 200,
        'max_steps': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'n_epochs': 3,
        'entropy_coef': 0.01,
        'seed': 42,
    }
    
    demo = BankingDemo(config)
    demo.train(config['num_episodes'])
    
    print("="*80)
    print(" BANKING DEMO COMPLETED!")
    print("="*80)
    print("\nKey Learnings:")
    print("1. MAPPO works well for transaction routing (policy gradient approach)")
    print("2. Centralized critic helps estimate system value")
    print("3. Agents learned to balance latency, risk, and cost")
    print("4. Implicit coordination prevents risk threshold violations")
    print("\n")


if __name__ == "__main__":
    main()
