"""
================================================================================
UCB (Upper Confidence Bound) - Complete Banking Sector Demo
================================================================================

Business Problem:
-----------------
A bank wants to optimize its credit card marketing campaigns. They have 5 
different promotional offers but don't know which one has the highest 
customer response rate.

Goal: Find the best offer while minimizing wasted marketing budget
      (i.e., minimize regret)

This demo covers:
1. Synthetic data generation (simulating customer responses)
2. UCB algorithm implementation
3. Comparison with Random and Greedy strategies
4. Visualization of results
5. Regret analysis
6. Business insights

Author: Banking Analytics Team
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# SECTION 1: DEFINE THE BANKING OFFERS (ARMS)
# ==============================================================================

@dataclass
class CreditCardOffer:
    """Represents a credit card promotional offer"""
    name: str
    description: str
    true_response_rate: float  # Hidden from the algorithm!
    avg_revenue_per_conversion: float  # Revenue if customer accepts
    
    def simulate_customer_response(self) -> Tuple[bool, float]:
        """
        Simulate a customer's response to this offer.
        Returns: (accepted: bool, revenue: float)
        """
        accepted = np.random.random() < self.true_response_rate
        revenue = self.avg_revenue_per_conversion if accepted else 0
        return accepted, revenue


def create_bank_offers() -> List[CreditCardOffer]:
    """
    Create 5 credit card offers with different (hidden) response rates.
    In real world, these rates are unknown and must be learned!
    """
    offers = [
        CreditCardOffer(
            name="5% Cashback",
            description="5% cashback on all purchases for 3 months",
            true_response_rate=0.12,  # 12% response rate
            avg_revenue_per_conversion=2500
        ),
        CreditCardOffer(
            name="0% APR",
            description="0% APR on balance transfers for 6 months",
            true_response_rate=0.08,  # 8% response rate
            avg_revenue_per_conversion=4000
        ),
        CreditCardOffer(
            name="Double Points",
            description="Double reward points on dining and travel",
            true_response_rate=0.15,  # 15% response rate (SECOND BEST)
            avg_revenue_per_conversion=1800
        ),
        CreditCardOffer(
            name="₹500 Bonus",
            description="₹500 welcome bonus on first purchase",
            true_response_rate=0.18,  # 18% response rate (BEST!)
            avg_revenue_per_conversion=1500
        ),
        CreditCardOffer(
            name="Lounge Access",
            description="Free airport lounge access for 1 year",
            true_response_rate=0.06,  # 6% response rate (targets niche)
            avg_revenue_per_conversion=5000
        ),
    ]
    return offers


# ==============================================================================
# SECTION 2: UCB ALGORITHM IMPLEMENTATION
# ==============================================================================

class UCBBandit:
    """
    Upper Confidence Bound (UCB1) Algorithm for Multi-Armed Bandits
    
    The UCB formula balances:
    - Exploitation: Pick arms with high observed rewards
    - Exploration: Pick arms with high uncertainty (less tried)
    
    UCB = x̄ᵢ + c × √(ln(N) / nᵢ)
    
    Where:
    - x̄ᵢ = average reward from arm i
    - N = total number of pulls across all arms
    - nᵢ = number of times arm i was pulled
    - c = exploration parameter (default √2)
    """
    
    def __init__(self, n_arms: int, c: float = 1.41):
        """
        Initialize UCB algorithm.
        
        Args:
            n_arms: Number of arms (offers) to choose from
            c: Exploration parameter (higher = more exploration)
        """
        self.n_arms = n_arms
        self.c = c
        
        # Tracking statistics for each arm
        self.counts = np.zeros(n_arms)        # Times each arm was pulled
        self.values = np.zeros(n_arms)         # Estimated value (mean reward)
        self.total_counts = 0                  # Total pulls
        
        # History for analysis
        self.history = {
            'arm_pulled': [],
            'reward': [],
            'ucb_values': [],
            'cumulative_reward': [],
            'cumulative_regret': []
        }
    
    def calculate_ucb(self) -> np.ndarray:
        """
        Calculate UCB value for each arm.
        
        Returns:
            Array of UCB values for each arm
        """
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                # Unexplored arm gets infinite UCB (must try first)
                ucb_values[arm] = float('inf')
            else:
                # UCB = exploitation + exploration bonus
                exploitation = self.values[arm]
                exploration = self.c * np.sqrt(
                    np.log(self.total_counts) / self.counts[arm]
                )
                ucb_values[arm] = exploitation + exploration
        
        return ucb_values
    
    def select_arm(self) -> int:
        """
        Select which arm to pull based on UCB values.
        
        Returns:
            Index of selected arm
        """
        ucb_values = self.calculate_ucb()
        
        # If multiple arms have same UCB (e.g., all infinite), pick randomly
        max_ucb = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_ucb)[0]
        
        return np.random.choice(best_arms)
    
    def update(self, arm: int, reward: float):
        """
        Update statistics after pulling an arm.
        
        Args:
            arm: Which arm was pulled
            reward: Reward received
        """
        self.counts[arm] += 1
        self.total_counts += 1
        
        # Incremental mean update: new_mean = old_mean + (reward - old_mean) / n
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
    
    def run_trial(self, arm: int, reward: float, best_arm_value: float):
        """
        Run a single trial: select arm, get reward, update, record history.
        """
        # Record UCB values before selection
        ucb_vals = self.calculate_ucb()
        
        # Update with observed reward
        self.update(arm, reward)
        
        # Calculate cumulative reward
        cum_reward = sum(self.history['reward']) + reward if self.history['reward'] else reward
        
        # Calculate regret (difference from optimal)
        instant_regret = best_arm_value - reward
        cum_regret = (
            self.history['cumulative_regret'][-1] + instant_regret 
            if self.history['cumulative_regret'] else instant_regret
        )
        
        # Store history
        self.history['arm_pulled'].append(arm)
        self.history['reward'].append(reward)
        self.history['ucb_values'].append(ucb_vals.copy())
        self.history['cumulative_reward'].append(cum_reward)
        self.history['cumulative_regret'].append(cum_regret)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics for each arm."""
        return pd.DataFrame({
            'Arm': range(self.n_arms),
            'Times Pulled': self.counts.astype(int),
            'Estimated Value': self.values.round(4),
            'Pull Percentage': (self.counts / self.total_counts * 100).round(1)
        })


# ==============================================================================
# SECTION 3: BASELINE STRATEGIES FOR COMPARISON
# ==============================================================================

class RandomStrategy:
    """Random selection - no learning, pure exploration"""
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.history = {'reward': [], 'cumulative_regret': []}
    
    def select_arm(self) -> int:
        return np.random.randint(self.n_arms)
    
    def update(self, arm: int, reward: float):
        pass  # Random doesn't learn
    
    def run_trial(self, reward: float, best_arm_value: float):
        cum_reward = sum(self.history['reward']) + reward if self.history['reward'] else reward
        instant_regret = best_arm_value - reward
        cum_regret = (
            self.history['cumulative_regret'][-1] + instant_regret 
            if self.history['cumulative_regret'] else instant_regret
        )
        self.history['reward'].append(reward)
        self.history['cumulative_regret'].append(cum_regret)


class GreedyStrategy:
    """
    Epsilon-Greedy with epsilon=0 after initial exploration.
    Pure exploitation after trying each arm once.
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.history = {'reward': [], 'cumulative_regret': []}
    
    def select_arm(self) -> int:
        # Try each arm once first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        # Then always pick the best observed
        return np.argmax(self.values)
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
    
    def run_trial(self, reward: float, best_arm_value: float):
        cum_reward = sum(self.history['reward']) + reward if self.history['reward'] else reward
        instant_regret = best_arm_value - reward
        cum_regret = (
            self.history['cumulative_regret'][-1] + instant_regret 
            if self.history['cumulative_regret'] else instant_regret
        )
        self.history['reward'].append(reward)
        self.history['cumulative_regret'].append(cum_regret)


# ==============================================================================
# SECTION 4: SIMULATION ENGINE
# ==============================================================================

def run_simulation(
    offers: List[CreditCardOffer],
    n_customers: int = 1000,
    exploration_param: float = 1.41
) -> Dict:
    """
    Run complete simulation comparing UCB vs Random vs Greedy.
    
    Args:
        offers: List of credit card offers (arms)
        n_customers: Number of customers to target (trials)
        exploration_param: UCB exploration parameter
    
    Returns:
        Dictionary with results for each strategy
    """
    n_arms = len(offers)
    
    # Find the true best arm (oracle knowledge)
    true_response_rates = [o.true_response_rate for o in offers]
    best_arm_idx = np.argmax(true_response_rates)
    best_arm_value = true_response_rates[best_arm_idx]
    
    print("=" * 70)
    print("SIMULATION SETUP")
    print("=" * 70)
    print(f"\nNumber of offers (arms): {n_arms}")
    print(f"Number of customers (trials): {n_customers}")
    print(f"UCB exploration parameter (c): {exploration_param}")
    print(f"\n{'Offer':<20} {'Response Rate':<15} {'Revenue/Conv':<15}")
    print("-" * 50)
    for i, offer in enumerate(offers):
        best_marker = " ⭐ BEST" if i == best_arm_idx else ""
        print(f"{offer.name:<20} {offer.true_response_rate:<15.1%} ₹{offer.avg_revenue_per_conversion:<14}{best_marker}")
    
    # Initialize strategies
    ucb = UCBBandit(n_arms, c=exploration_param)
    random_strat = RandomStrategy(n_arms)
    greedy_strat = GreedyStrategy(n_arms)
    
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION...")
    print("=" * 70)
    
    # Run simulation
    for customer in range(n_customers):
        # UCB selection and update
        ucb_arm = ucb.select_arm()
        accepted, _ = offers[ucb_arm].simulate_customer_response()
        reward = 1 if accepted else 0  # Binary reward for response
        ucb.run_trial(ucb_arm, reward, best_arm_value)
        
        # Random selection
        random_arm = random_strat.select_arm()
        accepted_r, _ = offers[random_arm].simulate_customer_response()
        reward_r = 1 if accepted_r else 0
        random_strat.run_trial(reward_r, best_arm_value)
        
        # Greedy selection
        greedy_arm = greedy_strat.select_arm()
        accepted_g, _ = offers[greedy_arm].simulate_customer_response()
        reward_g = 1 if accepted_g else 0
        greedy_strat.update(greedy_arm, reward_g)
        greedy_strat.run_trial(reward_g, best_arm_value)
        
        # Progress indicator
        if (customer + 1) % 200 == 0:
            print(f"  Processed {customer + 1}/{n_customers} customers...")
    
    print("\n✅ Simulation complete!")
    
    return {
        'ucb': ucb,
        'random': random_strat,
        'greedy': greedy_strat,
        'offers': offers,
        'best_arm_idx': best_arm_idx,
        'best_arm_value': best_arm_value,
        'n_customers': n_customers
    }


# ==============================================================================
# SECTION 5: ANALYSIS AND VISUALIZATION
# ==============================================================================

def analyze_results(results: Dict):
    """Analyze and visualize simulation results."""
    
    ucb = results['ucb']
    random_strat = results['random']
    greedy_strat = results['greedy']
    offers = results['offers']
    best_arm_idx = results['best_arm_idx']
    n_customers = results['n_customers']
    
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. UCB Arm Selection Summary
    # -------------------------------------------------------------------------
    print("\n📊 UCB ARM SELECTION SUMMARY:")
    print("-" * 50)
    summary = ucb.get_summary()
    summary['Offer Name'] = [offers[i].name for i in range(len(offers))]
    summary['True Rate'] = [offers[i].true_response_rate for i in range(len(offers))]
    summary = summary[['Arm', 'Offer Name', 'Times Pulled', 'Pull Percentage', 
                       'Estimated Value', 'True Rate']]
    print(summary.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 2. Comparison of Total Conversions
    # -------------------------------------------------------------------------
    print("\n\n📈 TOTAL CONVERSIONS COMPARISON:")
    print("-" * 50)
    
    ucb_conversions = sum(ucb.history['reward'])
    random_conversions = sum(random_strat.history['reward'])
    greedy_conversions = sum(greedy_strat.history['reward'])
    oracle_expected = n_customers * results['best_arm_value']
    
    print(f"{'Strategy':<20} {'Conversions':<15} {'Rate':<10} {'vs Oracle':<15}")
    print("-" * 60)
    print(f"{'Oracle (theoretical)':<20} {oracle_expected:<15.0f} {results['best_arm_value']:<10.1%} {'—':<15}")
    print(f"{'UCB':<20} {ucb_conversions:<15.0f} {ucb_conversions/n_customers:<10.1%} {ucb_conversions/oracle_expected:<15.1%}")
    print(f"{'Greedy':<20} {greedy_conversions:<15.0f} {greedy_conversions/n_customers:<10.1%} {greedy_conversions/oracle_expected:<15.1%}")
    print(f"{'Random':<20} {random_conversions:<15.0f} {random_conversions/n_customers:<10.1%} {random_conversions/oracle_expected:<15.1%}")
    
    # -------------------------------------------------------------------------
    # 3. Final Regret Comparison
    # -------------------------------------------------------------------------
    print("\n\n📉 CUMULATIVE REGRET (Lower is Better):")
    print("-" * 50)
    
    ucb_regret = ucb.history['cumulative_regret'][-1]
    random_regret = random_strat.history['cumulative_regret'][-1]
    greedy_regret = greedy_strat.history['cumulative_regret'][-1]
    
    print(f"{'Strategy':<20} {'Final Regret':<15} {'Regret/Trial':<15}")
    print("-" * 50)
    print(f"{'UCB':<20} {ucb_regret:<15.2f} {ucb_regret/n_customers:<15.4f}")
    print(f"{'Greedy':<20} {greedy_regret:<15.2f} {greedy_regret/n_customers:<15.4f}")
    print(f"{'Random':<20} {random_regret:<15.2f} {random_regret/n_customers:<15.4f}")
    
    # -------------------------------------------------------------------------
    # 4. Business Impact
    # -------------------------------------------------------------------------
    print("\n\n💰 BUSINESS IMPACT (Revenue Estimation):")
    print("-" * 50)
    
    # Assume average revenue per conversion
    avg_revenue = 2000  # ₹2000 per customer who accepts
    
    ucb_revenue = ucb_conversions * avg_revenue
    random_revenue = random_conversions * avg_revenue
    greedy_revenue = greedy_conversions * avg_revenue
    oracle_revenue = oracle_expected * avg_revenue
    
    print(f"{'Strategy':<20} {'Est. Revenue':<20} {'vs Random Gain':<20}")
    print("-" * 60)
    print(f"{'Oracle':<20} ₹{oracle_revenue:,.0f}")
    print(f"{'UCB':<20} ₹{ucb_revenue:,.0f}{' ':>5} +₹{ucb_revenue - random_revenue:,.0f}")
    print(f"{'Greedy':<20} ₹{greedy_revenue:,.0f}{' ':>5} +₹{greedy_revenue - random_revenue:,.0f}")
    print(f"{'Random':<20} ₹{random_revenue:,.0f}{' ':>5} —")
    
    return {
        'ucb_conversions': ucb_conversions,
        'random_conversions': random_conversions,
        'greedy_conversions': greedy_conversions,
        'ucb_regret': ucb_regret,
        'random_regret': random_regret,
        'greedy_regret': greedy_regret
    }


def create_visualizations(results: Dict, save_path: str = None):
    """Create comprehensive visualizations."""
    
    ucb = results['ucb']
    random_strat = results['random']
    greedy_strat = results['greedy']
    offers = results['offers']
    best_arm_idx = results['best_arm_idx']
    n_customers = results['n_customers']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('UCB Algorithm - Banking Marketing Campaign Optimization', 
                 fontsize=14, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Plot 1: Cumulative Regret Comparison
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    trials = range(1, n_customers + 1)
    
    ax1.plot(trials, ucb.history['cumulative_regret'], 
             label='UCB', color='green', linewidth=2)
    ax1.plot(trials, greedy_strat.history['cumulative_regret'], 
             label='Greedy', color='orange', linewidth=2, linestyle='--')
    ax1.plot(trials, random_strat.history['cumulative_regret'], 
             label='Random', color='red', linewidth=2, linestyle=':')
    
    # Add theoretical O(√t) reference line
    sqrt_reference = 3 * np.sqrt(trials)
    ax1.plot(trials, sqrt_reference, label='O(√t) Reference', 
             color='gray', linewidth=1, linestyle='-.')
    
    ax1.set_xlabel('Number of Customers')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title('Cumulative Regret Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Arm Selection Distribution (UCB)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    offer_names = [o.name for o in offers]
    colors = ['#2ecc71' if i == best_arm_idx else '#3498db' for i in range(len(offers))]
    
    bars = ax2.bar(offer_names, ucb.counts, color=colors, edgecolor='black')
    ax2.set_xlabel('Offer')
    ax2.set_ylabel('Times Selected')
    ax2.set_title('UCB Arm Selection Distribution')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, count in zip(bars, ucb.counts):
        pct = count / n_customers * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # -------------------------------------------------------------------------
    # Plot 3: UCB Values Evolution Over Time
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    # Sample UCB values at intervals for cleaner plot
    sample_interval = max(1, n_customers // 100)
    sampled_trials = list(range(0, n_customers, sample_interval))
    
    for arm in range(len(offers)):
        ucb_over_time = [ucb.history['ucb_values'][t][arm] 
                        if ucb.history['ucb_values'][t][arm] != float('inf') else np.nan 
                        for t in sampled_trials]
        label = f"{offers[arm].name}" + (" ⭐" if arm == best_arm_idx else "")
        ax3.plot([t+1 for t in sampled_trials], ucb_over_time, label=label, linewidth=1.5)
    
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('UCB Value')
    ax3.set_title('UCB Values Evolution (Convergence to Best Arm)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Estimated vs True Response Rates
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    x = np.arange(len(offers))
    width = 0.35
    
    true_rates = [o.true_response_rate for o in offers]
    estimated_rates = ucb.values
    
    bars1 = ax4.bar(x - width/2, true_rates, width, label='True Rate', 
                    color='#3498db', edgecolor='black')
    bars2 = ax4.bar(x + width/2, estimated_rates, width, label='UCB Estimated', 
                    color='#2ecc71', edgecolor='black')
    
    ax4.set_xlabel('Offer')
    ax4.set_ylabel('Response Rate')
    ax4.set_title('True vs Estimated Response Rates')
    ax4.set_xticks(x)
    ax4.set_xticklabels([o.name for o in offers], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
    
    return fig


def print_ucb_step_by_step(results: Dict, n_steps: int = 10):
    """
    Print step-by-step UCB calculations for first few trials.
    Great for teaching/understanding the algorithm.
    """
    offers = results['offers']
    ucb = results['ucb']
    
    print("\n" + "=" * 70)
    print("STEP-BY-STEP UCB CALCULATIONS (First 10 trials)")
    print("=" * 70)
    print("\nFormula: UCB = x̄ᵢ + c × √(ln(N) / nᵢ)")
    print("Where: x̄ᵢ = avg reward, N = total trials, nᵢ = trials for arm i, c = 1.41")
    print("-" * 70)
    
    # Recreate first n_steps for demonstration
    demo_ucb = UCBBandit(len(offers))
    np.random.seed(42)  # Same seed as main simulation
    
    for step in range(min(n_steps, len(ucb.history['arm_pulled']))):
        print(f"\n📍 TRIAL {step + 1}:")
        
        # Calculate and show UCB values
        print(f"\n   {'Arm':<15} {'Avg (x̄)':<10} {'Pulls (n)':<10} {'UCB Calc':<30} {'UCB Value':<10}")
        print("   " + "-" * 75)
        
        ucb_values = demo_ucb.calculate_ucb()
        
        for arm in range(len(offers)):
            avg = demo_ucb.values[arm]
            n = int(demo_ucb.counts[arm])
            N = demo_ucb.total_counts
            
            if n == 0:
                calc_str = "∞ (unexplored)"
                ucb_val = "∞"
            else:
                exploration = 1.41 * np.sqrt(np.log(N) / n)
                calc_str = f"{avg:.3f} + 1.41×√(ln({N})/{n}) = {avg:.3f} + {exploration:.3f}"
                ucb_val = f"{ucb_values[arm]:.3f}"
            
            print(f"   {offers[arm].name:<15} {avg:<10.3f} {n:<10} {calc_str:<30} {ucb_val:<10}")
        
        # Show selection
        selected_arm = ucb.history['arm_pulled'][step]
        reward = ucb.history['reward'][step]
        
        print(f"\n   → Selected: {offers[selected_arm].name} (highest UCB)")
        print(f"   → Customer response: {'✅ ACCEPTED' if reward else '❌ DECLINED'}")
        
        # Update demo UCB
        demo_ucb.update(selected_arm, reward)


# ==============================================================================
# SECTION 6: MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "=" * 70)
    print("   🏦 UCB ALGORITHM - BANKING MARKETING OPTIMIZATION DEMO 🏦")
    print("=" * 70)
    
    # Create offers
    offers = create_bank_offers()
    
    # Run simulation
    results = run_simulation(
        offers=offers,
        n_customers=1000,
        exploration_param=1.41  # √2
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Show step-by-step for first few trials
    print_ucb_step_by_step(results, n_steps=8)
    
    # Create visualizations
    fig = create_visualizations(results, save_path='/Users/sandeepdiddi/Documents/Unext/RL-Learning-Jan2026/RL-Jan2026-Batch/10Feb2026/ucb_banking_results.png')
    
    # Final summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. UCB BALANCES EXPLORATION & EXPLOITATION
       - Initially tries all arms to gather information
       - Gradually shifts to exploiting the best arm
       - Uncertainty bonus ensures less-tried arms get chances
    
    2. REGRET ANALYSIS
       - UCB achieves O(√t) regret (sublinear) ✅
       - Random achieves O(t) regret (linear) ❌
       - Greedy can get stuck on suboptimal arm ⚠️
    
    3. BUSINESS VALUE
       - UCB found the best offer (₹500 Bonus) efficiently
       - Higher conversion rate = more revenue
       - Minimized wasted marketing spend during learning
    
    4. WHEN TO USE UCB
       - A/B testing with unknown best variant
       - Ad placement optimization
       - Product recommendation
       - Any scenario with explore/exploit trade-off
    """)
    
    print("=" * 70)
    print("Demo complete! Check 'ucb_banking_results.png' for visualizations.")
    print("=" * 70)
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()