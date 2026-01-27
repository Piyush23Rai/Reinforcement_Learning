"""
LLM Training Pipeline - Program 2: PPO, Challenges, and Integration
Complete end-to-end RLHF demonstration with pharmaceutical examples
"""

import numpy as np
from scipy.special import expit  # Logistic function
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("LLM TRAINING PIPELINE - PROGRAM 2")
print("Stages: PPO Optimization → Challenges → Complete Integration")
print("="*80)

# ============================================================================
# PART 1: PROXIMAL POLICY OPTIMIZATION (PPO)
# ============================================================================

class PPOOptimizer:
    """
    Proximal Policy Optimization: Optimize policy using trained reward model
    
    Mathematical: L_PPO = E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)] - β*KL(π||π_ref)
    """
    
    def __init__(self):
        self.queries = [
            "How to adjust for renal impairment?",
            "Warfarin + NSAIDs risk?",
            "Metformin in cirrhosis?",
            "Penicillin allergy + cephalosporin?",
            "CYP2D6 poor metabolizer + codeine?",
            "Lithium + kidney disease?",
            "Grapefruit + statin?",
            "Elderly on 10+ medications?"
        ]
        
        # Store SFT and PPO responses with rewards
        self.sft_responses = [
            "eGFR >60: normal. eGFR 30-59: reduce. eGFR <30: further reduce.",
            "NSAIDs can cause bleeding. Use acetaminophen instead.",
            "Metformin has renal concerns but may work.",
            "Cephalosporins might be okay but be careful.",
            "Codeine might not work well.",
            "Need to be careful with lithium dosing.",
            "Grapefruit might affect statin levels.",
            "Multiple medications can be risky."
        ]
        
        self.ppo_responses = [
            "eGFR >60: 100%. eGFR 30-59: 75%. eGFR <30: 50%. Monitor renal function weekly x4.",
            "CONTRAINDICATED. NSAIDs inhibit platelets + enhance warfarin → 2-3x bleeding. Use acetaminophen.",
            "CONTRAINDICATED. Severe lactic acidosis risk. Use insulin, GLP-1, SGLT2i.",
            "DEPENDS on type. Non-IgE: 3rd gen safe (1-3% cross). IgE: AVOID beta-lactams.",
            "INEFFECTIVE. Codeine requires CYP2D6→morphine. Use morphine/oxycodone directly.",
            "CRITICAL. Narrow window 0.6-1.2. Renal disease→toxicity. Monitor q1 week.",
            "YES-AVOID. CYP3A4 inhibitor→10-16x statin levels→myopathy. Switch statin.",
            "HIGH RISK. Deprescribe lowest-benefit drugs. Start with TCAs, benzos."
        ]
        
        self.sft_rewards = [3.8, 3.2, 2.1, 2.9, 3.4, 3.5, 3.6, 3.2]
        self.ppo_rewards = [5.3, 5.1, 5.2, 5.4, 5.2, 5.4, 5.3, 5.2]
        
        self.epsilon = 0.2  # PPO clipping
        self.beta = 0.5  # KL weight
        self.learning_rate = 0.001
        
        self.ppo_history = {
            "iteration": [],
            "loss": [],
            "reward": [],
            "kl_div": [],
            "clipped": []
        }
    
    def train_ppo(self, iterations=1000):
        """Train policy with PPO for multiple iterations"""
        print("\n" + "="*80)
        print("STAGE 4: POLICY OPTIMIZATION VIA PPO")
        print("="*80)
        print("\nPPO optimizes policy to maximize RM reward while staying close to SFT.")
        print("Mathematical: L = E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)] - β*KL\n")
        print(f"Training PPO for {iterations} iterations on 8 pharmaceutical queries...\n")
        
        for iteration in range(iterations):
            batch_loss = 0
            batch_reward = 0
            batch_kl = 0
            batch_clipped = 0
            
            # Mini-batch training
            for i in range(min(4, len(self.queries))):
                sft_logprob = np.log(0.8)  # Approximate SFT log probability
                ppo_logprob = np.log(0.85)  # Improved log probability
                
                # Probability ratio
                ratio = np.exp(ppo_logprob - sft_logprob)
                
                # Advantage (reward - baseline)
                baseline = np.mean(self.sft_rewards)
                advantage = self.ppo_rewards[i] - baseline
                
                # Clipped objective
                clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
                ppo_objective = min(ratio * advantage, clipped_ratio * advantage)
                
                # KL divergence approximation
                kl = sft_logprob - ppo_logprob
                
                # Total loss
                loss = -(ppo_objective - self.beta * kl)
                batch_loss += loss
                batch_reward += self.ppo_rewards[i]
                batch_kl += kl
                
                if clipped_ratio != ratio:
                    batch_clipped += 1
            
            batch_size = min(4, len(self.queries))
            avg_loss = batch_loss / batch_size
            avg_reward = batch_reward / batch_size
            avg_kl = batch_kl / batch_size
            clipped_frac = batch_clipped / batch_size
            
            self.ppo_history["iteration"].append(iteration + 1)
            self.ppo_history["loss"].append(avg_loss)
            self.ppo_history["reward"].append(avg_reward)
            self.ppo_history["kl_div"].append(avg_kl)
            self.ppo_history["clipped"].append(clipped_frac)
            
            # Dynamic KL adjustment
            if iteration % 250 == 0:
                if avg_kl > 0.01:
                    self.beta *= 1.05
                elif avg_kl < 0.001:
                    self.beta *= 0.95
            
            if (iteration + 1) % 250 == 0:
                print(f"Iter {iteration+1:4d}: Loss={avg_loss:.4f}, Reward={avg_reward:.2f}, "
                      f"KL={avg_kl:.4f}, Clipped={clipped_frac:.0%}, β={self.beta:.3f}")
    
    def display_improvements(self):
        """Display SFT vs PPO response improvements"""
        print("\n" + "="*80)
        print("PHARMACEUTICAL RESPONSE IMPROVEMENTS: SFT vs PPO")
        print("="*80)
        print("\nComparing 8 queries - baseline SFT vs PPO-optimized responses\n")
        
        improvements = [ppo - sft for ppo, sft in zip(self.ppo_rewards, self.sft_rewards)]
        
        for i in range(min(5, len(self.queries))):
            print(f"Query {i+1}: {self.queries[i]}")
            print(f"  SFT Reward:     {self.sft_rewards[i]:.1f}/10")
            print(f"  SFT Response:   {self.sft_responses[i][:60]}...")
            print(f"  PPO Reward:     {self.ppo_rewards[i]:.1f}/10")
            print(f"  PPO Response:   {self.ppo_responses[i][:60]}...")
            print(f"  Improvement:    +{improvements[i]:.1f} ({improvements[i]/self.sft_rewards[i]:.0%})\n")
        
        print(f"Average Improvement: +{np.mean(improvements):.2f} (+{np.mean(improvements)/np.mean(self.sft_rewards):.0%})")

# ============================================================================
# PART 2: RLHF CHALLENGES
# ============================================================================

class RLHFChallenges:
    """Identify and explain RLHF training challenges"""
    
    def __init__(self):
        self.challenges = [
            {
                "name": "Reward Hacking",
                "severity": "CRITICAL",
                "description": "Policy exploits RM defects rather than improving genuinely",
                "example": "If RM rewards length, model generates verbose repetitive text",
                "solution": "Human validation, ensemble RMs, strong KL regularization",
                "mitigation": 0.92
            },
            {
                "name": "Annotation Quality",
                "severity": "HIGH",
                "description": "Different annotators have different preferences",
                "example": "A prioritizes safety (eGFR cutoffs), B prioritizes flexibility (alternatives)",
                "solution": "Inter-rater reliability (Fleiss' kappa ≥0.6), multi-annotator consensus",
                "mitigation": 0.85
            },
            {
                "name": "Distribution Shift",
                "severity": "HIGH",
                "description": "RM trained on SFT examples, PPO produces different outputs",
                "example": "RM unreliable on novel PPO-generated responses",
                "solution": "Active learning: collect preferences on PPO outputs, retrain RM",
                "mitigation": 0.78
            },
            {
                "name": "Scalability",
                "severity": "MEDIUM",
                "description": "Expert time is expensive. 50k preferences = $500k cost",
                "example": "Pharmacist: 30 min per 10 preferences = 2500 hours",
                "solution": "Focus on high-value data, RM-guided sampling, crowdsource with validation",
                "mitigation": 0.80
            }
        ]
    
    def display_challenges(self):
        """Display all challenges with solutions"""
        print("\n" + "="*80)
        print("CHALLENGES IN RLHF TRAINING")
        print("="*80)
        print("\nMajor obstacles to RLHF and how to mitigate them:\n")
        
        severity_emoji = {"CRITICAL": "🔴🔴", "HIGH": "🔴", "MEDIUM": "🟡"}
        
        for i, challenge in enumerate(self.challenges, 1):
            print(f"{i}. {challenge['name']} {severity_emoji[challenge['severity']]}")
            print(f"   Severity: {challenge['severity']}")
            print(f"   Problem: {challenge['description']}")
            print(f"   Example: {challenge['example']}")
            print(f"   Solution: {challenge['solution']}")
            print(f"   Mitigation Effectiveness: {challenge['mitigation']:.0%}\n")

# ============================================================================
# PART 3: ALIGNMENT EVALUATION
# ============================================================================

class AlignmentEvaluation:
    """Evaluate LLM alignment on pharmaceutical domain"""
    
    def __init__(self):
        self.metrics = [
            {"name": "Medical Accuracy", "current": 0.91, "target": 0.95},
            {"name": "Safety (Contraindication Recall)", "current": 0.94, "target": 1.00},
            {"name": "Specificity (False Positives)", "current": 0.89, "target": 0.95},
            {"name": "Dose Calculation Accuracy", "current": 0.91, "target": 0.98},
            {"name": "Humility (Uncertainty Admission)", "current": 0.87, "target": 1.00},
        ]
    
    def display_scorecard(self):
        """Display alignment evaluation scorecard"""
        print("\n" + "="*80)
        print("ALIGNMENT EVALUATION SCORECARD - PHARMACEUTICAL AI")
        print("="*80)
        print("\nMetrics for evaluating LLM alignment quality:\n")
        
        for i, metric in enumerate(self.metrics, 1):
            current_pct = int(metric["current"] * 40)  # Bar length
            bar = "█" * current_pct + "░" * (40 - current_pct)
            gap = metric["target"] - metric["current"]
            status = "✓ PASS" if gap <= 0 else f"⚠ Need {gap:.0%} improvement"
            
            print(f"{i}. {metric['name']}")
            print(f"   Current: {metric['current']:.0%} | Target: {metric['target']:.0%} | {status}")
            print(f"   [{bar}] {metric['current']:.0%}\n")

# ============================================================================
# PART 4: COMPLETE INTEGRATION
# ============================================================================

class CompleteIntegration:
    """Show how all components fit together"""
    
    @staticmethod
    def display_full_pipeline():
        """Display complete pipeline comparison"""
        print("\n" + "="*80)
        print("COMPLETE LLM TRAINING PIPELINE: END-TO-END VIEW")
        print("="*80)
        
        print("\n┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")
        print("│ STAGE           │ INPUT           │ PROCESS         │ OUTPUT QUALITY  │")
        print("├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
        print("│ 1. Pretraining  │ Web text        │ Next-token pred │ Broad, rambling │")
        print("│    (Billion+)   │ (billions)      │ Loss=-logP(w)   │ + No alignment  │")
        print("├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
        print("│ 2. SFT          │ Expert demos    │ Behavioral      │ Good, structured│")
        print("│    (1-2 weeks)  │ (10k-100k)      │ Loss=-logπ(y)   │ - Capped by exp │")
        print("├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
        print("│ 3. RM Train     │ Preferences     │ Bradley-Terry   │ Excellent,      │")
        print("│    (days)       │ (20k)           │ Loss=-logσ(r)   │ specific        │")
        print("├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
        print("│ 4. PPO          │ RM + SFT        │ Clipped PPO     │ Expert-level +  │")
        print("│    (weeks)      │ (same data)     │ L-βKL           │ aligned         │")
        print("└─────────────────┴─────────────────┴─────────────────┴─────────────────┘")
        
        print("\nWHY ALL FOUR STAGES ARE NECESSARY:\n")
        print("Stage 1 (Pretraining):")
        print("  ✓ Provides linguistic foundation")
        print("  ✗ No safety, no alignment → Need Stage 2\n")
        
        print("Stage 2 (SFT):")
        print("  ✓ Teaches instruction-following & safe responses")
        print("  ✗ Capped by expert quality, no preference ranking → Need Stage 3\n")
        
        print("Stage 3 (RM Training):")
        print("  ✓ Learns to rank responses on human values")
        print("  ✗ RM is just a classifier, doesn't optimize policy → Need Stage 4\n")
        
        print("Stage 4 (PPO):")
        print("  ✓ Actively optimizes policy toward RM reward")
        print("  ✓ Maintains stability via KL constraint")
        print("  ✓ Achieves expert-level, aligned performance\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Stage 4: PPO
    ppo = PPOOptimizer()
    ppo.train_ppo(iterations=1000)
    ppo.display_improvements()
    
    # Challenges
    challenges = RLHFChallenges()
    challenges.display_challenges()
    
    # Alignment
    alignment = AlignmentEvaluation()
    alignment.display_scorecard()
    
    # Complete Integration
    integration = CompleteIntegration()
    integration.display_full_pipeline()
    
    # Visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss over PPO iterations
    axes[0, 0].plot(ppo.ppo_history["iteration"], ppo.ppo_history["loss"], 'r-', linewidth=2)
    axes[0, 0].set_xlabel("PPO Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("PPO: Loss Progression")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward over PPO iterations
    axes[0, 1].plot(ppo.ppo_history["iteration"], ppo.ppo_history["reward"], 'g-', linewidth=2)
    axes[0, 1].axhline(y=np.mean(ppo.sft_rewards), color='b', linestyle='--', label='SFT Avg')
    axes[0, 1].set_xlabel("PPO Iteration")
    axes[0, 1].set_ylabel("Average Reward")
    axes[0, 1].set_title("PPO: Reward Maximization")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[1, 0].plot(ppo.ppo_history["iteration"], ppo.ppo_history["kl_div"], 'b-', linewidth=2)
    axes[1, 0].set_xlabel("PPO Iteration")
    axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].set_title("PPO: KL from SFT Model (Stability Control)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Response quality comparison
    queries_short = [q[:15] + "..." for q in ppo.queries[:8]]
    x = np.arange(len(ppo.sft_rewards))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, ppo.sft_rewards, width, label='SFT', color='#2E75B6', alpha=0.8)
    axes[1, 1].bar(x + width/2, ppo.ppo_rewards, width, label='PPO', color='#70AD47', alpha=0.8)
    axes[1, 1].set_xlabel("Query")
    axes[1, 1].set_ylabel("Reward Score (/10)")
    axes[1, 1].set_title("Response Quality: SFT vs PPO-Optimized")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(queries_short, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylim([0, 6])
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    # Save to a local outputs directory relative to this script. Create it if missing.
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "llm_program_2_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_path}\n")
    
    # Final Summary
    print("="*80)
    print("PROGRAM 2 SUMMARY")
    print("="*80)
    
    improvements = [ppo - sft for ppo, sft in zip(ppo.ppo_rewards, ppo.sft_rewards)]
    
    print(f"\n✓ PPO TRAINING:")
    print(f"  - Iterations: 1,000")
    print(f"  - Final Loss: {ppo.ppo_history['loss'][-1]:.4f}")
    print(f"  - Final Average Reward: {ppo.ppo_history['reward'][-1]:.2f}/10")
    print(f"  - Improvement over SFT: +{np.mean(improvements):.2f} (+{np.mean(improvements)/np.mean(ppo.sft_rewards):.0%})\n")
    
    print(f"✓ CHALLENGES IDENTIFIED:")
    print(f"  - Reward Hacking (Critical, 92% mitigation)")
    print(f"  - Annotation Quality (High, 85% mitigation)")
    print(f"  - Distribution Shift (High, 78% mitigation)")
    print(f"  - Scalability (Medium, 80% mitigation)\n")
    
    print(f"✓ ALIGNMENT METRICS:")
    print(f"  - Medical Accuracy: 91% (target 95%)")
    print(f"  - Safety Recall: 94% (target 100%)")
    print(f"  - Dose Accuracy: 91% (target 98%)")
    print(f"  - Gap to target: ~7% overall\n")
    
    print(f"✓ COMPLETE PIPELINE:")
    print(f"  - All 4 stages necessary")
    print(f"  - Each stage solves previous limitations")
    print(f"  - Together: Expert-level + aligned pharmaceutical guidance\n")
    
    print("="*80)
    print("✓ PROGRAMS 1 & 2 COMPLETE - FULL LLM PIPELINE DEMONSTRATED")
    print("="*80)

if __name__ == "__main__":
    main()
