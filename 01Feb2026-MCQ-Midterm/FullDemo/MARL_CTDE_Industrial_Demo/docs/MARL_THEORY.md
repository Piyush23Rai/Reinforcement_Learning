# Multi-Agent Reinforcement Learning (MARL) Theory & Practice

## Table of Contents
1. [Single-Agent RL Recap](#single-agent-rl-recap)
2. [Introduction to MARL](#introduction-to-marl)
3. [Challenge: Non-Stationarity](#challenge-non-stationarity)
4. [CTDE Architecture](#ctde-architecture)
5. [Algorithms](#algorithms)

---

## Single-Agent RL Recap

In standard single-agent RL:
- **State (s)**: Agent's observation of environment
- **Action (a)**: Agent's decision
- **Reward (r)**: Feedback signal
- **Policy (π)**: Mapping from state to action
- **Value Function V(s)**: Expected future reward from state s
- **Q-Function Q(s,a)**: Expected future reward from state s taking action a

### Key Algorithms (Single-Agent)
- **Q-Learning**: Off-policy value-based learning
- **Policy Gradient (PG)**: Direct policy optimization  
- **Actor-Critic**: Combines policy gradient with value baseline
- **DDPG**: Deterministic policy gradient for continuous actions

---

## Introduction to MARL

### What Changes with Multiple Agents?

In MARL, we have n agents operating in the same environment:

```
Environment
    ↓
[Observation 1] → [Agent 1] → [Action 1]
[Observation 2] → [Agent 2] → [Action 2]
[Observation 3] → [Agent 3] → [Action 3]
    ↓
[Reward 1, Reward 2, Reward 3]
```

**Key Difference**: The reward of one agent depends on actions of ALL agents!

- **Agent 1's reward** depends on Agent 1's action AND Agent 2's action AND Agent 3's action
- This violates the independent, identically distributed (i.i.d.) assumption
- Environment becomes **non-stationary** from each agent's perspective

### Three Types of MARL Problems

#### 1. **Cooperative** (Team Problem)
- All agents share the same goal
- Agents want to work together
- Example: Warehouse coordination (retail demo)
- Solution: Value function decomposition, centralized critic

#### 2. **Competitive** (Game Theory)
- Agents compete for resources
- Zero-sum or opposing rewards
- Example: Game playing (chess, poker)
- Solution: Game-theoretic algorithms, Nash equilibrium

#### 3. **Mixed** (General-Sum)
- Agents have both cooperative and competitive objectives
- Example: Autonomous driving (don't crash, minimize time)
- Solution: Combination of cooperative and competitive learning

---

## Challenge: Non-Stationarity

### The Problem

In single-agent RL:
- Environment is **stationary** (rules don't change)
- You can estimate V(s) or Q(s,a) reliably
- Use Temporal Difference (TD) to learn: `V(s) = E[r + γV(s')]`

In MARL:
- **Other agents are also learning** (changing their policies)
- When Agent 1 learns a policy, Agent 2 is simultaneously changing its policy
- From Agent 1's perspective, the environment **keeps changing**
- Traditional RL guarantees break down!

### Illustration

```
Agent 1 perspective at time t:
  Q(s, a₁) = E[r + γQ(s', a₁)]
  
But at time t+1:
  Environment changed because Agent 2 updated its policy!
  Q(s, a₁) ≠ E[r + γQ(s', a₁)]  ← Prediction was wrong!
```

### Solutions

1. **Independent Learners** (Naive)
   - Each agent learns as if others are stationary
   - Simple but fails in practice (non-convergence)
   - Poor sample efficiency

2. **Centralized Learning**
   - Train one giant network with all observations and actions
   - Problem: Doesn't scale (exponential growth)
   - Loses modularity and interpretability

3. **CTDE (Our Approach)** ⭐
   - Centralized training: Use full information during learning
   - Decentralized execution: Use only local info during deployment
   - Best of both worlds!

---

## CTDE Architecture

### Core Concept

```
┌─────────────────────────────────────────────┐
│           TRAINING PHASE                    │
│  (Offline, with full observability)         │
│                                             │
│  Shared Replay Buffer: All agents' data    │
│       ↓                                     │
│  Centralized Critic: Sees all obs & acts   │
│  Decentralized Actors: Local policies      │
│       ↓                                     │
│  Compute Gradients: Global context         │
└──────────────────┬──────────────────────────┘
                   │
                   │ Deploy trained policies
                   ↓
┌─────────────────────────────────────────────┐
│        EXECUTION PHASE                      │
│  (Online, deployed system)                  │
│                                             │
│  Agent 1: local_obs → π₁ → action          │
│  Agent 2: local_obs → π₂ → action          │
│  Agent N: local_obs → πₙ → action          │
│                                             │
│  NO communication, NO central control       │
└─────────────────────────────────────────────┘
```

### Why CTDE Works

1. **Training Context**: During training, we know what all agents observe and do
   - Can compute how actions affect outcomes
   - Can estimate value of joint actions
   - Handles non-stationarity through shared gradient context

2. **Scalable Execution**: Each agent acts independently
   - O(n) execution complexity (n agents, each O(1) decision)
   - vs. Centralized execution which is O(exponential)

3. **Implicit Coordination**: 
   - Agents don't communicate but coordinate through training
   - Each agent learns optimal policy given others' learned behaviors
   - Emerges automatically through shared value feedback

---

## Algorithms

### Algorithm 1: MADDPG (Multi-Agent DDPG)

**Extends**: DDPG to multi-agent setting

**Architecture**:
```
Actor Networks:       πᵢ(sᵢ → aᵢ)        [local, decentralized]
Critic Networks:      Qᵢ([s₁...sₙ, a₁...aₙ] → Q)  [centralized]
```

**Training**:
```
1. Sample experience: (s, a, r, s')
2. Get next actions: a'ᵢ = π'ᵢ(sᵢ) for all agents
3. Compute target: y = r + γQ(s', [a'₁...a'ₙ])
4. Update Critic: L_Q = (Q(s, a) - y)²
5. Update Actor:  L_π = -E[Q(s, [π₁(s₁)...πₙ(sₙ)])]
```

**Why it works for MARL**:
- Critic sees all agents' actions for training
- Actor uses only local observation for execution
- Deterministic policy enables stable gradient computation

**Best for**: Continuous action spaces, competitive/mixed environments

---

### Algorithm 2: MAPPO (Multi-Agent PPO)

**Extends**: PPO to multi-agent setting

**Architecture**:
```
Actor Networks:    πᵢ(sᵢ → distribution)  [local, decentralized]
Critic Network:    V([s₁...sₙ] → value)   [centralized]
```

**Training**:
```
1. Collect episode with all agents
2. Compute returns: Gₜ = Σ γᵏrₜ₊ₖ
3. Compute advantages: Aₜ = Gₜ - V(sₜ)
4. Update actors:   L_π = -min(rₜ·Aₜ, clip(rₜ,1-ε,1+ε)·Aₜ)
5. Update critic:   L_V = (V(s) - G)²
```

Where:
- rₜ = probability ratio (new policy / old policy)
- ε = clipping parameter (typically 0.2)

**Why it works for MARL**:
- Centralized value reduces variance
- Policy clipping prevents instability
- Works well with shared reward structures

**Best for**: Cooperative problems, policy gradient preference

---

### Algorithm 3: QMIX (Q-value Mixing)

**Key Insight**: For cooperative MARL, decompose joint Q into individual Q-functions!

**Architecture**:
```
Local Q-Networks:   Qᵢ(sᵢ, aᵢ → Q-value)      [decentralized]
Mixing Network:     Ψ([Q₁...Qₙ], state → Q_total)  [centralized]
```

**Constraint (Monotonicity)**:
```
∂Q_total / ∂Qᵢ ≥ 0  for all i

Ensures: argmax(Q_total) = sum(argmax(Qᵢ))

This means optimal joint action = combination of individual optimal actions!
```

**Training**:
```
1. Get local Q-values: Qᵢ(sᵢ, aᵢ)
2. Mix: Q_total = Ψ([Q₁...Qₙ], s)
3. Compute target: y = r + γmax(Q_total(s', a'))
4. Update: L = (Q_total - y)²
```

**Why it works**:
- Scalable: Only trains n Q-networks
- Implicit coordination through monotonic mixing
- No communication needed during execution

**Best for**: Cooperative problems, discrete actions, scalability

---

## Summary Comparison

| Aspect | MADDPG | MAPPO | QMIX |
|--------|--------|-------|------|
| Action Space | Continuous | Continuous/Discrete | Discrete only |
| Environment | Cooperative, Mixed | Cooperative | Cooperative |
| Scalability | Moderate | Moderate | High |
| Sample Efficiency | Good | Excellent | Good |
| Implementation | Medium | Medium | Complex (monotonicity) |
| Coordination | Via critic | Via baseline | Via mixing |
| Communication | None | None | None |

---

## Key Insights

### 1. **Centralized Critic Solves Non-Stationarity**
The centralized critic sees how other agents' actions affect outcomes, providing stable gradient signals even as other agents learn.

### 2. **Implicit Coordination is Powerful**
Agents don't need explicit communication or coordination mechanism. They learn to coordinate through shared training.

### 3. **CTDE is Practical**
Decentralized execution makes these algorithms practical for real systems where you can't have central control.

### 4. **Algorithm Choice Matters**
- MADDPG: Good for continuous control with strong gradients
- MAPPO: Good for cooperative settings with variance reduction
- QMIX: Best for scalability in highly cooperative settings

---

## Further Reading

1. **MADDPG Paper**: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
   - https://arxiv.org/abs/1706.02275

2. **MAPPO Paper**: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
   - https://arxiv.org/abs/2103.01955

3. **QMIX Paper**: QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL
   - https://arxiv.org/abs/1905.06175

4. **General MARL Survey**: Multi-Agent Reinforcement Learning: A Selective Overview
   - https://arxiv.org/abs/2106.01895

---

**Remember**: The key to successful MARL is choosing the right algorithm for your problem structure and leveraging CTDE to handle the non-stationarity while maintaining scalability!
