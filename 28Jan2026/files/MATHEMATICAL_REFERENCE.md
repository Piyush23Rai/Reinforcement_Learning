# Multi-Agent RL: Mathematical Reference & Deep Dives
## Study Guide for Advanced Learners

---

## PART 1: FOUNDATIONAL MATHEMATICS

### Stochastic Games: Formal Definition

A stochastic game is a tuple **G = (N, S, A, P, R, γ):**

- **N** = {1, 2, ..., n} = set of agents
- **S** = state space (finite or continuous)
- **A = A₁ × A₂ × ... × Aₙ** = joint action space
- **P(s'|s, a₁, ..., aₙ)** = Markovian transition dynamics
- **R = {R₁, R₂, ..., Rₙ}** where Rᵢ(s, a₁, ..., aₙ) = reward for agent i
- **γ ∈ [0,1)** = discount factor

**Execution Sequence:**
```
At time t:
  State s_t ∈ S
  All agents simultaneously choose: a_i,t ∈ A_i
  Joint action: a_t = (a_1,t, ..., a_n,t)
  Rewards: r_i,t = R_i(s_t, a_1,t, ..., a_n,t)
  Transition: s_{t+1} ~ P(·|s_t, a_t)
```

**Key Difference from MDP:**
- **MDP:** Only one agent acts. Environment stationary from observer's perspective.
- **Stochastic Game:** All agents act simultaneously. Non-stationary from every agent's view.

### Cumulative Returns and Nash Equilibrium

Agent i's cumulative return:
```
V_i(π) = E[Σ_{t=0}^∞ γ^t R_i(s_t, a_1,t, ..., a_n,t) | initial policy π]
```

Notice: V_i depends on all policies, not just πᵢ.

**Definition (Pure Strategy Nash Equilibrium):**

A joint policy π* = (π₁*, π₂*, ..., πₙ*) is NE if for all agents i:
```
V_i(π_i*, π_{-i}*) ≥ V_i(π_i, π_{-i}*)  for all alternative πᵢ
```

**Interpretation:** No agent improves by unilaterally deviating from their NE strategy.

---

## PART 2: MADDPG MATHEMATICS

### Single-Agent DDPG Baseline

Deterministic policy: a = μ(s) (outputs single action, not distribution)

**Policy Gradient via Q-Function:**
```
∇_θ J = 𝔼[∇_θ μ(s) · ∇_a Q(s, a)|_{a=μ(s)}]
```

**Why Deterministic?** Direct chain rule allows gradient propagation through Q. More sample-efficient than stochastic.

### MADDPG Extension to Multi-Agent

Each agent i has:
- Deterministic actor: μᵢ(oᵢ) → action ∈ Aᵢ
- Centralized critic: Qᵢ^cen(s, a₁, ..., aₙ) → Q-value for agent i

**Policy Gradient for Agent i:**
```
∇_{θ_i} J_i = 𝔼[∇_{θ_i} μ_i(o_i) · ∇_{a_i} Q_i^cen(s, a_1, ..., a_n)|_{a_j=μ_j(o_j)}]
```

**Interpretation:** Actor μᵢ gets direction to move from critic's evaluation of joint action.

### MADDPG Algorithm (Pseudocode)

```
Initialize:
  For each agent i:
    Actor μᵢ(oᵢ; θμᵢ), Target μᵢ⁻
    Critic Qᵢ(s, a; φᵢ), Target Qᵢ⁻
  Replay buffer B

For episode = 1 to N:
  
  s₀ ← environment.reset()
  
  For t = 0 to T:
    For each agent i:
      aᵢ,ₜ = μᵢ(oᵢ,ₜ; θμᵢ) + ε,  ε ~ 𝒩(0, σ²)
    
    Execute joint action → (r₁,ₜ, ..., rₙ,ₜ), sₜ₊₁
    Store (s_t, a_t, r_t, s_{t+1}) in B
  
  For K training iterations:
    
    For each agent i:
      Sample mini-batch M from B
      
      -- CRITIC UPDATE --
      For each (s, a, r, s') in M:
        y_i = r_i + γ Q_i⁻(s', μ₁⁻(o₁'), ..., μₙ⁻(oₙ'))
        L_i^crit = (Q_i(s, a; φᵢ) - y_i)²
      
      φᵢ ← φᵢ - α_crit ∇_{φᵢ} L_i^crit
      
      -- ACTOR UPDATE --
      ∇_{θμᵢ} J_i = 𝔼[∇_{θμᵢ} μᵢ(oᵢ) · ∇_{aᵢ} Q_i(s, a)]
      θμᵢ ← θμᵢ + α_actor ∇_{θμᵢ} J_i
      
      -- TARGET UPDATES --
      θμᵢ⁻ ← τ θμᵢ + (1-τ) θμᵢ⁻
      φᵢ⁻ ← τ φᵢ + (1-τ) φᵢ⁻
```

**Hyperparameters:** α_crit, α_actor, τ (typically 0.001), σ (exploration noise), γ (0.99)

### Worked Example: 3-Store Pricing

**Setup:**
- Stores set prices pᵢ ∈ [1, 10]
- Demand: Dᵢ = 100 - 2pᵢ + 0.5(pⱼ + pₖ)
- Profit: rᵢ = pᵢ · Dᵢ

**Training Example:**

```
State: s = [D_1, D_2, D_3, inv_1, inv_2, inv_3]
       (recent demands + inventory)

Actions: a = [p_1, p_2, p_3] (prices)
         μ_1(o_1) outputs p_1
         μ_2(o_2) outputs p_2  
         μ_3(o_3) outputs p_3

Execution:
  Store 1 sees o_1 = [D_1 history, inv_1]
  Outputs p_1 ≈ 5.2 via μ_1(o_1)
  
  Store 2 sees o_2 = [D_2 history, inv_2]
  Outputs p_2 ≈ 4.8 via μ_2(o_2)
  
  Store 3 outputs p_3 ≈ 6.1

Demand computation:
  D_1 = 100 - 2(5.2) + 0.5(4.8 + 6.1) = 95.3
  D_2 = 100 - 2(4.8) + 0.5(5.2 + 6.1) = 99.2
  D_3 = 100 - 2(6.1) + 0.5(5.2 + 4.8) = 91.0

Profits:
  r_1 = 5.2 × 95.3 ≈ 496
  r_2 = 4.8 × 99.2 ≈ 477
  r_3 = 6.1 × 91.0 ≈ 555

Critic Evaluation:
  Q_1(s, [5.2, 4.8, 6.1]) = estimated value for store 1's position
  Q_2(...) = estimated value for store 2
  Q_3(...) = estimated value for store 3

Target:
  y_1 = 496 + 0.99 × Q_1⁻(s', p_1', p_2', p_3')
  
Actor Update:
  ∇_{θμ_1} μ_1(o_1) gets direction from ∇_{a_1} Q_1(s, a)
  If critic says "increase p_1", actor increases μ_1
  If critic says "decrease p_1", actor decreases μ_1
```

---

## PART 3: MAPPO MATHEMATICS

### PPO Review (Single-Agent)

**Importance Sampling Ratio:**
```
r_t(θ) = π(a_t|s_t; θ_new) / π(a_t|s_t; θ_old)
```

**Clipped Surrogate Loss:**
```
L^CLIP(θ) = 𝔼[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

**Intuition:** If ratio r_t gets extreme, clipping prevents overshooting. Keeps policy update bounded.

### MAPPO Extension

Directly apply PPO to each agent:

**For Agent i:**
```
L^CLIP_i(θᵢ) = 𝔼[min(r_{i,t}(θᵢ)Â_{i,t}, clip(r_{i,t}(θᵢ), 1-ε, 1+ε)Â_{i,t})]

where:
  r_{i,t}(θᵢ) = π_i(a_{i,t}|o_{i,t}; θᵢ) / π_i(a_{i,t}|o_{i,t}; θᵢ_old)
  Â_{i,t} = R_t - V^cen(s_t)
```

### MAPPO Algorithm (Pseudocode)

```
Initialize:
  For each agent i:
    Policy π_i(a|o; θᵢ)
  Value function V^cen(s; φ)
  Trajectory buffer T

For episode = 1 to N:
  
  s₀ ← environment.reset()
  
  -- COLLECTION (On-policy) --
  For t = 0 to T:
    For each agent i:
      a_{i,t} ~ π_i(·|o_{i,t}; θᵢ)
      log_prob_{i,t} = log π_i(a_{i,t}|o_{i,t})
    
    Execute joint action → rewards, s'
    Store (o, a, r, s, log_prob, V(s))

  -- ADVANTAGE COMPUTATION --
  For t = T down to 0:
    V_t = V^cen(s_t)
    Return: G_t = Σ_{t'≥t} γ^(t'-t) r_{t'}
    Advantage: Â_{i,t} = G_t - V_t
  
  -- TRAINING (K epochs) --
  For epoch = 1 to K:
    
    For mini-batch M in trajectory:
      
      For each agent i:
        
        -- Actor Update (PPO) --
        r_{i,t} = π_i(a_{i,t}|o_{i,t}) / π_i_old(a_{i,t}|o_{i,t})
        L_i = -min(r_{i,t} Â_{i,t}, clip(r_{i,t}, 1-ε, 1+ε) Â_{i,t})
        θᵢ ← θᵢ - α_actor ∇_{θᵢ} L_i
      
      -- Value Update (Supervised) --
      L_V = (V^cen(s) - G)²
      φ ← φ - α_value ∇_φ L_V
```

### Why Stochastic Policies in MAPPO?

MAPPO uses πᵢ(aᵢ|oᵢ), not deterministic μᵢ(oᵢ).

**Advantages:**
- Natural exploration: Policy entropy σᵢ learned by network
- Avoids deterministic mode collapse (e.g., all stores setting same price)
- More stable in non-stationary settings
- Better convergence properties empirically

**Trade-off:** Lower sample efficiency (on-policy; don't reuse old data)

### Worked Example: Warehouse Inventory

```
State: s = [inventory at all warehouses, demand history]

Observations per warehouse i:
  o_i = [inventory_i, demand_history_i, seasonality]

Policy (Gaussian):
  π_i(a_i|o_i) = 𝒩(μ_i(o_i), σ_i²)
  
  Example:
    o_1 = [inventory=50, recent_demand=40, winter]
    μ_1(o_1) outputs mean restock = 35
    σ_1 = 8 (std dev)
    Sample action: a_1 ~ 𝒩(35, 8²) = 42 (restock 42 units)

Reward:
  r_i = items_sold - restock_cost × a_i - penalty × max(0, demand - inventory)
  
  Example:
    With demand=45, inventory=50, restock=42:
    items_sold = min(45, 50+42) = 45
    cost = 5 × 42 = 210
    penalty = 0
    r_1 = 45 - 210 - 0 = -165

Central Value Function:
  V^cen(s) = expected cumulative reward from state s
  V^cen(s=current) = 500 (optimistic estimate)

Advantage:
  G_t = sum of rewards from t onward = -165 + ... = 200
  Â_1,t = G_t - V(s) = 200 - 500 = -300
  
Interpretation: Worse than expected. Warehouse 1's action (restock 42) was suboptimal.

PPO Update:
  Old policy outputted a_1 ~ 𝒩(35, 8²), got a_1=42
  New policy outputs μ'_1(o_1) = 32 (lower mean)
  
  r_t = π_new(42|...) / π_old(42|...) ≈ 0.8 (slightly less likely under new)
  
  L = -min(0.8 × (-300), clip(0.8, 0.8, 1.2) × (-300))
    = -min(-240, -240) = 240 (loss; will decrease policy weight)
  
  Policy update: reduce probability of high restock values.
```

---

## PART 4: QMIX MATHEMATICS

### Value Function Factorization

**Goal:** Decompose joint value function using individual values:
```
Q_total(s, a_1, ..., a_n) = f(Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_n(o_n, a_n) | s)
```

**Benefit:** If each agent independently maximizes its Qᵢ, joint action maximizes Q_total.

### Monotonicity Constraint (Critical)

```
∂Q_total/∂Q_i ≥ 0  for all agents i
```

**Theorem:** If Q_total is monotonic in all Qᵢ, then:
```
argmax_{a_1,...,a_n} Q_total(s, a) = (argmax_{a_1} Q_1(o_1, a_1),
                                      argmax_{a_2} Q_2(o_2, a_2),
                                      ...
                                      argmax_{a_n} Q_n(o_n, a_n))
```

**Proof Sketch:** If Qᵢ ≤ Qᵢ* and Q_total is increasing in Qᵢ, then Q_total ≤ Q_total* via monotonicity. Thus greedy choices on Qᵢ lead to greedy on Q_total.

### QMIX Mixing Network Architecture

```
Individual Q-Networks:              Mixing Network:
─────────────────────────           ──────────────────
o_i → Neural Network → Q_i(a_i)     Input: [Q_1, Q_2, ..., Q_n], state s
                                     
                                     Layer 1:
                                       w1_raw = Neural Network(s)
                                       w1 = abs(w1_raw)  ← ensures w ≥ 0
                                       b1 = Neural Network(s)
                                       hidden = ReLU(w1 ⊙ [Q_1, Q_2, ...] + b1)
                                     
                                     Layer 2:
                                       w2 = abs(Neural Network(s))
                                       b2 = Neural Network(s)
                                       Q_total = w2 ⊙ hidden + b2

Target:  y = r + γ Q_total'
```

**Key Trick:** `abs()` on weights enforces w ≥ 0. Combined with ReLU, ensures ∂Q_total/∂Qᵢ ≥ 0.

### QMIX Algorithm

```
Initialize:
  For each agent i: Q_i(o_i, a_i; ψ_i), Target Q_i⁻
  Mixing network M(Q_1,...,Q_n,s; ξ), Target M⁻
  Replay buffer B

For episode = 1 to N:
  
  s₀ ← environment.reset()
  
  For t = 0 to T:
    For each agent i:
      a_{i,t} = ε-greedy on Q_i(o_{i,t}, ·; ψ_i)
    
    Execute joint action → r_t, s_{t+1}
    Store (s_t, a_t, r_t, s_{t+1}) in B

  For K training iterations:
    
    Sample mini-batch M from B
    
    -- COMPUTE Q_TOTAL --
    For each (s, a, r, s') in M:
      
      Forward:
        Q_i = Q_i(o_i, a_i; ψ_i) for each i
        Q_total = M([Q_1, ..., Q_n], s; ξ)
      
      Target:
        Q_i' = Q_i⁻(o_i', argmax_{a'_i} Q_i(o_i', a'_i); ψ_i⁻)
        Q_total' = M⁻([Q_1', ..., Q_n'], s'; ξ⁻)
        y = r + γ Q_total'
    
    -- LOSS (All agents + mixing jointly) --
    L = (Q_total - y)²
    
    -- UPDATE --
    For each i: ψ_i ← ψ_i - α ∇_{ψᵢ} L
    ξ ← ξ - α ∇_ξ L
    
    -- TARGET UPDATES --
    ψ_i⁻ ← τ ψ_i + (1-τ) ψ_i⁻
    ξ⁻ ← τ ξ + (1-τ) ξ⁻
```

### Worked Example: Store Fulfillment

```
Setup: 10 stores, order arrives at location X.
       One store must fulfill (constraint: Σ aᵢ = 1).

Observations per store i:
  o_i = [inventory_i, distance_to_X, current_load_i]

Individual Q-Values:
  Q_i(o_i, aᵢ=1) = -cost(i, X) - penalty(inventory_i)
  Q_i(o_i, aᵢ=0) = 0
  
  Example costs:
    Store A (close): cost = 2 km, Q_A(1) = -2
    Store B (medium): cost = 5 km, Q_B(1) = -5
    Store C (far): cost = 15 km, Q_C(1) = -15
    Store D (close, low inv): cost = 3 km + 10 penalty, Q_D(1) = -13

Mixing Network learns:
  Q_total = -min(|Q_1|, |Q_2|, |Q_3|, |Q_4|)
  (approximately; picks best store)

Decentralized Execution:
  Store A: Q_A(1) = -2 (best)
  Store B: Q_B(1) = -5
  Store C: Q_C(1) = -15
  Store D: Q_D(1) = -13
  
  Greedy: Store A maximizes (least negative = best)
  
Monotonicity Guarantee:
  If mixing network is monotonic in Q_i:
  argmax_i Q_i(o_i, 1) = argmax Q_total
  
  Store A's greedy choice = globally optimal!
  No coordination needed; decentralized exec works.
```

---

## PART 5: CONVERGENCE & THEORY

### MADDPG Convergence (Informal)

**Theorem (Lowe et al. 2017):**
Under assumptions:
1. Sufficient exploration (all states visited)
2. Critic function approximation bounded
3. Other agents' policies quasi-static (slow change)
4. γ < 1

→ MADDPG converges to local Nash Equilibrium of stochastic game.

**Critical Issues:**
- Assumption 3 violated (agents change simultaneously)
- Local NE may be exponentially many
- No guarantee on which NE emerges
- Empirically can oscillate or diverge

### MAPPO Convergence (Empirical)

- Empirically converges to stationary policy in cooperative settings
- Stronger convergence than independent learners (non-stationary)
- Weaker theory than MADDPG
- Convergence depends on problem structure

**Recommendation:** Use MAPPO for empirical stability, not for convergence theory.

### QMIX Convergence (Formal)

**Theorem (Rashid et al. 2020):**
If agents use independent Q-learning on their Qᵢ and mixing network learns monotonic decomposition with sufficient exploration:
→ QMIX converges to optimal cooperative policy.

**Key Advantage:** QMIX provides formal guarantees for cooperative settings, unlike MADDPG/MAPPO.

---

## PART 6: EQUATION SUMMARY

### Quick Reference

**Stochastic Game Bellman (Implicit, for Nash):**
```
For NE: V_i*(s) = 𝔼_{a~π*}[R_i(s,a) + γ V_i*(s')]
```

**MADDPG:**
```
Critic:  Q_i^cen(s, a) → predict value for agent i
Actor:   π_i(a|o) = δ(a - μ_i(o))  (deterministic)
Update:  ∇_θᵢ J_i = 𝔼[∇_{a_i} Q_i ∇_{θᵢ} μ_i]
Target:  y_i = r_i + γ Q_i⁻(s', a')
```

**MAPPO:**
```
Policy:    π_i(a|o) = stochastic (e.g., Gaussian)
Value:     V^cen(s) = baseline for advantage
Advantage: Â_i = G - V^cen(s)  (return minus baseline)
Loss:      L = -min(r·Â, clip(r, 1±ε)·Â)
Update:    PPO clipped policy gradient
```

**QMIX:**
```
Individual Q: Q_i(o_i, a_i) → value for agent i's action
Mixing:       Q_total = M([Q_1, ..., Q_n], s)  (monotonic)
Target:       y = r + γ Q_total'(s', a'_i)
Constraint:   ∂Q_total/∂Q_i ≥ 0  (monotonicity)
Execution:    a_i* = argmax Q_i(o_i, a_i)  independently
```

---

## PART 7: DEBUGGING & COMMON ISSUES

### Issue 1: Training Loss Increases

**Symptom:** Loss increases instead of decreasing.
**Cause:** Non-stationary environment. Critic trained to evaluate old policies; now facing new policies.
**Fix:** 
- MADDPG: Increase replay buffer size, reduce learning rates, smaller τ
- MAPPO: Check advantage computation; ensure returns are computed correctly

### Issue 2: Agents Don't Improve

**Symptom:** Rewards plateau early.
**Cause:** Sparse rewards (monthly feedback) or poor credit assignment.
**Fix:**
- Design intermediate rewards (daily/weekly signals)
- Use reward shaping
- In MAPPO, check entropy; ensure exploration not collapsed

### Issue 3: Policy Oscillates

**Symptom:** Agents alternate between actions; never settle.
**Cause:** Competitive dynamics or deterministic mode collapse.
**Fix:**
- MADDPG: Use MAPPO instead (stochasticity helps)
- MAPPO: Check PPO clipping; ensure ε large enough
- Add constraints to action space (e.g., min price floors)

### Issue 4: Critic Overfits

**Symptom:** Good training loss, poor execution.
**Cause:** Critic saw full state; actor only sees partial observation.
**Fix:**
- Design observations to be sufficient (include relevant features)
- During training, sometimes mask information to match execution-time visibility
- Monitor train vs. test performance separately

---

## PART 8: RESEARCH CONNECTIONS

**Non-Stationarity & Convergence:**
- Key open problem in MARL
- Connection to game theory (perfect vs. imperfect info games)
- Potential solution: explicit modeling of opponent learning

**Credit Assignment:**
- Connection to interpretability (which agent helped?)
- Potential solution: counterfactual explanations ("what if i acted differently?")

**Scaling:**
- Mean-field games approximate many agents with average behavior
- Graph neural networks encode local interactions

**Safety:**
- Constrained RL: formally enforce constraints
- Constitutional AI: encode values as inviolable rules

---

**End of Mathematical Reference**

Use alongside HTML module and Instructor Guide for comprehensive technical understanding.
