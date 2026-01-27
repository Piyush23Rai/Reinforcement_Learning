# Multi-Agent RL Fundamentals: Instructor Guide
## 2-Hour Teaching Module

**Course Duration:** 120 minutes  
**Target:** Students with single-agent RL background  
**Focus:** Domain-agnostic MARL foundations with retail applications

---

## SECTION TIMING & FACILITATION

### SECTION 1: SA → MA (12 min)

**Content:** Single-agent limitations, stochastic games, Nash Equilibrium

**Facilitation Tips:**
- Opening: "Single-agent assumes stationary environment. Retail has 5 stores all learning simultaneously. What breaks?"
- Emphasize: **non-stationarity** is the core difference
- Use concrete numbers: 5 stores × 10 actions = 100K joint actions (vs. 10 per agent)
- **Discussion:** Could one store set high price while others set low? (Answer: Maybe, but not stable NE)

**Common Errors:** Conflating "multiple agents" with "multi-objective." Clarify: multiple agents = non-stationary environment.

---

### SECTION 2: CTDE Paradigm (15 min)

**Content:** Centralized training, decentralized execution, information gap

**Facilitation Tips:**
- Core insight: Train with full data, execute independently
- Draw two-phase diagram clearly
- **Key point:** Critic sees everything (state s), actor only sees observation (oᵢ)
- Information gap challenge: How can actor replicate critic's guidance without full info?
- **Discussion:** If new store opens with no data, can you deploy immediately? (Answer: Likely no, unless observations are sufficient)

**Common Errors:** "Decentralized execution" = agents communicate. Clarify: No communication; independent decisions using trained policy.

---

### SECTION 3: MADDPG (25 min)

**Content:** Deterministic policies, off-policy learning, continuous control

**Facilitation Tips:**
- DDPG review: Deterministic μ(o) allows direct gradient through Q-function
- Extension to multi-agent: Centralized Q^cen(s, a₁,...,aₙ) with individual actors μᵢ(oᵢ)
- 3-store pricing example: Walk through with numbers
  - Example: store A prices 4.2, demand Dₐ = 96, profit = 403
  - Critic evaluates: "Is 4.2 the right price?" 
  - Actor updates to try better prices
- **Challenges:** Non-stationarity (other agents change), credit assignment (sparse rewards), competitive collapse (price wars)
- **Discussion:** Can MADDPG escape bad equilibrium (low prices) via exploration? (Answer: Difficult; deterministic + small noise limits exploration)

**Common Errors:**
- Assuming agents coordinate → They don't; learn implicitly
- Convergence guaranteed → It's not; theory assumes quasi-static others

---

### SECTION 4: MAPPO (25 min)

**Content:** Stochastic policies, PPO clipping, on-policy learning, stability

**Facilitation Tips:**
- PPO review: Clipping prevents policy ratio from going extreme. Stabilizes learning.
- Multi-agent: Same PPO but with centralized advantage Âᵢ = return - V^cen(s)
- Stochastic policies: πᵢ(aᵢ|oᵢ) = 𝒩(μᵢ(oᵢ), σᵢ²)
  - Natural exploration (σᵢ learned by network)
  - Avoids deterministic cycles (price wars, oscillations)
  - More stable in non-stationary settings
- Warehouse inventory case: On-policy learning suits quick episodes
- MAPPO vs MADDPG comparison:
  - MADDPG: deterministic, off-policy, efficient, mode collapse risk
  - MAPPO: stochastic, on-policy, stable, lower efficiency
  - When to pick: MAPPO for stability, MADDPG for sample efficiency
- **Discussion:** Can MAPPO's stochasticity help warehouse A discover sharing is beneficial? (Answer: Yes; exploration via σᵢ learns both actions, advantage signals guide toward sharing)

**Common Errors:** On-policy is always worse. Clarify: depends on problem; on-policy more robust to non-stationarity.

---

### SECTION 5: QMIX (20 min)

**Content:** Value factorization, monotonicity constraint, decentralized execution guarantee

**Facilitation Tips:**
- Factorization problem: Decompose Q_total into combination of individual Qᵢ's
- QMIX key insight: Monotonic mixing ensures greedy execution = global optimum
  - If ∂Q_total/∂Qᵢ ≥ 0, then argmax_aᵢ Qᵢ(oᵢ, aᵢ) → argmax Q_total
  - Powerful in fully cooperative settings
- Implementation: torch.abs() on weights ensures monotonicity
- Fulfillment case: 10 stores, one fulfills each order
  - Cost = distance + inventory penalty
  - Individual Qᵢ ≈ -cost if store i fulfills
  - Mixing network picks lowest cost store (highest Q)
  - Monotonicity guarantees this is globally optimal
- When QMIX works: Fully cooperative (shared reward), discrete actions, clear local preferences
- When it fails: Competitive goals, dense communication needed, 1000s+ agents
- **Discussion:** Why does monotonicity matter? Non-monotonic mixing breaks credit assignment. Agent i with high Qᵢ but low Q_total gets blamed; won't learn to take that action.

**Common Errors:**
- Monotonicity = linearity → No; just non-decreasing (can be nonlinear)
- QMIX solves coordination automatically → No; provides value guidance; still need enforcement (e.g., one agent rule)

---

### SECTION 6: Comparison (12 min)

**Content:** Algorithm selection, retail scenarios, red flags

**Facilitation Tips:**
- Decision tree approach:
  - Fully cooperative? YES → QMIX. NO → Continue.
  - Continuous or discrete? CONTINUOUS → Have offline data? YES → MADDPG. NO → MAPPO.
  - Scale/Stability? 100+ agents → QMIX. Need stability → MAPPO. Need efficiency → MADDPG.
- Retail scenarios:
  - Pricing (continuous, competitive, offline data) → MADDPG. Risk: price wars; add constraints.
  - Inventory (continuous, cooperative, online simulation) → MAPPO. Stable against oscillations.
  - Fulfillment (discrete, cooperative, 10 agents) → QMIX. Natural factorization.
  - Scheduling (discrete, mixed, 20 managers) → MAPPO. Scales better than MADDPG.
- Red flags: Extreme asymmetry → CTDE breaks. Adversarial → Wrong paradigm. 1000s agents → Centralized critics bottleneck.
- **Discussion:** Use all three simultaneously? (Answer: Yes, but coordination challenges. Each subsystem optimizes locally; may not coordinate globally. Needs unified objective or communication protocol.)

---

### SECTION 7: Research Frontiers (11 min)

**Content:** 6 open problems, validation approaches

**Facilitation Tips:**

1. **Non-Stationarity & Convergence:** Other agents' changing policies → target distribution shifts. No general theory. Research: explicitly model learning dynamics.

2. **Credit Assignment:** Sparse rewards (quarterly profit) → hard to know who helped. Research: counterfactual reasoning, reward shaping.

3. **Scaling:** Centralized critics' input grows with n agents. Research: mean-field games, hierarchical MARL, graph neural networks.

4. **Mixed Competitive/Cooperative:** Real retail is both. QMIX for cooperation, but MADDPG/MAPPO give no guarantees. Research: multi-task learning, mechanism design.

5. **Safety & Robustness:** Emergent behaviors harm business (price wars, bullwhip, demand destruction). Research: constrained MARL, interpretability, human oversight.

6. **Communication:** When should agents communicate? What? How prevent collusion? Open questions.

- **Final Discussion:** (5 min) "Deploying 50-store system. Which problem tackle first? Why? How validate?"
  - Expected answers vary; guide toward: (1) start with simulation safety (constraints), (2) test on subset, (3) monitor emergent behaviors, (4) iterate with human feedback.

---

## ASSESSMENT & RUBRICS

### Participation Rubric

| Level | Criteria | Score |
|-------|----------|-------|
| Novice | Restates question or single concept | 1 |
| Intermediate | Connects two concepts or partial reasoning | 2 |
| Proficient | Synthesizes multiple ideas with logic | 3 |
| Advanced | Novel connections, identifies trade-offs | 4 |

### Post-Class Essay (Optional)

**Prompt:** "You're deploying MARL to a 50-store chain (pricing, inventory, staffing). Which of the 6 open problems would you address first? Why? How would you validate success?"

**Grading:**
- Problem selection & justification (3 pts)
- Understanding of problem & solution approaches (4 pts)
- Validation methodology (2 pts)
- Clarity & organization (1 pt)
- **Total:** 10 pts

---

## PREPARATION CHECKLIST

### Before Class (30 min)
- [ ] Test HTML module on projector/display
- [ ] Review all discussion prompts
- [ ] Prepare whiteboard examples
- [ ] Set timing alerts for each section

### During Class
- [ ] Use consistent retail examples throughout
- [ ] Actually PAUSE for discussion; don't lecture past discussion prompts
- [ ] Watch engagement; ask follow-ups if flat
- [ ] Note student misconceptions for clarification

### After Class (15 min)
- [ ] Share HTML module link
- [ ] Post essay prompt with due date
- [ ] Offer office hours for questions

---

## FREQUENTLY ASKED QUESTIONS

**Q: Why centralized training if we learn the critic anyway?**
A: Training centralization helps learn better policies using all data. Execution decentralization ensures deployability. CTDE bridges this.

**Q: Why not just have agents communicate?**
A: Communication option, but has costs (latency, bandwidth, failures, potential collusion). CTDE avoids via pre-training.

**Q: What if two agents' Qᵢ conflict in QMIX?**
A: QMIX requires full cooperation. Conflicts suggest wrong algorithm; use MADDPG/MAPPO instead (handle mixed settings).

**Q: Is Nash Equilibrium always good?**
A: No! Classic prisoner's dilemma: mutual defection is NE but cooperation is better. Retail: low prices is NE but high prices better for all.

**Q: How validate multi-agent policies work?**
A: (1) Extensive simulation testing, (2) Monte Carlo over initial conditions, (3) Monitor for emergent harms (price wars, oscillations), (4) Deploy to subset first, (5) Monitor live performance continuously.

**Q: Can you transfer a trained MADDPG policy to a new domain?**
A: Partially. Policy πᵢ(oᵢ) might transfer if observations oᵢ are similar. But MADDPG actors are trained on specific opponent behaviors; new domain = new opponents = different policy needed.

---

## KEY THEMES TO REINFORCE

1. **Non-Stationarity is Hard:** Single-agent RL theory assumes stationary env. Multi-agent breaks this. No easy fix.

2. **CTDE is Powerful:** Train with information you have, execute with info you don't. Elegant solution to deployment problem.

3. **Algorithm Choice Matters:** Context-dependent. No one-size-fits-all. Understand problem structure first.

4. **Convergence is Empirical:** No guarantees. Test extensively. Monitor for pathologies.

5. **Safety is Non-Negotiable:** Multi-agent systems can exhibit unwanted emergent behaviors. Design constraints, oversight, interpretability.

6. **Research Opportunities Abound:** MARL is young. Many open problems. Today's students could solve them tomorrow.

---

**End of Instructor Guide**

Reference alongside HTML module (interactive navigation) and Mathematical Reference (deep dives) for comprehensive course delivery.
