import random
import math

# =========================
# Toy "Real World" MDP: Coupon Policy
# =========================
# States:
#   0 = ENGAGED (good)
#   1 = NEUTRAL
#   2 = AT_RISK (might churn)
#   3 = COUPON_ADDICT (buys only with coupons; lower margins)
#
# Actions:
#   0 = NO_COUPON
#   1 = COUPON
#
# Key idea:
# - COUPON gives higher immediate reward, but pushes user toward COUPON_ADDICT state.
# - NO_COUPON may give less immediate reward, but preserves long-term profitability.

S = [0, 1, 2, 3]
A = [0, 1]
ENGAGED, NEUTRAL, AT_RISK, ADDICT = 0, 1, 2, 3
NO_COUPON, COUPON = 0, 1

def step_env(state, action, rng):
    """
    Transition + reward.
    Returns: next_state, reward
    """
    # Immediate rewards (business outcome)
    # Coupon boosts short-term conversion, but margins are worse for addicts.
    if action == COUPON:
        if state == ADDICT:
            reward = 1.0   # they buy, but margin is low
        else:
            reward = 3.0   # quick conversion pop
    else:
        # No coupon: smaller short-term reward
        if state == ENGAGED:
            reward = 2.0
        elif state == NEUTRAL:
            reward = 1.0
        elif state == AT_RISK:
            reward = 0.2
        else:  # ADDICT with no coupon
            reward = 0.0   # they usually don't buy without discount

    # Transition dynamics (how customer state changes)
    r = rng.random()

    if action == COUPON:
        # Coupon increases chance of becoming coupon-addicted over time
        if state == ENGAGED:
            next_state = ENGAGED if r < 0.60 else (NEUTRAL if r < 0.85 else ADDICT)
        elif state == NEUTRAL:
            next_state = ENGAGED if r < 0.50 else (NEUTRAL if r < 0.75 else ADDICT)
        elif state == AT_RISK:
            next_state = NEUTRAL if r < 0.55 else (ENGAGED if r < 0.70 else ADDICT)
        else:  # ADDICT
            next_state = ADDICT if r < 0.90 else NEUTRAL
    else:
        # No coupon can drift customer toward at-risk sometimes, but avoids addiction
        if state == ENGAGED:
            next_state = ENGAGED if r < 0.75 else NEUTRAL
        elif state == NEUTRAL:
            next_state = ENGAGED if r < 0.20 else (NEUTRAL if r < 0.70 else AT_RISK)
        elif state == AT_RISK:
            next_state = NEUTRAL if r < 0.25 else AT_RISK
        else:  # ADDICT
            next_state = NEUTRAL if r < 0.30 else (AT_RISK if r < 0.60 else ADDICT)

    return next_state, reward

# =========================
# Myopic (Reactive) Policy: gamma -> 0
# =========================
# Chooses action that maximizes *immediate* reward based on current state
def myopic_policy(state):
    # compute immediate reward for both actions (without considering transitions)
    # (simple "reactive rule")
    immediate_no = 2.0 if state == ENGAGED else (1.0 if state == NEUTRAL else (0.2 if state == AT_RISK else 0.0))
    immediate_yes = 1.0 if state == ADDICT else 3.0
    return COUPON if immediate_yes > immediate_no else NO_COUPON

# =========================
# Planning Policy: Value Iteration (uses gamma)
# =========================
# We'll estimate transition + reward model by Monte Carlo sampling (small and classroom-friendly)
def estimate_model(num_samples=3000, seed=0):
    rng = random.Random(seed)
    P = {s: {a: {s2: 0 for s2 in S} for a in A} for s in S}
    R = {s: {a: 0.0 for a in A} for s in S}

    for s in S:
        for a in A:
            total_r = 0.0
            counts = {s2: 0 for s2 in S}
            for _ in range(num_samples):
                s2, r = step_env(s, a, rng)
                counts[s2] += 1
                total_r += r
            for s2 in S:
                P[s][a][s2] = counts[s2] / num_samples
            R[s][a] = total_r / num_samples

    return P, R

def value_iteration(P, R, gamma=0.9, iters=200):
    V = {s: 0.0 for s in S}
    for _ in range(iters):
        V_new = {}
        for s in S:
            q_values = []
            for a in A:
                expected = R[s][a] + gamma * sum(P[s][a][s2] * V[s2] for s2 in S)
                q_values.append(expected)
            V_new[s] = max(q_values)
        V = V_new

    # Derive deterministic optimal policy pi(s) = argmax_a Q(s,a)
    pi = {}
    for s in S:
        best_a, best_q = None, -1e9
        for a in A:
            q = R[s][a] + gamma * sum(P[s][a][s2] * V[s2] for s2 in S)
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a
    return V, pi

# =========================
# Simulation + Regret
# =========================
def run_policy(policy_fn, T=200, seed=1, start_state=NEUTRAL):
    rng = random.Random(seed)
    s = start_state
    total = 0.0
    for _ in range(T):
        a = policy_fn(s)
        s, r = step_env(s, a, rng)
        total += r
    return total

def main():
    # 1) Build a "planning" policy using estimated model + value iteration
    P, R = estimate_model(num_samples=4000, seed=42)
    gamma = 0.90
    V, planned_pi = value_iteration(P, R, gamma=gamma, iters=250)

    def planned_policy(s):
        return planned_pi[s]

    # 2) Compare performance and regret over increasing horizons
    print("State meanings: 0=ENGAGED, 1=NEUTRAL, 2=AT_RISK, 3=ADDICT")
    print("Action meanings: 0=NO_COUPON, 1=COUPON")
    print("\nPlanned policy (gamma=0.90):", planned_pi)
    print("Myopic policy (gamma->0): uses only immediate reward\n")

    for T in [50, 100, 200, 400, 800]:
        # average across a few runs to reduce randomness
        trials = 30
        opt_sum = 0.0
        myo_sum = 0.0
        for k in range(trials):
            opt_sum += run_policy(planned_policy, T=T, seed=100 + k, start_state=NEUTRAL)
            myo_sum += run_policy(myopic_policy,  T=T, seed=100 + k, start_state=NEUTRAL)

        opt_avg = opt_sum / trials
        myo_avg = myo_sum / trials

        regret = opt_avg - myo_avg
        print(f"T={T:4d} | Planned(avg)={opt_avg:7.2f} | Myopic(avg)={myo_avg:7.2f} | Regret(T)={regret:6.2f} | sqrt(T)={math.sqrt(T):5.2f}")

    print("\nInterpretation:")
    print("- Myopic reacts to the current state and grabs short-term reward (coupons).")
    print("- Planned policy uses gamma=0.90 to value future profit and avoids pushing users into ADDICT too often.")
    print("- Regret(T) is the gap between them over T steps. sqrt(T) is printed to relate to Ω(√T) discussions.")

if __name__ == "__main__":
    main()
