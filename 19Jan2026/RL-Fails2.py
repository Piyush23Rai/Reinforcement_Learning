#!/usr/bin/env python3
"""
Tabular RL Scaling: Math + Inference + Simulation (Class Demo)

Goal:
Show why tabular Q-learning doesn't scale even if you "can store" the table:
- You also need to *learn* each Q(s,a) from experience.
- With huge state spaces, the agent rarely revisits the same state.
"""

from __future__ import annotations
import math
import random
from collections import Counter

# ----------------------------
# 1) Core setup (as in the lesson)
# ----------------------------
S = 1_000_000   # number of states
A = 20          # number of actions
N = 200_000     # number of interactions / time steps in our simulation (change to 1_000_000 if you like)

# ----------------------------
# 2) Helper functions
# ----------------------------
def bytes_to_human(n_bytes: int) -> str:
    gib = n_bytes / (1024**3)
    mib = n_bytes / (1024**2)
    kib = n_bytes / 1024
    if gib >= 1:
        return f"{gib:.2f} GiB"
    if mib >= 1:
        return f"{mib:.2f} MiB"
    if kib >= 1:
        return f"{kib:.2f} KiB"
    return f"{n_bytes} bytes"

def simulate_uniform_state_visits(num_states: int, steps: int, seed: int = 7) -> Counter:
    """
    Simplified model:
    Each time step we land in a random state uniformly among S states.
    This models a worst-case "very diverse" state space: repeats are rare.
    """
    random.seed(seed)
    visits = Counter()
    for _ in range(steps):
        s = random.randrange(num_states)
        visits[s] += 1
    return visits

# ----------------------------
# 3) The math (what to teach)
# ----------------------------
def expected_state_visits(N: int, S: int) -> float:
    """
    Expected number of visits to a particular state under uniform sampling:
      E[visits(state i)] = N * (1/S)
    """
    return N / S

def prob_state_unvisited(N: int, S: int) -> float:
    """
    Probability a specific state is NEVER visited in N steps:
      P(unvisited) = (1 - 1/S)^N  ≈ exp(-N/S)
    """
    return (1.0 - 1.0 / S) ** N

def expected_unique_states(N: int, S: int) -> float:
    """
    Expected number of unique states visited in N steps (uniform):
      E[unique] = S * (1 - (1 - 1/S)^N)  ≈ S * (1 - exp(-N/S))
    """
    return S * (1.0 - (1.0 - 1.0 / S) ** N)

# ----------------------------
# 4) Print teaching narrative + inference
# ----------------------------
def main() -> None:
    print("=== Tabular RL Scaling (Math + Inference) ===\n")

    # Table size math
    q_values = S * A
    print("1) Table size math")
    print(f"   If |S| = {S:,} and |A| = {A}:")
    print(f"   Q-table entries = |S|×|A| = {S:,}×{A} = {q_values:,} numbers")

    mem_f32 = q_values * 4
    mem_f64 = q_values * 8
    print("   Memory (just raw array, not Python dict overhead):")
    print(f"   - float32: {bytes_to_human(mem_f32)}")
    print(f"   - float64: {bytes_to_human(mem_f64)}")

    print("\n   Inference (important):")
    print("   Storing is not the biggest problem. LEARNING is.")
    print("   Each Q(s,a) needs repeated experience to become reliable.\n")

    # Learning sparsity math
    print("2) Learning sparsity math (why Q-values stay noisy)")
    ev = expected_state_visits(N, S)
    punv = prob_state_unvisited(N, S)
    euniq = expected_unique_states(N, S)

    print(f"   We run N = {N:,} interactions.")
    print("   Under a 'many distinct states' assumption (uniform over states):")
    print(r"   Expected visits to a specific state i:")
    print(r"     E[visits(i)] = N/S")
    print(f"     = {N:,}/{S:,} = {ev:.6f} visits per state (<< 1 visit!)")

    print("\n   Probability a specific state is never visited:")
    print(r"     P(unvisited) = (1 - 1/S)^N ≈ exp(-N/S)")
    print(f"     (1 - 1/{S:,})^{N:,} ≈ exp(-{N:,}/{S:,}) = exp(-{N/S:.6f})")
    print(f"     ≈ {math.exp(-N/S):.6f}  (very high)")

    print("\n   Expected number of unique states visited:")
    print(r"     E[unique] = S * (1 - (1 - 1/S)^N) ≈ S*(1 - exp(-N/S))")
    approx = S * (1 - math.exp(-N / S))
    print(f"     ≈ {S:,} * (1 - exp(-{N/S:.6f})) ≈ {approx:,.0f} unique states")

    print("\n   Inference")
    print("   - Most states are visited 0 times.")
    print("   - Many visited states are visited only 1 time.")
    print("   - With so few repeats, Q(s,a) cannot converge → values stay noisy.")
    print("   - That’s why we need function approximation (neural nets) to generalize.\n")

    # Action-level sparsity (even worse)
    print("3) Even worse: (state, action) pairs")
    print(r"   For one state, experience is split across actions.")
    print(r"   Expected visits to a specific (s,a) under random actions:")
    print(r"     E[visits(s,a)] = N * (1/S) * (1/A) = N/(S*A)")
    esa = N / (S * A)
    print(f"     = {N:,}/({S:,}×{A}) = {esa:.8f} visits per (s,a)")
    print("   Inference:")
    print("   - Learning 20 million Q-values is impossible unless you have massive repeated data.\n")

    # Simulation
    print("4) Simulation (to make it concrete)")
    print(f"   Simulating {N:,} steps with uniform state visits across {S:,} states...\n")
    visits = simulate_uniform_state_visits(S, N, seed=7)
    unique_states = len(visits)
    freq_of_freq = Counter(visits.values())
    once = freq_of_freq.get(1, 0)
    twice = freq_of_freq.get(2, 0)
    max_vis = max(visits.values()) if visits else 0

    print(f"   Unique states actually visited: {unique_states:,}")
    print(f"   Visited exactly once: {once:,} ({once/unique_states*100:.1f}% of visited states)")
    print(f"   Visited exactly twice: {twice:,} ({twice/unique_states*100:.1f}% of visited states)")
    print(f"   Max visits to any state: {max_vis}")

    print("\n   Final inference (connect to lesson):")
    print("   Tabular RL = memorization per exact state.")
    print("   Real world = huge/continuous states → almost no exact repeats.")
    print("   Therefore: we use neural networks to generalize across similar states.\n")

if __name__ == "__main__":
    main()
