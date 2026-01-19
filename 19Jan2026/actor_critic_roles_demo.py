#!/usr/bin/env python3
"""
Actor–Critic Roles (Who Does What) — Teaching Demo

Actor (policy network):
  - converts state -> action probabilities π(a|s)
  - does NOT say how good the state is, only "what to do"

Critic (value network):
  - predicts V(s) = expected total future reward from state s
  - helps reduce noise by giving a baseline

Key teaching line:
  "The actor learns what to do.
   The critic learns how good the situation is,
   so the actor doesn’t overreact to random noise."

This script prints:
- logits and softmax probabilities (actor)
- V(s) estimate (critic)
- advantage = reward - V(s)
- a simple actor-critic update step
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# 1) Softmax demo: logits=[2,1] -> softmax ~ [0.73, 0.27]
# ----------------------------
def softmax_np(logits):
    x = np.array(logits, dtype=np.float64)
    x = x - x.max()  # for stability
    e = np.exp(x)
    return e / e.sum()

print("=== Mini Example: logits -> softmax ===")
demo_logits = [2.0, 1.0]
demo_probs = softmax_np(demo_logits)
print(f"logits = {demo_logits}  => softmax probs ≈ [{demo_probs[0]:.2f}, {demo_probs[1]:.2f}]")
print("Plain English: action0 is chosen more often because it has higher probability.\n")


# ----------------------------
# 2) Toy state: a customer situation vector (d=4)
#    s = [recency_norm, freq_norm, spend_norm, sensitivity]
# ----------------------------
def make_state(recency_days: int, freq_30d: int, spend: float, sensitivity: float) -> torch.Tensor:
    recency_norm = min(recency_days, 365) / 365.0
    freq_norm = min(freq_30d, 30) / 30.0
    spend_norm = max(0.0, min(spend, 1000.0)) / 1000.0
    sens = max(0.0, min(sensitivity, 1.0))
    return torch.tensor([recency_norm, freq_norm, spend_norm, sens], dtype=torch.float32)


ACTIONS = ["SEND_COUPON", "NO_COUPON"]


# ----------------------------
# 3) Actor network: state -> logits -> softmax probabilities
# ----------------------------
class Actor(nn.Module):
    def __init__(self, d_in: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        logits = self.net(s)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


# ----------------------------
# 4) Critic network: state -> V(s)
# ----------------------------
class Critic(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)  # scalar


# ----------------------------
# 5) Toy reward function (profit)
#    (simple: higher for price-sensitive when coupon sent)
# ----------------------------
def simulate_reward(state: torch.Tensor, action: int, rng: random.Random) -> float:
    """
    This is a tiny "business" simulator:
    - If coupon sent, purchase prob increases with sensitivity.
    - Profit = margin - coupon_cost if purchase else 0.
    """
    sensitivity = float(state[3].item())
    spend_norm = float(state[2].item())
    spend = spend_norm * 1000.0

    margin = 0.20 * spend  # 20% margin
    coupon_cost = 50.0     # fixed coupon cost for demo

    base_p = 0.20
    uplift = 0.50 * sensitivity if action == 0 else 0.0  # coupon increases buy prob
    p_buy = min(0.95, base_p + uplift)

    buy = (rng.random() < p_buy)
    if not buy:
        return 0.0
    return float(margin - (coupon_cost if action == 0 else 0.0))


# ----------------------------
# 6) Actor–Critic training loop (tiny)
#    Update rules:
#      Advantage A = r - V(s)
#      Actor loss  = -logπ(a|s) * A
#      Critic loss = (V(s) - r)^2
# ----------------------------
def main():
    rng = random.Random(0)
    torch.manual_seed(0)

    actor = Actor(d_in=4, n_actions=2)
    critic = Critic(d_in=4)

    opt_actor = optim.Adam(actor.parameters(), lr=1e-2)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-2)

    print("=== Actor–Critic Demo (Tiny) ===")
    print("Actor: state -> action probabilities")
    print("Critic: state -> V(s)\n")

    # We'll use one fixed state for teaching clarity
    # Example: loyal-ish customer: recent + frequent + medium spend + low sensitivity
    s = make_state(recency_days=20, freq_30d=12, spend=600.0, sensitivity=0.15)

    for step in range(1, 21):
        logits, probs = actor(s)
        dist = torch.distributions.Categorical(probs)
        a = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(a))

        v = critic(s)  # V(s)

        r = simulate_reward(s, a, rng)  # reward/profit observed
        r_t = torch.tensor(r, dtype=torch.float32)

        # Advantage = reward - baseline
        # Plain English: "Was the outcome better or worse than what critic expected?"
        advantage = (r_t - v.detach())

        # Actor learns "what to do"
        actor_loss = -(logp * advantage)

        # Critic learns "how good the situation is"
        critic_loss = (critic(s) - r_t) ** 2

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # Print a compact teaching log
        print(f"Step {step:>2} | logits=[{logits[0]:+.2f},{logits[1]:+.2f}] "
              f"probs=[{probs[0]:.2f},{probs[1]:.2f}] "
              f"action={ACTIONS[a]:>10} | reward r={r:6.1f} | V(s)={float(v.item()):6.2f} | A={float(advantage.item()):6.2f}")

    print("\nTeaching summary:")
    print("- Actor outputs probabilities and gets updated to favor actions with positive advantage.")
    print("- Critic predicts V(s) so the actor compares against an expectation (reduces overreaction to random outcomes).")
    print('- Key line: "Actor learns what to do. Critic learns how good the situation is."')

if __name__ == "__main__":
    main()
