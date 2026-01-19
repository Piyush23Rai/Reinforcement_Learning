
"""
Tiny tabular Q-learning demo (Retail Coupon).
- Q(s,a) = expected long-run profit score for doing action a in situation s.
- We print the mini example:
    Q("LOYAL", SEND_COUPON)=7.0 vs Q("LOYAL", NO_COUPON)=9.5
  -> coupon not worth it for loyal customers (in this toy example).

Bonus:
- We also show a single Q-learning update step:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
"""

from __future__ import annotations
from typing import Dict, List, Tuple

STATES: List[str] = ["LOYAL", "PRICE_SENSITIVE", "COUPON_ADDICT"]
ACTIONS: List[str] = ["SEND_COUPON", "NO_COUPON"]

# A tiny Q-table with made-up values for teaching in class
Q: Dict[str, Dict[str, float]] = {
    "LOYAL": {
        "SEND_COUPON": 7.0,
        "NO_COUPON": 9.5,
    },
    "PRICE_SENSITIVE": {
        "SEND_COUPON": 8.8,
        "NO_COUPON": 6.1,
    },
    "COUPON_ADDICT": {
        "SEND_COUPON": 5.2,
        "NO_COUPON": 4.7,
    },
}


def best_action(state: str) -> Tuple[str, float]:
    """Return (best_action, best_q_value) for a given state."""
    action = max(Q[state], key=Q[state].get)
    return action, Q[state][action]


def explain_state(state: str) -> None:
    """Print Q-values for both actions and the chosen best action."""
    send = Q[state]["SEND_COUPON"]
    no = Q[state]["NO_COUPON"]
    chosen, chosen_q = best_action(state)

    print(f"\nState s = '{state}'")
    print(f"  Q(s, SEND_COUPON) = {send:.1f}")
    print(f"  Q(s, NO_COUPON)   = {no:.1f}")
    print(f"  => Best action a* = {chosen}  (because {chosen_q:.1f} is higher)")

    if chosen == "NO_COUPON":
        print("  Output: Coupon is not worth it here (lower long-term profit in this toy table).")
    else:
        print("  Output: Coupon is worth it here (higher long-term profit in this toy table).")


def q_learning_update(
    s: str,
    a: str,
    r: float,
    s_next: str,
    alpha: float = 0.10,
    gamma: float = 0.95,
) -> None:
    """
    One-step Q-learning update:
      Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))
    """
    old_q = Q[s][a]
    _, best_next_q = best_action(s_next)

    target = r + gamma * best_next_q
    new_q = old_q + alpha * (target - old_q)

    print("\n--- One Q-learning update step (bonus) ---")
    print(f"Given: s='{s}', a='{a}', r={r:.2f}, s_next='{s_next}'")
    print(f"Old Q(s,a) = {old_q:.3f}")
    print(f"Best next Q(s_next, ·) = {best_next_q:.3f}")
    print(f"Target = r + gamma*best_next_q = {r:.2f} + {gamma:.2f}*{best_next_q:.3f} = {target:.3f}")
    print(f"Update: Q <- Q + alpha*(Target - Q)")
    print(f"New Q(s,a) = {old_q:.3f} + {alpha:.2f}*({target:.3f} - {old_q:.3f}) = {new_q:.3f}")

    Q[s][a] = new_q


def main() -> None:
    print("=== Tiny Tabular Q Demo (Retail Coupon) ===")
    print("Interpretation: Q(s,a) is a long-run profit score for taking action a in state s.")

    # The exact mini example you asked for
    explain_state("LOYAL")

    # Show more states (optional but useful for teaching)
    explain_state("PRICE_SENSITIVE")
    explain_state("COUPON_ADDICT")

    # Bonus: show one Q-learning update to demonstrate learning mechanics
    # Example scenario: We sent a coupon to a PRICE_SENSITIVE customer,
    # got reward r=+2.0 (profit), and the next state became COUPON_ADDICT.
    q_learning_update(
        s="PRICE_SENSITIVE",
        a="SEND_COUPON",
        r=2.0,
        s_next="COUPON_ADDICT",
        alpha=0.10,
        gamma=0.95,
    )

    # Print the updated row to show the effect
    print("\nUpdated Q row for PRICE_SENSITIVE:")
    for act in ACTIONS:
        print(f"  Q('PRICE_SENSITIVE', {act}) = {Q['PRICE_SENSITIVE'][act]:.3f}")


if __name__ == "__main__":
    main()
