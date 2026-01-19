#!/usr/bin/env python3
"""
Worked Retail State Explosion (Discretization still hurts)

We discretize customer state into buckets:

recency_days:   0..365   => 366 values
freq_30d:       0..30    => 31 values
basket_bucket:  50 buckets
sensitivity:    20 buckets
segment:        10 values

Compute:
- total number of possible states |S|
- Q-table size |S| * |A| for actions |A|
- memory estimate (float32/float64)
"""

from __future__ import annotations
import math

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

def compute_state_space(recency_vals: int, freq_vals: int, basket_buckets: int, sensitivity_buckets: int, segments: int) -> int:
    return recency_vals * freq_vals * basket_buckets * sensitivity_buckets * segments

def main() -> None:
    # Buckets (as per your slide)
    recency_vals = 366       # 0..365 inclusive
    freq_vals = 31           # 0..30 inclusive
    basket_buckets = 50
    sensitivity_buckets = 20
    segments = 10

    print("=== Worked Retail State Explosion ===\n")
    print("Discretized buckets:")
    print(f"  recency_days:      0..365  => {recency_vals} values")
    print(f"  freq_30d:          0..30   => {freq_vals} values")
    print(f"  basket_bucket:             => {basket_buckets} buckets")
    print(f"  sensitivity:               => {sensitivity_buckets} buckets")
    print(f"  segment:                   => {segments} values\n")

    # State space size
    S = compute_state_space(recency_vals, freq_vals, basket_buckets, sensitivity_buckets, segments)

    # Show the math explicitly
    print("State space math:")
    print(f"  |S| = {recency_vals} × {freq_vals} × {basket_buckets} × {sensitivity_buckets} × {segments}")
    print(f"     = {S:,} possible states\n")

    print("Plain English (teaching line):")
    print(f"  Even with buckets, we create about {S/1e6:.1f} million possible situations.\n")

    # Q-table size for different action counts
    print("Q-table size for actions |A|:")
    for A in [2, 5, 10, 20]:
        q_entries = S * A
        mem_f32 = q_entries * 4   # float32
        mem_f64 = q_entries * 8   # float64
        print(f"  If |A| = {A:>2}:  Q entries = |S|×|A| = {S:,} × {A} = {q_entries:,}")
        print(f"           memory (float32) ≈ {bytes_to_human(mem_f32)} | (float64) ≈ {bytes_to_human(mem_f64)}")

    print("\nMini example (as in your slide):")
    A = 2
    print(f"  With {A} actions, Q-table entries ≈ {S*A:,} (~{(S*A)/1e6:.1f} million)\n")

    print("Key inference:")
    print("  1) This is just the number of entries. Learning them requires repeated visits.")
    print("  2) In real life, most (state, action) pairs are rarely repeated.")
    print("  3) So tabular RL struggles; neural networks generalize across similar states.")

if __name__ == "__main__":
    main()
