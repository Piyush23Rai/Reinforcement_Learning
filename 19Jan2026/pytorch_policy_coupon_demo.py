#!/usr/bin/env python3  # This line tells the computer to run this file using Python 3 interpreter.
"""
Real-world-ish PyTorch demo: Policy Network for Retail Coupon Decision

Goal for class:
- Show "state as a numeric vector"
- Show policy πθ(a|s) from a neural network
- Sample action stochastically
- Compute reward (profit)
- Update policy to increase probability of profitable actions
"""  # This is a multi-line comment (docstring) explaining the program's purpose and learning goals.

from __future__ import annotations  # This imports future features to allow better type hints in Python.
import random  # This imports the random module to generate random numbers for simulations.
from dataclasses import dataclass  # This imports dataclass to create simple classes for data storage.
from typing import Tuple, List  # This imports type hints for better code documentation.

import torch  # This imports PyTorch, a library for building and training neural networks.
import torch.nn as nn  # This imports the neural network module from PyTorch.
import torch.optim as optim  # This imports optimization algorithms from PyTorch for training.


# ----------------------------
# 1) Define the "business environment" (toy but realistic)
# ----------------------------  # These lines are comments separating different sections of the code.
@dataclass  # This decorator makes the class automatically handle data storage.
class Customer:  # This defines a class to represent a customer with various attributes.
    recency_days: int          # days since last purchase  # Number of days since the customer's last purchase.
    freq_30d: int              # purchases in last 30 days  # Number of purchases in the last 30 days.
    avg_basket: float          # avg basket value ($)  # Average value of the customer's shopping basket in dollars.
    coupon_sensitivity: float  # 0..1 (how likely coupon increases conversion)  # How much a coupon affects the customer's buying decision, from 0 to 1.
    segment: str               # "LOYAL" | "PRICE_SENSITIVE" | "COUPON_ADDICT"  # The type of customer segment they belong to.


SEGMENTS = ["LOYAL", "PRICE_SENSITIVE", "COUPON_ADDICT"]  # This list defines the possible customer segments.
ACTIONS = ["SEND_COUPON", "NO_COUPON"]  # This list defines the possible actions: sending a coupon or not.


def build_state_vector(c: Customer) -> torch.Tensor:  # This function converts a Customer object into a numerical vector (state) for the neural network.
    """
    d = 7 state features:
      [recency_norm, freq_norm, basket_norm, sensitivity,
       is_loyal, is_price_sensitive, is_addict]
    """  # This docstring explains the 7 features that make up the state vector.
    recency_norm = min(c.recency_days, 365) / 365.0  # Normalize recency to a value between 0 and 1 by capping at 365 days and dividing.
    freq_norm = min(c.freq_30d, 30) / 30.0  # Normalize frequency to 0-1 by capping at 30 and dividing.
    basket_norm = max(0.0, min(c.avg_basket, 300.0)) / 300.0  # Normalize basket value to 0-1, assuming max $300.
    sens = max(0.0, min(c.coupon_sensitivity, 1.0))  # Ensure sensitivity is between 0 and 1.

    # One-hot segment  # Comment: Now encoding the segment as one-hot (binary) values.
    is_loyal = 1.0 if c.segment == "LOYAL" else 0.0  # 1 if loyal, 0 otherwise.
    is_price = 1.0 if c.segment == "PRICE_SENSITIVE" else 0.0  # 1 if price sensitive, 0 otherwise.
    is_addict = 1.0 if c.segment == "COUPON_ADDICT" else 0.0  # 1 if coupon addict, 0 otherwise.

    s = torch.tensor([recency_norm, freq_norm, basket_norm, sens, is_loyal, is_price, is_addict],
                     dtype=torch.float32)  # Create a PyTorch tensor (array) with these 7 float values.
    return s  # Return the state vector tensor.


def simulate_purchase_and_reward(c: Customer, action: int, rng: random.Random) -> float:  # This function simulates if a customer buys and calculates the profit (reward).
    """
    Reward = profit.
    - If purchase happens: profit = margin - coupon_cost(if coupon sent)
    - If no purchase: profit = 0

    Purchase probability:
    - base depends on segment
    - sending coupon increases p for price-sensitive, but can reinforce addiction in long run (not modeled here)
    """  # Docstring explaining how reward is calculated and purchase probability.
    SEND = 0  # Constant for sending coupon action.
    NO = 1  # Constant for not sending coupon action.

    # Base purchase probability by segment (toy)  # Comment: Different segments have different base chances of buying.
    base_p = {"LOYAL": 0.45, "PRICE_SENSITIVE": 0.20, "COUPON_ADDICT": 0.30}[c.segment]  # Dictionary mapping segment to base probability.

    # Coupon uplift: depends on sensitivity, strongest for price-sensitive/addict  # Comment: Coupon increases buy chance based on sensitivity.
    uplift = c.coupon_sensitivity * (0.25 if c.segment != "LOYAL" else 0.10)  # Calculate uplift amount.

    p_buy = base_p + (uplift if action == SEND else 0.0)  # Add uplift if coupon sent.
    p_buy = max(0.0, min(p_buy, 0.95))  # Clamp probability between 0 and 0.95.

    buy = (rng.random() < p_buy)  # Randomly decide if customer buys based on probability.

    margin = 0.25 * c.avg_basket  # Calculate profit margin as 25% of basket value.
    coupon_cost = 5.0  # Cost of sending a coupon.

    if not buy:  # If no purchase, profit is 0.
        return 0.0

    profit = margin - (coupon_cost if action == SEND else 0.0)  # Profit is margin minus coupon cost if sent.
    return float(profit)  # Return the profit as a float.


def sample_customer(rng: random.Random) -> Customer:  # This function randomly generates a customer based on segment probabilities.
    seg = rng.choices(SEGMENTS, weights=[0.45, 0.40, 0.15], k=1)[0]  # Randomly choose segment with given weights.

    if seg == "LOYAL":  # If loyal segment, set attributes within loyal ranges.
        recency = rng.randint(1, 60)  # Random recency between 1 and 60 days.
        freq = rng.randint(5, 20)  # Random frequency between 5 and 20.
        basket = rng.uniform(80, 250)  # Random basket between $80 and $250.
        sens = rng.uniform(0.05, 0.35)  # Random sensitivity between 0.05 and 0.35.
    elif seg == "PRICE_SENSITIVE":  # If price sensitive, set attributes accordingly.
        recency = rng.randint(10, 180)  # Random recency 10-180 days.
        freq = rng.randint(1, 10)  # Frequency 1-10.
        basket = rng.uniform(40, 160)  # Basket $40-160.
        sens = rng.uniform(0.40, 0.90)  # Sensitivity 0.40-0.90.
    else:  # COUPON_ADDICT  # If coupon addict.
        recency = rng.randint(1, 45)  # Recency 1-45 days.
        freq = rng.randint(3, 15)  # Frequency 3-15.
        basket = rng.uniform(30, 120)  # Basket $30-120.
        sens = rng.uniform(0.70, 1.00)  # Sensitivity 0.70-1.00.

    return Customer(recency, freq, basket, sens, seg)  # Return the created Customer object.


# ----------------------------
# 2) Policy network πθ(a|s)
# ----------------------------  # Comment: Section for the policy neural network.
class PolicyNet(nn.Module):  # This class defines the policy network, inheriting from PyTorch's Module.
    def __init__(self, d_in: int, n_actions: int):  # Constructor taking input dimension and number of actions.
        super().__init__()  # Call parent class constructor.
        self.net = nn.Sequential(  # Define a sequential neural network.
            nn.Linear(d_in, 32),  # First layer: linear transformation from d_in to 32 neurons.
            nn.ReLU(),  # Activation function: ReLU for non-linearity.
            nn.Linear(32, n_actions),  # Second layer: to number of actions.
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # Forward pass method.
        logits = self.net(s)  # Pass state through network to get logits (raw scores).
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities using softmax.
        return probs  # Return the probability distribution over actions.


# ----------------------------
# 3) Training loop (policy gradient)
# ----------------------------  # Comment: Section for the training process using policy gradient.
def main():  # Main function where the training happens.
    rng = random.Random(0)  # Create a random number generator with seed 0 for reproducibility.
    torch.manual_seed(0)  # Set PyTorch's random seed for consistent results.

    d_in = 7  # Input dimension: 7 features in state.
    n_actions = 2  # Number of actions: send or not send coupon.
    policy = PolicyNet(d_in, n_actions)  # Create the policy network.
    opt = optim.Adam(policy.parameters(), lr=1e-2)  # Create Adam optimizer with learning rate 0.01.

    # We'll track average profit  # Comment: Initialize list to track profits.
    profits: List[float] = []  # List to store profit from each step.
    baseline = 0.0  # Moving average baseline to reduce training variance.

    print("=== PyTorch Policy Demo: Retail Coupon ===")  # Print header message.
    print("Actions: 0=SEND_COUPON, 1=NO_COUPON\n")  # Print action meanings.

    for step in range(1, 501):  # Loop for 500 training steps.
        c = sample_customer(rng)  # Sample a random customer.
        s = build_state_vector(c)  # Build state vector from customer.

        probs = policy(s)  # Get action probabilities from policy network.
        dist = torch.distributions.Categorical(probs)  # Create categorical distribution from probabilities.
        a = int(dist.sample().item())  # Sample an action randomly based on probabilities.

        profit = simulate_purchase_and_reward(c, a, rng)  # Simulate purchase and get profit.
        profits.append(profit)  # Add profit to the list.

        # Baseline update (moving average) to stabilize learning  # Comment: Update baseline for better learning.
        baseline = 0.99 * baseline + 0.01 * profit  # Exponential moving average of profit.
        advantage = profit - baseline  # Calculate advantage as profit minus baseline.

        # REINFORCE loss: maximize E[profit] -> minimize negative  # Comment: Policy gradient loss.
        # loss = -log π(a|s) * advantage  # Formula for loss.
        logp = dist.log_prob(torch.tensor(a))  # Log probability of the taken action.
        loss = -(logp * torch.tensor(advantage, dtype=torch.float32))  # Compute loss.

        opt.zero_grad()  # Clear previous gradients.
        loss.backward()  # Compute gradients.
        opt.step()  # Update network parameters.

        # Print occasionally  # Comment: Print progress every 50 steps.
        if step % 50 == 0:  # Every 50 steps.
            avg_profit = sum(profits[-50:]) / 50.0  # Average profit of last 50 steps.
            print(f"Step {step:>3} | avg_profit(last 50) = {avg_profit:6.2f} | baseline≈{baseline:5.2f}")  # Print stats.

            # Show behavior for a canonical "LOYAL" example  # Comment: Show policy for a sample loyal customer.
            loyal = Customer(recency_days=15, freq_30d=12, avg_basket=150, coupon_sensitivity=0.20, segment="LOYAL")  # Create example customer.
            s_loyal = build_state_vector(loyal)  # Build state.
            p = policy(s_loyal).detach().numpy()  # Get probabilities, detach from graph, convert to numpy.
            print(f"  Policy for LOYAL example: P(SEND)={p[0]:.3f}, P(NO)={p[1]:.3f}")  # Print probabilities.

    print("\nDone. Key teaching points:")  # Print final message.
    print("1) State is a vector of numbers (features).")  # Teaching point 1.
    print("2) Policy network outputs probabilities πθ(a|s).")  # Teaching point 2.
    print("3) We sample actions stochastically to explore.")  # Teaching point 3.
    print("4) Profitable actions become more likely over time.\n")  # Teaching point 4.


if __name__ == "__main__":  # This checks if the script is run directly (not imported).
    main()  # Call the main function.
