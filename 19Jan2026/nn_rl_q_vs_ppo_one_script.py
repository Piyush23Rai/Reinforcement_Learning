#!/usr/bin/env python3  # This line tells the computer to run this file using Python 3 interpreter.
"""
Neural Approximation in RL — One Script Teaching Pack
=======================================================

This script teaches the *full concept* with ONE runnable file:

1) Why tabular RL doesn't scale (huge |S| and sparse revisits)
2) States as vectors: s in R^d (what a neural network sees)
3) Option A (Value-based): Learn Qθ(s,a)  -> "DQN lineage"
4) Option B (Policy-based): Learn πθ(a|s) directly with Proximal Policy Optimization (PPO)
   - PPO clipping
   - Value baseline Vϕ(s)
   - Advantage estimates

We use a single "Retail Coupon" toy environment that is realistic enough for class:
- State features: recency, freq, basket, sensitivity, segment one-hot
- Actions: SEND_COUPON, NO_COUPON
- Reward: profit = margin - coupon_cost(if coupon sent) if purchase happens else 0
- Transition: state evolves a little; coupon can increase "addiction risk" mildly

This is NOT production RL code.
It is intentionally small and readable for teaching.
"""  # This is a multi-line comment (docstring) explaining the script's purpose and structure.

from __future__ import annotations  # This imports future features to allow better type hints in Python.
from dataclasses import dataclass  # This imports dataclass to create simple classes for data storage.
from collections import deque  # This imports deque for efficient queue operations, used for replay buffer.
from typing import List, Tuple, Dict  # This imports type hints for better code documentation.
import random  # This imports the random module for generating random numbers.
import math  # This imports the math module for mathematical functions.

import torch  # This imports PyTorch, a library for building and training neural networks.
import torch.nn as nn  # This imports the neural network module from PyTorch.
import torch.optim as optim  # This imports optimization algorithms from PyTorch for training.


# ----------------------------
# 0) Global configuration
# ----------------------------  # These lines are comments separating different sections of the code.
SEED = 7  # This sets a seed value for reproducible random number generation.
random.seed(SEED)  # This sets the seed for the random module.
torch.manual_seed(SEED)  # This sets the seed for PyTorch's random number generator.

DEVICE = torch.device("cpu")  # keep classroom simple  # This sets the device to CPU for simplicity in teaching.
A_SEND, A_NO = 0, 1  # This defines constants for actions: 0 for send coupon, 1 for no coupon.
ACTIONS = ["SEND_COUPON", "NO_COUPON"]  # This list defines the action names.


# ----------------------------
# 1) Environment: Retail Coupon (toy but realistic)
# ----------------------------  # Comment: Section for defining the environment.
@dataclass  # This decorator makes the class automatically handle data storage.
class CustomerState:  # This defines a class to represent the state of a customer.
    recency_days: int         # 0..365  # Number of days since last purchase.
    freq_30d: int             # 0..30  # Number of purchases in last 30 days.
    avg_basket: float         # dollars  # Average basket value in dollars.
    sensitivity: float        # 0..1  # How sensitive the customer is to coupons.
    segment: int              # 0=LOYAL,1=PRICE_SENSITIVE,2=COUPON_ADDICT  # Customer segment as an integer.
    addiction: float          # 0..1 (latent-ish risk score)  # Addiction level, a hidden risk factor.


def clamp(x: float, lo: float, hi: float) -> float:  # This function clamps a value between low and high.
    return max(lo, min(hi, x))  # Returns the value clamped to the range.


def featurize(s: CustomerState) -> torch.Tensor:  # This function converts a CustomerState into a numerical vector (features) for the neural network.
    """
    State as a vector s ∈ R^d (what the network sees).

    d = 8 features:
      [recency_norm, freq_norm, basket_norm, sensitivity, addiction,
       one_hot_loyal, one_hot_price, one_hot_addict]
    """  # Docstring explaining the 8 features in the state vector.
    recency_norm = clamp(s.recency_days, 0, 365) / 365.0  # Normalize recency to 0-1.
    freq_norm = clamp(s.freq_30d, 0, 30) / 30.0  # Normalize frequency to 0-1.
    basket_norm = clamp(s.avg_basket, 0.0, 300.0) / 300.0  # Normalize basket to 0-1.
    sens = clamp(s.sensitivity, 0.0, 1.0)  # Clamp sensitivity to 0-1.
    addict = clamp(s.addiction, 0.0, 1.0)  # Clamp addiction to 0-1.

    one_hot = [0.0, 0.0, 0.0]  # Initialize one-hot encoding list.
    one_hot[s.segment] = 1.0  # Set the corresponding segment to 1.

    x = torch.tensor(  # Create a PyTorch tensor with the features.
        [recency_norm, freq_norm, basket_norm, sens, addict] + one_hot,
        dtype=torch.float32,
        device=DEVICE,
    )
    return x  # Return the feature tensor.


def sample_initial_state(rng: random.Random) -> CustomerState:  # This function randomly generates an initial customer state.
    """
    Draw a customer from a mixture of segments.
    """  # Docstring explaining the function.
    segment = rng.choices([0, 1, 2], weights=[0.45, 0.40, 0.15], k=1)[0]  # Randomly choose segment with weights.
    if segment == 0:  # LOYAL  # If loyal segment.
        rec = rng.randint(1, 60)  # Random recency 1-60.
        freq = rng.randint(6, 20)  # Random frequency 6-20.
        basket = rng.uniform(90, 260)  # Random basket 90-260.
        sens = rng.uniform(0.05, 0.35)  # Random sensitivity 0.05-0.35.
        addiction = rng.uniform(0.00, 0.10)  # Random addiction 0.00-0.10.
    elif segment == 1:  # PRICE_SENSITIVE  # If price sensitive.
        rec = rng.randint(10, 220)  # Random recency 10-220.
        freq = rng.randint(1, 10)  # Frequency 1-10.
        basket = rng.uniform(40, 180)  # Basket 40-180.
        sens = rng.uniform(0.40, 0.90)  # Sensitivity 0.40-0.90.
        addiction = rng.uniform(0.05, 0.25)  # Addiction 0.05-0.25.
    else:  # COUPON_ADDICT  # If coupon addict.
        rec = rng.randint(1, 50)  # Recency 1-50.
        freq = rng.randint(3, 15)  # Frequency 3-15.
        basket = rng.uniform(30, 130)  # Basket 30-130.
        sens = rng.uniform(0.70, 1.00)  # Sensitivity 0.70-1.00.
        addiction = rng.uniform(0.35, 0.75)  # Addiction 0.35-0.75.

    return CustomerState(rec, freq, basket, sens, segment, addiction)  # Return the created CustomerState.


def step_env(s: CustomerState, action: int, rng: random.Random) -> Tuple[CustomerState, float, bool]:  # This function simulates one step in the environment.
    """
    One environment step.

    Reward (profit):
      - Purchase occurs with probability p_buy
      - If purchase: profit = margin - coupon_cost(if send)
      - Else: 0

    Transition:
      - recency resets if purchase else increases
      - freq_30d moves slowly
      - addiction increases slightly when coupon sent
    """  # Docstring explaining the step function.
    # Base purchase probabilities by segment  # Comment: Different base probabilities for each segment.
    base_p = [0.45, 0.20, 0.30][s.segment]  # List of base probabilities.

    # Coupon uplift depends on sensitivity; smaller uplift for loyal  # Comment: Coupon increases buy chance.
    uplift = s.sensitivity * (0.22 if s.segment != 0 else 0.08)  # Calculate uplift.

    # Addiction makes "no coupon" harder (customer expects coupon)  # Comment: Addiction affects probability.
    addiction_penalty = 0.15 * s.addiction if action == A_NO else 0.0  # Penalty if no coupon and addicted.

    p_buy = base_p + (uplift if action == A_SEND else 0.0) - addiction_penalty  # Calculate total buy probability.
    p_buy = clamp(p_buy, 0.02, 0.95)  # Clamp probability.

    buy = rng.random() < p_buy  # Randomly decide if purchase happens.

    margin = 0.25 * s.avg_basket  # Calculate profit margin.
    coupon_cost = 5.0  # Cost of coupon.

    reward = 0.0  # Initialize reward.
    if buy:  # If purchase.
        reward = margin - (coupon_cost if action == A_SEND else 0.0)  # Calculate reward.

    # Next state evolution (toy)  # Comment: Update state for next step.
    rec_next = 0 if buy else min(365, s.recency_days + rng.randint(1, 7))  # Update recency.
    freq_next = s.freq_30d  # Start with current frequency.
    if buy:  # If bought.
        freq_next = min(30, freq_next + 1)  # Increase frequency.
    else:  # If not bought.
        # slowly decay frequency if no purchase  # Comment: Slowly decrease frequency.
        if rng.random() < 0.10:  # 10% chance.
            freq_next = max(0, freq_next - 1)  # Decrease frequency.

    basket_next = clamp(s.avg_basket + rng.uniform(-10, 10), 30.0, 300.0)  # Slightly change basket.

    # Coupon slightly increases addiction risk  # Comment: Update addiction.
    addiction_next = s.addiction + (0.02 if action == A_SEND else -0.005)  # Increase if coupon sent.
    addiction_next = clamp(addiction_next, 0.0, 1.0)  # Clamp addiction.

    s_next = CustomerState(rec_next, freq_next, basket_next, s.sensitivity, s.segment, addiction_next)  # Create next state.

    done = False  # we will use fixed-horizon episodes  # Episodes don't end naturally.
    return s_next, float(reward), done  # Return next state, reward, and done flag.


# ----------------------------
# 2) Option A: Learn Qθ(s,a) (Value-based, DQN lineage)
# ----------------------------  # Comment: Section for Q-learning (value-based).
class QNet(nn.Module):  # This class defines the Q-network.
    """
    Qθ(s,a): network maps state vector -> Q-values for each action
    """  # Docstring explaining the Q-network.
    def __init__(self, d_in: int, n_actions: int = 2):  # Constructor.
        super().__init__()  # Call parent constructor.
        self.net = nn.Sequential(  # Define sequential network.
            nn.Linear(d_in, 64),  # Linear layer from input to 64 neurons.
            nn.ReLU(),  # ReLU activation.
            nn.Linear(64, n_actions),  # Linear layer to actions.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass.
        return self.net(x)  # Q-values (scores), not probabilities  # Return Q-values.


@dataclass  # Decorator for data class.
class Transition:  # Class to store a transition (experience).
    s: torch.Tensor  # State.
    a: int  # Action.
    r: float  # Reward.
    s_next: torch.Tensor  # Next state.
    done: bool  # Done flag.


def train_q_learning_value_based(  # Function to train Q-learning.
    rng: random.Random,
    steps: int = 4000,
    batch_size: int = 64,
    gamma: float = 0.98,
    lr: float = 2e-3,
    target_sync: int = 250,
) -> QNet:  # Returns trained Q-network.
    """
    Minimal DQN-style training loop:
    - Collect transitions into a replay buffer
    - TD target: r + γ max_a' Q_target(s',a')
    - Optimize MSE between Q(s,a) and TD target
    """  # Docstring explaining the training loop.

    d_in = 8  # Input dimension.
    q = QNet(d_in).to(DEVICE)  # Create Q-network.
    q_target = QNet(d_in).to(DEVICE)  # Create target Q-network.
    q_target.load_state_dict(q.state_dict())  # Copy parameters.

    opt = optim.Adam(q.parameters(), lr=lr)  # Adam optimizer.
    loss_fn = nn.MSELoss()  # Mean squared error loss.

    buffer: deque[Transition] = deque(maxlen=10_000)  # Replay buffer.
    s = sample_initial_state(rng)  # Sample initial state.

    print("\n==============================")  # Print header.
    print("Option A: Learn Qθ(s,a) (DQN lineage)")  # Print title.
    print("==============================")  # Print separator.
    print("Plain English: Network outputs a long-term SCORE for each action.")  # Explain.
    print("We pick action = argmax_a Qθ(s,a).\n")  # Explain action selection.

    for t in range(1, steps + 1):  # Training loop.
        x = featurize(s)  # Featurize state.

        # Epsilon-greedy exploration (classic DQN idea)  # Comment: Exploration strategy.
        eps = max(0.05, 0.70 * (1 - t / steps))  # Calculate epsilon.
        if rng.random() < eps:  # If explore.
            a = rng.choice([A_SEND, A_NO])  # Random action.
        else:  # If exploit.
            with torch.no_grad():  # No gradients.
                qvals = q(x)  # Get Q-values.
            a = int(torch.argmax(qvals).item())  # Choose best action.

        s_next, r, done = step_env(s, a, rng)  # Step environment.
        x_next = featurize(s_next)  # Featurize next state.

        buffer.append(Transition(x, a, r, x_next, done))  # Add to buffer.
        s = s_next  # Update state.

        if len(buffer) >= batch_size:  # If enough samples.
            batch = rng.sample(list(buffer), batch_size)  # Sample batch.

            S = torch.stack([tr.s for tr in batch], dim=0)  # Stack states.
            A = torch.tensor([tr.a for tr in batch], dtype=torch.int64, device=DEVICE)  # Actions.
            R = torch.tensor([tr.r for tr in batch], dtype=torch.float32, device=DEVICE)  # Rewards.
            S2 = torch.stack([tr.s_next for tr in batch], dim=0)  # Next states.
            D = torch.tensor([tr.done for tr in batch], dtype=torch.float32, device=DEVICE)  # Dones.

            # Q(s,a) for chosen actions  # Comment: Get Q-values for chosen actions.
            q_sa = q(S).gather(1, A.view(-1, 1)).squeeze(1)  # Gather Q(s,a).

            # TD target = r + γ * max_a' Q_target(s', a')  # Comment: Calculate target.
            with torch.no_grad():  # No gradients.
                max_q_next = q_target(S2).max(dim=1).values  # Max Q next.
                target = R + gamma * (1.0 - D) * max_q_next  # TD target.

            loss = loss_fn(q_sa, target)  # Compute loss.

            opt.zero_grad()  # Clear gradients.
            loss.backward()  # Backpropagate.
            opt.step()  # Update parameters.

        # Sync target network occasionally  # Comment: Sync target network.
        if t % target_sync == 0:  # Every target_sync steps.
            q_target.load_state_dict(q.state_dict())  # Copy parameters.

        # Print teaching diagnostics  # Comment: Print progress.
        if t % 800 == 0:  # Every 800 steps.
            loyal_example = CustomerState(15, 12, 150.0, 0.20, 0, 0.05)  # loyal-like  # Example customer.
            xL = featurize(loyal_example)  # Featurize.
            with torch.no_grad():  # No gradients.
                qL = q(xL)  # Get Q-values.
            print(f"Step {t:>4} | eps={eps:.2f} | Q(loyal, SEND)={qL[0]:.2f}  Q(loyal, NO)={qL[1]:.2f}  -> pick {ACTIONS[int(torch.argmax(qL))]}")  # Print Q-values and action.

    print("\nTeaching wrap-up for Option A:")  # Print summary.
    print("- Q-network learns 'scores' for actions.")  # Explain.
    print("- This connects to DQN: Q-table replaced by neural network Qθ.\n")  # Explain.
    return q  # Return trained network.


# ----------------------------
# 3) Option B: Learn πθ(a|s) directly with Proximal Policy Optimization (PPO)
# ----------------------------  # Comment: Section for PPO (policy-based).
class PolicyNet(nn.Module):  # Class for policy network.
    """
    πθ(a|s): network outputs action probabilities
    """  # Docstring.
    def __init__(self, d_in: int, n_actions: int = 2):  # Constructor.
        super().__init__()  # Parent.
        self.net = nn.Sequential(  # Network.
            nn.Linear(d_in, 64),  # Layer.
            nn.Tanh(),  # Activation.
            nn.Linear(64, n_actions),  # Layer.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward.
        logits = self.net(x)  # Logits.
        return torch.softmax(logits, dim=-1)  # Probabilities.


class ValueNet(nn.Module):  # Class for value network.
    """
    Vϕ(s): critic baseline predicting expected return from state
    """  # Docstring.
    def __init__(self, d_in: int):  # Constructor.
        super().__init__()  # Parent.
        self.net = nn.Sequential(  # Network.
            nn.Linear(d_in, 64),  # Layer.
            nn.Tanh(),  # Activation.
            nn.Linear(64, 1),  # Layer.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward.
        return self.net(x).squeeze(-1)  # Value.


@dataclass  # Decorator.
class Rollout:  # Class for rollout data.
    states: torch.Tensor      # [T, d]  # States.
    actions: torch.Tensor     # [T]  # Actions.
    rewards: torch.Tensor     # [T]  # Rewards.
    logp_old: torch.Tensor    # [T]  # Old log probabilities.
    values: torch.Tensor      # [T]  # Values.


def compute_returns_advantages(  # Function to compute returns and advantages.
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns returns and advantages.
    """
    Simple advantage estimate:
      Return_t = r_t + γ r_{t+1} + ...
      Advantage_t = Return_t - V(s_t)

    For teaching: keep it simple (GAE can be added later).
    """  # Docstring.
    T = rewards.shape[0]  # Time steps.
    returns = torch.zeros_like(rewards)  # Initialize returns.
    G = 0.0  # Cumulative return.
    for t in reversed(range(T)):  # Backward pass.
        G = rewards[t] + gamma * G  # Update G.
        returns[t] = G  # Set return.
    adv = returns - values  # Advantage.
    # Normalize advantage (common PPO trick)  # Comment: Normalize for stability.
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # Normalize.
    return returns, adv  # Return.


def train_policy_with_ppo(  # Function to train with PPO.
    rng: random.Random,
    iters: int = 80,
    horizon: int = 60,
    gamma: float = 0.98,
    clip_eps: float = 0.20,
    policy_lr: float = 2e-3,
    value_lr: float = 2e-3,
    ppo_epochs: int = 4,
) -> Tuple[PolicyNet, ValueNet]:  # Returns policy and value networks.
    """
    Minimal Proximal Policy Optimization (PPO) training loop:
    - Collect one rollout (T steps) with current policy (store logp_old)
    - Compute returns and advantages using critic baseline
    - Update:
        ratio = π_new(a|s) / π_old(a|s)
        clip ratio to [1-ε, 1+ε]
      + value regression for critic

    This is the core Proximal Policy Optimization (PPO) idea.
    """  # Docstring.

    d_in = 8  # Input dimension.
    policy = PolicyNet(d_in).to(DEVICE)  # Policy network.
    value = ValueNet(d_in).to(DEVICE)  # Value network.

    opt_pi = optim.Adam(policy.parameters(), lr=policy_lr)  # Policy optimizer.
    opt_v = optim.Adam(value.parameters(), lr=value_lr)  # Value optimizer.

    print("\n===============================================")  # Print header.
    print("Option B: Learn πθ(a|s) with Proximal Policy Optimization (PPO)")  # Title.
    print("===============================================")  # Separator.
    print("Plain English: Network outputs action probabilities.")  # Explain.
    print("Proximal Policy Optimization (PPO) updates policy carefully using clipping + value baseline.\n")  # Explain.

    # For readable progress  # Comment: Track recent returns.
    recent_returns: deque[float] = deque(maxlen=10)  # Deque for recent returns.

    for it in range(1, iters + 1):  # Iteration loop.
        # Start each iteration with a fresh customer episode  # Comment: New episode.
        s = sample_initial_state(rng)  # Sample initial state.

        # Collect rollout  # Comment: Collect trajectory.
        states, actions, rewards, logp_old, values = [], [], [], [], []  # Lists for data.
        for _ in range(horizon):  # For each step in horizon.
            x = featurize(s)  # Featurize.
            with torch.no_grad():  # No gradients.
                probs = policy(x)  # Get probabilities.
                v = value(x)  # Get value.

            dist = torch.distributions.Categorical(probs)  # Distribution.
            a = dist.sample()  # Sample action.
            lp = dist.log_prob(a)  # Log prob.

            s2, r, done = step_env(s, int(a.item()), rng)  # Step.

            states.append(x)  # Append state.
            actions.append(a)  # Action.
            rewards.append(torch.tensor(r, dtype=torch.float32, device=DEVICE))  # Reward.
            logp_old.append(lp)  # Log prob.
            values.append(v)  # Value.

            s = s2  # Update state.

        roll = Rollout(  # Create rollout object.
            states=torch.stack(states),  # Stack states.
            actions=torch.stack(actions).to(torch.int64),  # Actions.
            rewards=torch.stack(rewards),  # Rewards.
            logp_old=torch.stack(logp_old),  # Log probs.
            values=torch.stack(values),  # Values.
        )

        # Compute returns + advantages  # Comment: Compute advantages.
        returns, adv = compute_returns_advantages(roll.rewards, roll.values, gamma=gamma)  # Compute.

        # Proximal Policy Optimization (PPO) updates (multiple epochs over same rollout)  # Comment: PPO updates.
        for _epoch in range(ppo_epochs):  # For each epoch.
            probs_new = policy(roll.states)                       # [T, 2]  # New probabilities.
            dist_new = torch.distributions.Categorical(probs_new)  # New distribution.
            logp_new = dist_new.log_prob(roll.actions)            # [T]  # New log probs.
            ratio = torch.exp(logp_new - roll.logp_old)           # π_new / π_old  # Ratio.

            # Clipped surrogate  # Comment: Clipped loss.
            unclipped = ratio * adv  # Unclipped.
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv  # Clipped.
            policy_loss = -(torch.min(unclipped, clipped)).mean()  # Loss.

            # Critic loss  # Comment: Value loss.
            v_pred = value(roll.states)  # Predicted values.
            value_loss = ((v_pred - returns) ** 2).mean()  # MSE.

            # Small entropy bonus (keeps exploration)  # Comment: Entropy for exploration.
            entropy = dist_new.entropy().mean()  # Entropy.
            entropy_bonus = 0.01 * entropy  # Bonus.

            # Update policy  # Comment: Update policy.
            opt_pi.zero_grad()  # Clear gradients.
            (policy_loss - entropy_bonus).backward()  # Backprop.
            opt_pi.step()  # Step.

            # Update critic  # Comment: Update value.
            opt_v.zero_grad()  # Clear.
            value_loss.backward()  # Backprop.
            opt_v.step()  # Step.

        # Logging: mean return of the rollout  # Comment: Log return.
        mean_return = float(roll.rewards.sum().item())  # Sum rewards.
        recent_returns.append(mean_return)  # Append.

        if it % 10 == 0:  # Every 10 iterations.
            loyal_example = CustomerState(15, 12, 150.0, 0.20, 0, 0.05)  # Example.
            xL = featurize(loyal_example)  # Featurize.
            with torch.no_grad():  # No gradients.
                pL = policy(xL).cpu().numpy()  # Get probs.
            print(  # Print.
                f"Iter {it:>3} | recent mean return (last {len(recent_returns)} iters) = {sum(recent_returns)/len(recent_returns):8.2f} "
                f"| π(loyal): P(SEND)={pL[0]:.3f}, P(NO)={pL[1]:.3f}"
            )

    print("\nTeaching wrap-up for Option B:")  # Summary.
    print("- Policy network outputs probabilities πθ(a|s).")  # Explain.
    print("- Proximal Policy Optimization (PPO) uses ratio + clipping to prevent huge unsafe updates.")  # Explain.
    print("- Critic Vϕ(s) provides a baseline so learning is stable.\n")  # Explain.
    return policy, value  # Return networks.


# ----------------------------
# 4) Main demo: run both options
# ----------------------------  # Comment: Main section.
def main():  # Main function.
    rng = random.Random(SEED)  # Random generator.

    print("============================================================")  # Header.
    print("Neural Approximation Demo: Qθ(s,a) vs πθ(a|s) (One Script)")  # Title.
    print("============================================================")  # Separator.
    print("Key message for class:")  # Message.
    print("- Tabular RL doesn't scale (huge |S| and sparse revisits).")  # Explain.
    print("- Neural RL learns a function that generalizes across similar states.\n")  # Explain.

    # Run Option A (Q-learning, DQN lineage)  # Comment: Run Q-learning.
    qnet = train_q_learning_value_based(rng, steps=4000)  # Train.

    # Run Option B (Policy-based, Proximal Policy Optimization (PPO))  # Comment: Run PPO.
    policy, value = train_policy_with_ppo(rng, iters=80)  # Train.

    # Final side-by-side check on same examples  # Comment: Compare.
    print("============================================================")  # Header.
    print("Final comparison on a few customer examples")  # Title.
    print("============================================================")  # Separator.

    examples = [  # List of examples.
        ("LOYAL-like", CustomerState(15, 12, 150.0, 0.20, 0, 0.05)),  # Loyal.
        ("PRICE-like", CustomerState(160, 3, 110.0, 0.80, 1, 0.15)),  # Price.
        ("ADDICT-like", CustomerState(10, 10, 80.0, 0.95, 2, 0.70)),  # Addict.
    ]

    for name, s in examples:  # For each example.
        x = featurize(s)  # Featurize.

        with torch.no_grad():  # No gradients.
            q = qnet(x)  # Q-values.
            p = policy(x)  # Probabilities.

        best_q = int(torch.argmax(q).item())  # Best Q action.
        best_p = int(torch.argmax(p).item())  # Best P action.

        print(f"\n{name}:")  # Print name.
        print(f"  Option A (Qθ): Q(SEND)={q[0]:.2f}, Q(NO)={q[1]:.2f} -> pick {ACTIONS[best_q]}")  # Q.
        print(f"  Option B (πθ): P(SEND)={p[0]:.3f}, P(NO)={p[1]:.3f} -> most likely {ACTIONS[best_p]}")  # P.

    print("\nDone.")  # Done.
    print("Teaching note:")  # Note.
    print("- Option A connects to DQN: learn Qθ(s,a), choose argmax.")  # Explain.
    print("- Option B is Proximal Policy Optimization (PPO): learn πθ(a|s) directly with safe clipped updates.")  # Explain.


if __name__ == "__main__":  # If run directly.
    main()  # Call main.
