import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


# =========================
# 0) Coupon MDP (stationary)
# =========================
S = [0, 1, 2, 3]
A = [0, 1]
ENGAGED, NEUTRAL, AT_RISK, ADDICT = 0, 1, 2, 3
NO_COUPON, COUPON = 0, 1

STATE_NAME = {0: "ENGAGED", 1: "NEUTRAL", 2: "AT_RISK", 3: "ADDICT"}
ACTION_NAME = {0: "NO_COUPON", 1: "COUPON"}


def mdp_step(state: int, action: int, rng: random.Random) -> Tuple[int, float]:
    """
    Transition + reward (simple, classroom-friendly).
    Rewards encourage conversion, but addiction reduces margin.
    """
    # Rewards
    if action == COUPON:
        reward = 1.0 if state == ADDICT else 3.0
    else:
        if state == ENGAGED:
            reward = 2.0
        elif state == NEUTRAL:
            reward = 1.0
        elif state == AT_RISK:
            reward = 0.2
        else:
            reward = 0.0

    r = rng.random()

    # Transitions (coupon increases addiction risk)
    if action == COUPON:
        if state == ENGAGED:
            # 60% stay engaged, 25% neutral, 15% addicted
            if r < 0.60:
                ns = ENGAGED
            elif r < 0.85:
                ns = NEUTRAL
            else:
                ns = ADDICT
        elif state == NEUTRAL:
            # 50% engaged, 25% neutral, 25% addicted
            if r < 0.50:
                ns = ENGAGED
            elif r < 0.75:
                ns = NEUTRAL
            else:
                ns = ADDICT
        elif state == AT_RISK:
            # 55% neutral, 15% engaged, 30% addicted
            if r < 0.55:
                ns = NEUTRAL
            elif r < 0.70:
                ns = ENGAGED
            else:
                ns = ADDICT
        else:  # addicted
            ns = ADDICT if r < 0.90 else NEUTRAL
    else:
        # no coupon: safer but may drift toward risk
        if state == ENGAGED:
            ns = ENGAGED if r < 0.75 else NEUTRAL
        elif state == NEUTRAL:
            if r < 0.20:
                ns = ENGAGED
            elif r < 0.70:
                ns = NEUTRAL
            else:
                ns = AT_RISK
        elif state == AT_RISK:
            ns = NEUTRAL if r < 0.25 else AT_RISK
        else:  # addicted
            if r < 0.30:
                ns = NEUTRAL
            elif r < 0.60:
                ns = AT_RISK
            else:
                ns = ADDICT

    return ns, reward


# =========================
# 1) MCTS Node
# =========================
@dataclass
class Node:
    state: int
    depth: int
    parent: Optional["Node"] = None
    parent_action: Optional[int] = None

    children: Dict[int, "Node"] = field(default_factory=dict)  # action -> child
    N: int = 0  # visits to this state node
    N_sa: Dict[int, int] = field(default_factory=lambda: {a: 0 for a in A})
    Q_sa: Dict[int, float] = field(default_factory=lambda: {a: 0.0 for a in A})

    def is_fully_expanded(self) -> bool:
        return all(a in self.children for a in A)


def ucb_score(Q: float, N_s: int, N_sa: int, c: float) -> float:
    """
    UCB1 = Q + c*sqrt(ln(N_s)/N_sa)
    If N_sa==0, return +inf to force trying it at least once.
    """
    if N_sa == 0:
        return float("inf")
    return Q + c * math.sqrt(math.log(max(1, N_s)) / N_sa)


def select_action_ucb(node: Node, c: float) -> Tuple[int, Dict[int, float]]:
    """
    Returns chosen action and a dict of ucb scores for all actions (for display).
    """
    scores = {}
    best_a = None
    best_score = -1e18
    for a in A:
        score = ucb_score(node.Q_sa[a], node.N, node.N_sa[a], c)
        scores[a] = score
        if score > best_score:
            best_score = score
            best_a = a
    return best_a, scores


def rollout(state: int, H: int, gamma: float, rng: random.Random) -> float:
    """
    Simple rollout policy: random actions.
    Return G = sum_{t=0..H} gamma^t r_t
    """
    G = 0.0
    s = state
    for t in range(H):
        a = rng.choice(A)
        s2, r = mdp_step(s, a, rng)
        G += (gamma ** t) * r
        s = s2
    return G


def backprop(path: List[Tuple[Node, int]], G: float):
    """
    path: list of (node, action_taken_from_node) along selection/expansion
    Update:
      Q <- Q + (G - Q)/N_sa
      N_sa <- N_sa + 1
      node.N <- node.N + 1
    """
    for node, a in path:
        node.N += 1
        node.N_sa[a] += 1
        n = node.N_sa[a]
        q_old = node.Q_sa[a]
        node.Q_sa[a] = q_old + (G - q_old) / n


def mcts_search(root_state: int, iters: int, H: int, gamma: float, c: float, seed: int):
    rng = random.Random(seed)
    root = Node(state=root_state, depth=0)

    # For teaching: keep a log of first few iterations
    logs = []

    for i in range(iters):
        node = root
        path = []
        iter_log = {"iter": i + 1}

        # ---- 1) Selection + (2) Expansion ----
        while True:
            a, scores = select_action_ucb(node, c)
            if i < 10:  # log only early iterations to keep UI light
                iter_log[f"UCB@{STATE_NAME[node.state]}(d={node.depth})"] = {
                    ACTION_NAME[k]: (None if math.isinf(v) else round(v, 4))
                    for k, v in scores.items()
                }

            # Expand if this action-child doesn't exist yet
            if a not in node.children:
                # take one real environment step to create child state
                s2, _ = mdp_step(node.state, a, rng)
                child = Node(state=s2, depth=node.depth + 1, parent=node, parent_action=a)
                node.children[a] = child
                path.append((node, a))
                node = child
                break  # stop at new node for rollout
            else:
                # move down existing child
                path.append((node, a))
                node = node.children[a]

            # stop expanding deeper than H (tree depth limit)
            if node.depth >= H:
                break

        # ---- 3) Simulation ----
        G = rollout(node.state, H=max(1, H - node.depth), gamma=gamma, rng=rng)

        # ---- 4) Backpropagation ----
        if path:
            backprop(path, G)

        if i < 10:
            iter_log["G(return)"] = round(G, 4)
            logs.append(iter_log)

    return root, logs


# =========================
# 2) Visualization helpers
# =========================
def build_pyvis_tree(root: Node, max_nodes: int = 250) -> str:
    net = Network(height="560px", width="100%", directed=True)
    net.toggle_physics(False)

    # BFS traversal
    q = [(root, "root")]
    seen = set()
    count = 0

    def node_id(n: Node, tag: str) -> str:
        return f"{id(n)}-{tag}"

    while q and count < max_nodes:
        node, tag = q.pop(0)
        nid = node_id(node, tag)
        if nid in seen:
            continue
        seen.add(nid)
        count += 1

        title = (
            f"State: {STATE_NAME[node.state]}<br>"
            f"Depth: {node.depth}<br>"
            f"N(s)={node.N}<br>"
            f"Q(s,NO)={node.Q_sa[NO_COUPON]:.3f}, N(s,NO)={node.N_sa[NO_COUPON]}<br>"
            f"Q(s,COUPON)={node.Q_sa[COUPON]:.3f}, N(s,COUPON)={node.N_sa[COUPON]}"
        )

        net.add_node(
            nid,
            label=f"{STATE_NAME[node.state]}\nd={node.depth}\nN={node.N}",
            title=title,
        )

        for a, child in node.children.items():
            cid = node_id(child, f"from_{a}")
            edge_title = (
                f"Action: {ACTION_NAME[a]}<br>"
                f"Q={node.Q_sa[a]:.3f}<br>"
                f"N={node.N_sa[a]}"
            )
            net.add_node(
                cid,
                label=f"{STATE_NAME[child.state]}\nd={child.depth}\nN={child.N}",
                title=(
                    f"State: {STATE_NAME[child.state]}<br>"
                    f"Depth: {child.depth}<br>"
                    f"N(s)={child.N}"
                ),
            )
            net.add_edge(nid, cid, label=ACTION_NAME[a], title=edge_title)
            q.append((child, f"from_{a}"))

    return net.generate_html()


def root_table(root: Node, c: float) -> pd.DataFrame:
    rows = []
    for a in A:
        score = ucb_score(root.Q_sa[a], root.N, root.N_sa[a], c)
        rows.append(
            {
                "Action": ACTION_NAME[a],
                "Q(s,a) (avg return)": root.Q_sa[a],
                "N(s,a) (action trials)": root.N_sa[a],
                "N(s) (state visits)": root.N,
                "UCB = Q + c*sqrt(ln(N)/Nsa)": score if not math.isinf(score) else float("inf"),
            }
        )
    return pd.DataFrame(rows)


# =========================
# 3) Streamlit UI
# =========================
st.set_page_config(page_title="MCTS Visualizer (Coupon MDP)", layout="wide")
st.title("MCTS Visualizer — Coupon MDP (Selection → Expansion → Rollout → Backprop)")

with st.sidebar:
    st.header("Controls")
    root_state = st.selectbox("Root state (s0)", options=S, format_func=lambda x: f"{x} = {STATE_NAME[x]}", index=2)
    iters = st.slider("MCTS simulations (iterations)", 10, 2000, 400, step=10)
    H = st.slider("Rollout horizon H", 1, 12, 6, step=1)
    gamma = st.slider("Discount γ", 0.0, 0.99, 0.90, step=0.01)
    c = st.slider("Exploration constant c", 0.0, 5.0, 2.0, step=0.1)
    seed = st.number_input("Random seed", value=42, step=1)
    run = st.button("Run MCTS", type="primary")

st.markdown(
    """
**What to look at in class**
- **UCB table (root):** shows **Q**, **N**, and the exploration bonus changing.
- **Tree graph:** hover nodes/edges to see `N(s)`, `Q(s,a)`, `N(s,a)`.
- **First 10 iterations log:** shows how early “∞” UCB forces each action to be tried.
"""
)

if run:
    root, logs = mcts_search(root_state=root_state, iters=iters, H=H, gamma=gamma, c=c, seed=seed)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Root math table (this is the UCB1 equation in numbers)")
        df = root_table(root, c=c)
        st.dataframe(df, use_container_width=True)

        st.caption(
            "Tip: Early on, actions with N(s,a)=0 get UCB=∞ so they are forced to be explored at least once."
        )

        st.subheader("First 10 simulations (teaching log)")
        if logs:
            st.json(logs)
        else:
            st.write("Run with more iterations to see logs.")

    with col2:
        st.subheader("MCTS tree (hover for Q/N details)")
        html = build_pyvis_tree(root, max_nodes=250)
        components.html(html, height=600, scrolling=True)

    st.subheader("What policy would MCTS pick at the root?")
    best_a = max(A, key=lambda a: root.Q_sa[a])
    st.write(
        f"**Best by Q at root:** {ACTION_NAME[best_a]} "
        f"(Q={root.Q_sa[best_a]:.3f}, N={root.N_sa[best_a]})"
    )

else:
    st.info("Set parameters on the left and click **Run MCTS**.")