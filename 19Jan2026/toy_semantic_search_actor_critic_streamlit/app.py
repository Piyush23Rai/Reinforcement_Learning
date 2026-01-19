
import streamlit as st
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# Toy Production Example: Semantic Search + Data Contracts
# Actor–Critic demo (class-friendly)
# =========================================================
#
# Real-world story (plain English):
# - You have an enterprise data catalog (datasets + metadata).
# - A user searches: "monthly revenue by region".
# - The system must choose a retrieval strategy:
#     (vector vs hybrid), (strict contracts vs lenient), (top-k)
#
# Actor (policy network):
#   - takes the state s (query + role + risk flags) and outputs
#     probabilities for each strategy πθ(a|s).
#
# Critic (value network):
#   - predicts how good this situation is likely to be Vϕ(s)
#     (expected success / reward).
#
# After showing results, we collect feedback:
#   - Good / Bad (or simulated proxy KPI)
# and compute:
#   Advantage A = reward - V(s)
#
# Then we update:
#   Actor: increase probability of chosen action if A>0, decrease if A<0
#   Critic: learn to predict the reward baseline V(s)
#
# This is intentionally small and readable for teaching.
# =========================================================


# ----------------------------
# 1) Tiny "Data Catalog"
# ----------------------------
CATALOG = [
    {
        "id": "ds_rev_region",
        "title": "Revenue by Region (Gold)",
        "description": "Monthly revenue by region, currency and FX-adjusted. Finance certified.",
        "tags": ["revenue", "finance", "region", "monthly", "gold"],
        "completeness": 0.98,
        "pii": False,
        "certified": True,
    },
    {
        "id": "ds_rev_raw",
        "title": "Revenue Events (Raw)",
        "description": "Transaction-level revenue events. Region mapping can be missing.",
        "tags": ["revenue", "transactions", "raw", "events"],
        "completeness": 0.72,
        "pii": False,
        "certified": False,
    },
    {
        "id": "ds_orders",
        "title": "Orders (Silver)",
        "description": "Orders with customer and item counts. Not finance certified.",
        "tags": ["orders", "customers", "items", "silver"],
        "completeness": 0.90,
        "pii": False,
        "certified": False,
    },
    {
        "id": "ds_marketing_spend",
        "title": "Marketing Spend (Gold)",
        "description": "Campaign spend by channel/day with attribution keys.",
        "tags": ["marketing", "spend", "channel", "campaign", "gold"],
        "completeness": 0.95,
        "pii": False,
        "certified": True,
    },
    {
        "id": "api_contract_registry",
        "title": "Data Contract Registry API",
        "description": "API for contract schemas, required fields, and validation status.",
        "tags": ["data contract", "schema", "validation", "registry", "api"],
        "completeness": 0.99,
        "pii": False,
        "certified": True,
    },
    {
        "id": "ds_customer_profile",
        "title": "Customer Profile (PII Restricted)",
        "description": "Customer demographics + identifiers. Access-controlled due to PII.",
        "tags": ["customer", "pii", "restricted", "profile"],
        "completeness": 0.93,
        "pii": True,
        "certified": False,
    },
]

# Actions = retrieval strategies a0..a4
ACTIONS = [
    {"name": "Vector + Lenient + top10", "mode": "vector", "strict": False, "topk": 10},
    {"name": "Hybrid + Lenient + top10", "mode": "hybrid", "strict": False, "topk": 10},
    {"name": "Hybrid + Strict  + top10", "mode": "hybrid", "strict": True,  "topk": 10},
    {"name": "Hybrid + Lenient + top30", "mode": "hybrid", "strict": False, "topk": 30},
    {"name": "Hybrid + Strict  + top30", "mode": "hybrid", "strict": True,  "topk": 30},
]

ROLES = ["Analyst", "Engineer", "Executive"]


def contract_ok(asset: Dict, strict: bool, min_completeness: float) -> bool:
    """Strict contracts filter out low-completeness assets."""
    if not strict:
        return True
    return asset["completeness"] >= min_completeness


def contract_warning(asset: Dict, min_completeness: float) -> str:
    if asset["completeness"] >= min_completeness:
        return ""
    return f"⚠️ Completeness low ({asset['completeness']:.2f} < {min_completeness:.2f}). Fields may be missing/unstable."


# ----------------------------
# 2) Semantic Search (TF-IDF)
# ----------------------------
@st.cache_data
def build_vectorizer(catalog: List[Dict]):
    docs = []
    for a in catalog:
        docs.append(f"{a['title']} {a['description']} {' '.join(a['tags'])}")
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(docs)
    return vec, mat


def rank_assets(query: str, mode: str, tag_boost: float, title_boost: float) -> List[Tuple[Dict, float]]:
    vec, mat = build_vectorizer(CATALOG)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()

    q_words = set([w.lower() for w in query.split()])

    results = []
    for i, a in enumerate(CATALOG):
        score = float(sims[i])

        if mode == "hybrid":
            overlap = len(q_words.intersection(set(a["tags"])))
            score = score + tag_boost * overlap
            if any(w in a["title"].lower() for w in q_words):
                score = score + title_boost

        results.append((a, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ----------------------------
# 3) State vector s (what networks see)
# ----------------------------
def featurize_state(query: str, role: str, pii_allowed: bool) -> np.ndarray:
    """
    d = 8 interpretable features:
      0 q_len_norm        : query length / 30
      1 time_intent       : query mentions monthly/daily/week
      2 finance_intent    : query mentions revenue/spend/profit
      3 contract_intent   : query mentions contract/schema/validated
      4 role_Analyst      : one-hot
      5 role_Engineer     : one-hot
      6 role_Executive    : one-hot
      7 risk_flag         : 1 if pii_allowed is False AND query mentions customer/profile/email
    """
    q = query.lower()
    qlen = min(len(q.split()), 30) / 30.0
    time_intent = 1.0 if ("monthly" in q or "daily" in q or "week" in q) else 0.0
    finance_intent = 1.0 if ("revenue" in q or "spend" in q or "profit" in q) else 0.0
    contract_intent = 1.0 if ("contract" in q or "schema" in q or "validated" in q) else 0.0

    role_oh = [1.0 if role == r else 0.0 for r in ROLES]

    pii_query = ("customer" in q or "profile" in q or "email" in q or "phone" in q)
    risk_flag = 1.0 if (not pii_allowed and pii_query) else 0.0

    s = np.array([qlen, time_intent, finance_intent, contract_intent] + role_oh + [risk_flag], dtype=np.float32)
    return s


# ----------------------------
# 4) Actor–Critic Networks
# ----------------------------
class Actor(nn.Module):
    """Actor outputs action probabilities πθ(a|s)."""
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class Critic(nn.Module):
    """Critic outputs Vϕ(s) (expected reward)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def init_models(seed: int = 7, lr_actor: float = 2e-3, lr_critic: float = 2e-3):
    torch.manual_seed(seed)
    in_dim = 8
    n_actions = len(ACTIONS)
    actor = Actor(in_dim, n_actions)
    critic = Critic(in_dim)
    opt_a = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=lr_critic)
    return actor, critic, opt_a, opt_c


def choose_action(actor: Actor, s_np: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    x = torch.tensor(s_np).unsqueeze(0)  # [1,d]
    with torch.no_grad():
        logits, probs = actor(x)
    dist = torch.distributions.Categorical(probs.squeeze(0))
    a = int(dist.sample().item())
    return a, logits.squeeze(0).numpy(), probs.squeeze(0).numpy()


def actor_critic_update(actor, critic, opt_a, opt_c, s_np, a, reward):
    """
    One-step Actor–Critic update (teaching version):

    Advantage:
      A = r - V(s)

    Actor loss:
      L_actor = - log π(a|s) * A

    Critic loss:
      L_critic = (V(s) - r)^2
    """
    x = torch.tensor(s_np).unsqueeze(0)
    logits, probs = actor(x)
    dist = torch.distributions.Categorical(probs.squeeze(0))

    logp = dist.log_prob(torch.tensor(a))
    v = critic(x).squeeze(0)

    r = torch.tensor(reward, dtype=torch.float32)
    advantage = (r - v.detach())

    actor_loss = -(logp * advantage)
    critic_loss = (critic(x).squeeze(0) - r) ** 2

    opt_a.zero_grad()
    actor_loss.backward()
    opt_a.step()

    opt_c.zero_grad()
    critic_loss.backward()
    opt_c.step()

    return float(v.item()), float(advantage.item()), float(actor_loss.item()), float(critic_loss.item())


# ----------------------------
# 5) Auto proxy reward (toy KPI)
# ----------------------------
def auto_reward_proxy(query: str, results: List[Tuple[Dict, float]], require_certified: bool, pii_allowed: bool, min_completeness: float) -> Tuple[float, str]:
    """
    Toy production KPI reward (for teaching):

    +0.7 if top-1 matches finance intent
    +0.3 if top-1 matches time intent
    +0.2 if certified required and top-1 is certified
    -0.3 if certified required and top-1 is NOT certified
    -0.6 if query implies PII but pii not allowed and PII appears in top-5
    -0.2 if top-1 completeness below threshold
    -0.7 if no results (strict filtered everything)

    Reward is clamped to [-1, +1].
    """
    q = query.lower()
    pii_query = ("customer" in q or "profile" in q or "email" in q or "phone" in q)

    if len(results) == 0:
        return -0.7, "No results after strict filtering (-0.7)."

    top1 = results[0][0]
    msg = []
    r = 0.0

    finance_intent = ("revenue" in q or "spend" in q or "profit" in q)
    time_intent = ("monthly" in q or "daily" in q or "week" in q)

    if finance_intent and ("revenue" in top1["tags"] or "spend" in top1["tags"]):
        r += 0.7
        msg.append("Top-1 matches finance intent (+0.7).")
    if time_intent and ("monthly" in top1["tags"]):
        r += 0.3
        msg.append("Top-1 matches time intent (+0.3).")

    if require_certified and top1["certified"]:
        r += 0.2
        msg.append("Certified required and top-1 is certified (+0.2).")
    if require_certified and (not top1["certified"]):
        r -= 0.3
        msg.append("Certified required but top-1 not certified (-0.3).")

    if top1["completeness"] < min_completeness:
        r -= 0.2
        msg.append("Top-1 completeness below threshold (-0.2).")

    if (not pii_allowed) and pii_query:
        pii_in_top = any(a["pii"] for a, _ in results[:5])
        if pii_in_top:
            r -= 0.6
            msg.append("PII appears in top results while not allowed (-0.6).")
        else:
            r += 0.1
            msg.append("PII avoided (+0.1).")

    r = float(np.clip(r, -1.0, 1.0))
    return r, " ".join(msg) if msg else "Neutral outcome."


# ----------------------------
# 6) Streamlit UI
# ----------------------------
st.set_page_config(page_title="Toy Semantic Search Actor–Critic Demo", layout="wide")
st.title("Toy Semantic Search: Actor–Critic in a Production-like Workflow")
st.caption("Actor chooses retrieval strategy; Critic predicts expected KPI; system learns from feedback.")

with st.expander("Mini English (what to say in class)", expanded=True):
    st.markdown(
        """
- **Actor**: “Given this situation, which strategy should we run?” → probabilities for strategies.
- **Critic**: “How good is this situation likely to be?” → predicts expected KPI reward **V(s)**.
- **Reward**: comes from KPI feedback (user satisfaction, compliance, safety, latency).
- **Advantage**: **A = r - V(s)** → “better/worse than expected” drives learning.
"""
    )

# Sidebar controls
st.sidebar.header("Controls")
reward_mode = st.sidebar.radio("Reward source", ["Manual (Good/Bad)", "Auto proxy KPI (toy)"])
min_completeness = st.sidebar.slider("Strict contract completeness threshold", 0.70, 0.99, 0.92, 0.01)
tag_boost = st.sidebar.slider("Hybrid tag overlap boost", 0.00, 0.15, 0.03, 0.01)
title_boost = st.sidebar.slider("Hybrid title boost", 0.00, 0.20, 0.05, 0.01)

st.sidebar.markdown("---")
lr_actor = st.sidebar.slider("Actor LR", 1e-4, 1e-2, 2e-3, format="%.4f")
lr_critic = st.sidebar.slider("Critic LR", 1e-4, 1e-2, 2e-3, format="%.4f")

if "actor" not in st.session_state:
    st.session_state.actor, st.session_state.critic, st.session_state.opt_a, st.session_state.opt_c = init_models(
        seed=7, lr_actor=lr_actor, lr_critic=lr_critic
    )
    st.session_state.last = None
    st.session_state.hist = {"reward": [], "V": [], "A": [], "action": []}

# If LRs changed, update optimizers
for pg in st.session_state.opt_a.param_groups:
    pg["lr"] = lr_actor
for pg in st.session_state.opt_c.param_groups:
    pg["lr"] = lr_critic

col1, col2 = st.columns([1.05, 1])

with col1:
    st.subheader("Step 1 — Input (like production)")
    query = st.text_input("User query", value="monthly revenue by region")
    role = st.selectbox("User role", ROLES, index=0)
    require_certified = st.checkbox("Require certified datasets", value=True)
    pii_allowed = st.checkbox("PII allowed", value=False)

    s_np = featurize_state(query, role, pii_allowed=pii_allowed)

    st.markdown("#### State vector **s** (what the networks see)")
    st.code(
        "s = [q_len_norm, time_intent, finance_intent, contract_intent, "
        "role_Analyst, role_Engineer, role_Executive, risk_flag]\n"
        f"s = {np.array2string(s_np, precision=2)}"
    )

    if st.button("Run Search (Actor chooses)"):
        a, logits, probs = choose_action(st.session_state.actor, s_np)
        action = ACTIONS[a]

        ranked = rank_assets(query, mode=action["mode"], tag_boost=tag_boost, title_boost=title_boost)
        filtered = []
        for asset, score in ranked:
            if contract_ok(asset, strict=action["strict"], min_completeness=min_completeness):
                filtered.append((asset, score))
        results = filtered[: action["topk"]]

        x = torch.tensor(s_np).unsqueeze(0)
        with torch.no_grad():
            V_before = float(st.session_state.critic(x).item())

        st.session_state.last = {
            "s_np": s_np,
            "a": a,
            "action": action,
            "logits": logits,
            "probs": probs,
            "results": results,
            "V_before": V_before,
            "query": query,
            "role": role,
            "require_certified": require_certified,
            "pii_allowed": pii_allowed,
        }

with col2:
    st.subheader("Step 2 — Actor output + results")
    if st.session_state.last is None:
        st.info("Click **Run Search** to see Actor probabilities and results.")
    else:
        last = st.session_state.last

        st.markdown("#### Actor: logits → probabilities (softmax)")
        st.code(
            f"logits = {np.array2string(last['logits'], precision=2)}\n"
            f"probs  = {np.array2string(last['probs'], precision=3)}"
        )

        st.markdown("#### Strategy probabilities πθ(a|s)")
        for i, act in enumerate(ACTIONS):
            marker = "✅" if i == last["a"] else "  "
            st.write(f"{marker} **a{i}: {act['name']}** → `{last['probs'][i]:.3f}`")

        st.markdown("---")
        st.markdown("#### Critic: predicted expected reward V(s)")
        st.write(f"**V(s) = {last['V_before']:.3f}**")

        st.markdown("---")
        st.markdown("#### Top results (preview)")
        if len(last["results"]) == 0:
            st.warning("No results after strict filtering.")
        else:
            for asset, score in last["results"][:10]:
                st.write(f"**{asset['title']}** (score={score:.3f})")
                st.caption(asset["description"])
                if asset["pii"]:
                    st.error("Contains PII (restricted).")
                warn = contract_warning(asset, min_completeness=min_completeness)
                if warn:
                    st.warning(warn)
                st.write("")

        st.markdown("---")
        st.subheader("Step 3 — Reward + update (learning)")

        reward = None
        expl = ""

        if reward_mode == "Manual (Good/Bad)":
            cA, cB = st.columns(2)
            with cA:
                if st.button("👍 Good (r=+1)"):
                    reward = 1.0
                    expl = "Manual: user liked results."
            with cB:
                if st.button("👎 Bad (r=-1)"):
                    reward = -1.0
                    expl = "Manual: user disliked results."
        else:
            if st.button("Compute proxy KPI reward + update"):
                reward, expl = auto_reward_proxy(
                    last["query"],
                    last["results"],
                    require_certified=last["require_certified"],
                    pii_allowed=last["pii_allowed"],
                    min_completeness=min_completeness,
                )

        if reward is not None:
            V, A, Lpi, Lv = actor_critic_update(
                st.session_state.actor,
                st.session_state.critic,
                st.session_state.opt_a,
                st.session_state.opt_c,
                last["s_np"],
                last["a"],
                reward,
            )

            st.session_state.hist["reward"].append(reward)
            st.session_state.hist["V"].append(V)
            st.session_state.hist["A"].append(A)
            st.session_state.hist["action"].append(last["a"])

            st.success("Actor–Critic updated.")
            st.write(f"Reward **r = {reward:+.2f}** → {expl}")
            st.write(f"Critic predicted **V(s) = {V:.3f}**")
            st.write(f"Advantage **A = r - V(s) = {A:+.3f}**")
            st.caption("Mini English: A>0 means better than expected → do this strategy more. A<0 → do it less.")
            st.write(f"Actor loss = `{Lpi:.4f}` | Critic loss = `{Lv:.4f}`")

        if len(st.session_state.hist["reward"]) > 0:
            st.markdown("---")
            st.subheader("Learning curves")
            st.line_chart({"reward": st.session_state.hist["reward"]})
            st.line_chart({"V(s)": st.session_state.hist["V"]})
            st.line_chart({"advantage A": st.session_state.hist["A"]})
            st.caption("Reward is noisy. Critic learns the average. Advantage shows surprise.")

st.markdown("---")
st.markdown("### One-sentence production takeaway")
st.markdown("**Actor chooses the strategy. Critic predicts expected KPI. Advantage tells the actor if this choice was better or worse than expected.**")
