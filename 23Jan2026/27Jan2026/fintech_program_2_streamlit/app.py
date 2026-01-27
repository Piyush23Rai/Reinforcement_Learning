import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from common import (
    TinyCausalLM, RewardModel,
    encode, pad_batch,
    build_prompt_from_event,
    sample_response, split_prompt_response_tokens
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
CKPT = OUT_DIR / "checkpoints"
EVENTS_CSV = DATA_DIR / "fintech_events.csv"

st.set_page_config(page_title="FinTech RLHF Demo (Program 2)", layout="wide")

@st.cache_data
def load_events():
    df = pd.read_csv(EVENTS_CSV)
    for c in ["event_time","merchant_id","txn_id","currency","country","payment_method","auth_result","error_code","issuer_response","avs_result","three_ds"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    for c in ["amount","fraud_score","chargeback_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

@st.cache_data
def load_vocab():
    vocab = json.loads((CKPT / "vocab.json").read_text())["vocab"]
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return vocab, stoi, itos

@st.cache_resource
def load_models(vocab_size, device="cpu"):
    sft = TinyCausalLM(vocab_size=vocab_size).to(device)
    rm = RewardModel(vocab_size=vocab_size).to(device)
    sft.load_state_dict(torch.load(CKPT / "sft_lm.pt", map_location=device))
    rm.load_state_dict(torch.load(CKPT / "reward_model.pt", map_location=device))
    sft.eval(); rm.eval()
    return sft, rm

@torch.no_grad()
def rm_score(rm, stoi, prompt, response, device="cpu"):
    text = "PROMPT: " + prompt + " <sep> " + response
    ids = [encode(text, stoi, max_len=240)]
    x, m = pad_batch(ids, pad_id=stoi["<pad>"])
    x = x.to(device); m = m.to(device)
    return float(rm(x, m).item())

@torch.no_grad()
def generate_k(policy, rm, stoi, itos, prompt, k=6, device="cpu"):
    prompt_ids = encode(prompt, stoi, max_len=120)
    rows = []
    for i in range(k):
        full = sample_response(policy, prompt_ids, stoi, itos, max_new_tokens=70, temperature=0.9, top_k=30, device=device)
        _, resp = split_prompt_response_tokens(full)
        score = rm_score(rm, stoi, prompt, resp, device=device)
        rows.append({"candidate": i+1, "rm_score": score, "response": resp})
    rows.sort(key=lambda r: r["rm_score"], reverse=True)
    return rows

st.title("FinTech RLHF Demo — Program 2 (Candidates → RM Score → DPO)")
st.caption("Pick a payment event → generate multiple responses → rank by Reward Model. Train DPO offline, then compare SFT vs DPO policy.")

if not EVENTS_CSV.exists():
    st.error("Missing data/fintech_events.csv")
    st.stop()
if not (CKPT / "vocab.json").exists():
    st.error("Missing checkpoints. Run: python program1_train_and_save.py")
    st.stop()

df = load_events()
vocab, stoi, itos = load_vocab()
device = "cpu"
sft, rm = load_models(len(vocab), device=device)

left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("1) Choose event")
    idx = st.number_input("Row index", min_value=0, max_value=int(len(df)-1), value=0, step=1)
    row = df.iloc[int(idx)].to_dict()
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

with right:
    st.subheader("2) Generate candidates")
    policy_choice = st.selectbox("Policy to sample from", ["SFT (reference)", "DPO policy (if trained)"])
    policy = TinyCausalLM(vocab_size=len(vocab)).to(device)
    if policy_choice.startswith("DPO") and (CKPT / "dpo_policy.pt").exists():
        policy.load_state_dict(torch.load(CKPT / "dpo_policy.pt", map_location=device))
        st.success("Loaded DPO policy from out/checkpoints/dpo_policy.pt")
    else:
        policy.load_state_dict(sft.state_dict())
        st.info("Using SFT reference policy")
    policy.eval()

    prompt = build_prompt_from_event(row)
    st.text_area("Prompt", prompt, height=160)

    k = st.slider("Candidates (K)", 2, 10, 6)
    if st.button("Generate"):
        cand_rows = generate_k(policy, rm, stoi, itos, prompt, k=k, device=device)
        st.session_state["cand_rows"] = cand_rows

    if "cand_rows" in st.session_state:
        cand_rows = st.session_state["cand_rows"]
        st.write("Ranked by RM score (higher = preferred):")
        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True)
        st.markdown("### ✅ Best (RM-high)")
        st.write(cand_rows[0]["response"])
        st.markdown("### ❌ Worst (RM-low)")
        st.write(cand_rows[-1]["response"])

st.divider()
st.subheader("3) Train DPO (offline script)")
st.code("python program2_dpo_train.py --steps 200 --k 6", language="bash")
st.write("After training, switch policy selector to **DPO policy** and regenerate candidates to see better RM-ranked outputs.")