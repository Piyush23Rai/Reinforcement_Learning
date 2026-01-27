import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from common import (
    TinyCausalLM, RewardModel, encode, pad_batch,
    build_prompt_from_event, sample_response, split_prompt_response_tokens,
    logprob_of_response
)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
CKPT = OUT_DIR / "checkpoints"
EVENTS_CSV = DATA_DIR / "fintech_events.csv"

def load_vocab():
    vocab = json.loads((CKPT / "vocab.json").read_text())["vocab"]
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return vocab, stoi, itos

def load_models(stoi, device):
    vocab_size = len(stoi)
    sft = TinyCausalLM(vocab_size=vocab_size).to(device)
    rm = RewardModel(vocab_size=vocab_size).to(device)
    sft.load_state_dict(torch.load(CKPT / "sft_lm.pt", map_location=device))
    rm.load_state_dict(torch.load(CKPT / "reward_model.pt", map_location=device))
    sft.eval(); rm.eval()
    return sft, rm

def load_events():
    df = pd.read_csv(EVENTS_CSV)
    for c in ["event_time","merchant_id","txn_id","currency","country","payment_method","auth_result","error_code","issuer_response","avs_result","three_ds"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    for c in ["amount","fraud_score","chargeback_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

@torch.no_grad()
def rm_score_text(rm, stoi, prompt, response, device):
    text = "PROMPT: " + prompt + " <sep> " + response
    ids = [encode(text, stoi, max_len=240)]
    x, m = pad_batch(ids, pad_id=stoi["<pad>"])
    x = x.to(device); m = m.to(device)
    return float(rm(x, m).item())

def generate_candidates(policy, rm, stoi, itos, prompt_text, k, device):
    prompt_ids = encode(prompt_text, stoi, max_len=120)
    cands = []
    for _ in range(k):
        full = sample_response(policy, prompt_ids, stoi, itos, max_new_tokens=70, temperature=0.9, top_k=30, device=device)
        _, rtxt = split_prompt_response_tokens(full)
        score = rm_score_text(rm, stoi, prompt_text, rtxt, device)
        cands.append({"response": rtxt, "score": score})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands

def dpo_loss(policy, ref, stoi, batch, beta, device):
    losses = []
    for ex in batch:
        prompt = ex["prompt"]
        pref = ex["preferred"]
        disp = ex["dispreferred"]

        lp_pref_pi, _ = logprob_of_response(policy, stoi, prompt, pref, device=device)
        lp_disp_pi, _ = logprob_of_response(policy, stoi, prompt, disp, device=device)
        lp_pref_ref, _ = logprob_of_response(ref, stoi, prompt, pref, device=device)
        lp_disp_ref, _ = logprob_of_response(ref, stoi, prompt, disp, device=device)

        d_pi = lp_pref_pi - lp_disp_pi
        d_ref = lp_pref_ref - lp_disp_ref
        z = beta * (d_pi - d_ref)
        losses.append(-torch.log(torch.sigmoid(torch.tensor(z, device=device)) + 1e-12))
    return torch.stack(losses).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = args.device
    vocab, stoi, itos = load_vocab()
    sft, rm = load_models(stoi, device=device)

    policy = TinyCausalLM(vocab_size=len(vocab)).to(device)
    policy.load_state_dict(sft.state_dict())
    ref = TinyCausalLM(vocab_size=len(vocab)).to(device)
    ref.load_state_dict(sft.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    df = load_events()
    rows = df.sample(n=min(200, len(df)), random_state=SEED).to_dict(orient="records")

    opt = optim.Adam(policy.parameters(), lr=1e-4)

    losses = []
    gaps = []
    for step in range(1, args.steps + 1):
        batch = []
        gap_list = []
        for _ in range(args.batch):
            row = random.choice(rows)
            prompt = build_prompt_from_event(row)
            cands = generate_candidates(policy, rm, stoi, itos, prompt, k=args.k, device=device)
            best, worst = cands[0], cands[-1]
            batch.append({"prompt": prompt, "preferred": best["response"], "dispreferred": worst["response"]})
            gap_list.append(best["score"] - worst["score"])
        loss = dpo_loss(policy, ref, stoi, batch, beta=args.beta, device=device)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        gaps.append(float(np.mean(gap_list)))

        if step % max(10, args.steps//10) == 0:
            print(f"Step {step:>4}/{args.steps} | DPO loss={loss.item():.4f} | avg(RM best-worst gap)={gaps[-1]:.3f}")

    torch.save(policy.state_dict(), CKPT / "dpo_policy.pt")

    fig = plt.figure(figsize=(11,4))
    ax1 = fig.add_subplot(1,2,1); ax1.plot(losses); ax1.set_title("DPO Loss"); ax1.set_xlabel("Step")
    ax2 = fig.add_subplot(1,2,2); ax2.plot(gaps); ax2.set_title("Avg RM score gap (best-worst)"); ax2.set_xlabel("Step")
    fig.tight_layout()
    out_png = OUT_DIR / "dpo_metrics.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print("\nSaved:")
    print(f" - {CKPT / 'dpo_policy.pt'}")
    print(f" - {out_png}")

if __name__ == "__main__":
    main()