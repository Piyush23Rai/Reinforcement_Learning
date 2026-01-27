import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from common import build_vocab, encode, pad_batch, TinyCausalLM, RewardModel, DEFAULT_INSTRUCTION, event_to_text

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
CKPT = OUT_DIR / "checkpoints"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

EVENTS_CSV = DATA_DIR / "fintech_events.csv"

def load_events():
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"Missing {EVENTS_CSV}. Put your CSV there.")
    df = pd.read_csv(EVENTS_CSV)
    for c in ["event_time","merchant_id","txn_id","currency","country","payment_method","auth_result","error_code","issuer_response","avs_result","three_ds"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    for c in ["amount","fraud_score","chargeback_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def build_pretrain_corpus(df, n_blocks=2000):
    idx = np.random.randint(0, len(df), size=n_blocks*4)
    lines = [event_to_text(df.iloc[i].to_dict()) for i in idx]
    return [" \n".join(lines[i:i+4]) for i in range(0, len(lines), 4)]

def build_sft_pairs(df, n_pairs=400):
    pairs = []
    sample = df.sample(n=min(len(df), n_pairs), random_state=SEED)
    for _, r in sample.iterrows():
        r = r.to_dict()
        instr = DEFAULT_INSTRUCTION
        actions = []
        if r["auth_result"] == "FAILED" and r.get("error_code","") == "GATEWAY_TIMEOUT":
            actions += [
                "Check PSP status dashboard and p95 latency.",
                "Enable retries with exponential backoff (max 2) for idempotent calls.",
                "Failover routing to secondary PSP if available.",
                "Open vendor ticket with masked txn_id + timestamp."
            ]
        if r["auth_result"] == "DECLINED" and r.get("error_code","") == "ISSUER_DO_NOT_HONOR":
            actions += [
                "Recommend customer retry after cooldown (issuer do-not-honor is often transient).",
                "Prompt for alternative payment method if repeated."
            ]
        if int(r["fraud_score"]) >= 80 and r["three_ds"] != "CHALLENGE":
            actions += [
                "Increase step-up authentication (3DS challenge) for this risk cohort.",
                "Review velocity/device rules to reduce chargeback exposure."
            ]
        if r["avs_result"] == "AVS_MISMATCH":
            actions += ["Consider stricter AVS/zip match or step-up for AVS mismatch traffic."]
        if int(r["chargeback_flag"]) == 1:
            actions += ["Flag as potential dispute: gather evidence (3DS result, device/IP, delivery proof if applicable)."]
        if not actions:
            actions = ["Monitor approval/decline rates; no immediate action required."]

        resp = (
            f"**Summary**\n"
            f"- Merchant: {r['merchant_id']}\n"
            f"- Country: {r['country']} | Amount: {r['amount']} {r['currency']}\n"
            f"- Result: {r['auth_result']}"
            + (f" | Error: {r.get('error_code','')}" if str(r.get("error_code","")) else "")
            + f"\n- Risk: fraud_score={int(r['fraud_score'])}, avs={r['avs_result']}, 3ds={r['three_ds']}\n\n"
            f"**Next actions (safe)**\n"
            + "\n".join([f"- {a}" for a in actions])
            + "\n\n**Compliance note**\n- Do not paste card numbers; use masked IDs/tokens only."
        )
        pairs.append({"instruction": instr, "response": resp, "event": event_to_text(r)})
    return pairs

def build_preferences(sft_pairs, n_pref=400):
    prefs = []
    sample = random.sample(sft_pairs, k=min(n_pref, len(sft_pairs)))
    dispreferred_templates = [
        "Summary: issue observed. Next actions: retry a few times and monitor. (No segmentation, no compliance note.)",
        "Summary: approvals dropped. Next actions: relax some risk checks temporarily and retry more to recover approvals.",
        "Summary: issuer declines seen. Next actions: ask customers to try again; no additional action required."
    ]
    for ex in sample:
        prompt = f"INSTRUCTION: {ex['instruction']} <sep> EVENT: {ex['event']} <sep> RESPONSE:"
        prefs.append({
            "prompt": prompt,
            "preferred": ex["response"],
            "dispreferred": random.choice(dispreferred_templates)
        })
    return prefs

def train_lm(model, stoi, texts, epochs, lr, batch_size, max_len):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    losses = []
    for ep in range(1, epochs+1):
        random.shuffle(texts)
        run = 0.0
        n = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            ids = [encode(t, stoi, max_len=max_len) for t in batch]
            x, _ = pad_batch(ids, pad_id=stoi["<pad>"])
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            n += 1
        avg = run / max(n, 1)
        losses.append(avg)
        print(f"  Epoch {ep}/{epochs}: Loss = {avg:.4f}")
    return losses

def rm_batchify(prefs, stoi, batch_size, max_len):
    for i in range(0, len(prefs), batch_size):
        ch = prefs[i:i+batch_size]
        p1 = [encode("PROMPT: " + ex["prompt"] + " <sep> " + ex["preferred"], stoi, max_len=max_len) for ex in ch]
        p2 = [encode("PROMPT: " + ex["prompt"] + " <sep> " + ex["dispreferred"], stoi, max_len=max_len) for ex in ch]
        x1, m1 = pad_batch(p1, pad_id=stoi["<pad>"])
        x2, m2 = pad_batch(p2, pad_id=stoi["<pad>"])
        yield x1, m1, x2, m2

def train_rm(rm, stoi, prefs, epochs=5, lr=2e-3, batch_size=16, max_len=220):
    opt = optim.Adam(rm.parameters(), lr=lr)
    losses, accs, margins = [], [], []
    for ep in range(1, epochs+1):
        random.shuffle(prefs)
        run_l, run_a, run_m = 0.0, 0.0, 0.0
        n = 0
        for x1, m1, x2, m2 in rm_batchify(prefs, stoi, batch_size, max_len):
            r1 = rm(x1, m1)
            r2 = rm(x2, m2)
            diff = r1 - r2
            loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                acc = (diff > 0).float().mean().item() * 100.0
                margin = diff.mean().item()
            run_l += loss.item(); run_a += acc; run_m += margin; n += 1
        avg_l = run_l / max(n,1); avg_a = run_a / max(n,1); avg_m = run_m / max(n,1)
        losses.append(avg_l); accs.append(avg_a); margins.append(avg_m)
        print(f"  Epoch {ep:>2}: Loss={avg_l:.4f}, Margin={avg_m:+.3f}, Acc={avg_a:.1f}%")
    return losses, accs, margins

def main():
    print("="*80)
    print("FINTECH PROGRAM 1 (SAVE CHECKPOINTS)")
    print("="*80)
    df = load_events()
    print(f"Loaded events: {len(df):,}")

    corpus = build_pretrain_corpus(df, n_blocks=2000)
    sft_pairs = build_sft_pairs(df, n_pairs=400)
    sft_texts = [f"INSTRUCTION: {x['instruction']} <sep> EVENT: {x['event']} <sep> RESPONSE: {x['response']}" for x in sft_pairs]
    prefs = build_preferences(sft_pairs, n_pref=400)

    all_texts = corpus + sft_texts + \
        ["PROMPT: " + x["prompt"] + " <sep> " + x["preferred"] for x in prefs] + \
        ["PROMPT: " + x["prompt"] + " <sep> " + x["dispreferred"] for x in prefs]
    vocab, stoi, itos = build_vocab(all_texts, max_vocab=9000)

    (CKPT / "vocab.json").write_text(json.dumps({"vocab": vocab}))
    print(f"Vocab size: {len(vocab):,}")

    print("\n" + "="*80)
    print("STAGE 1: PRETRAINING")
    print("="*80)
    lm = TinyCausalLM(vocab_size=len(vocab))
    pre_losses = train_lm(lm, stoi, corpus, epochs=3, lr=3e-3, batch_size=32, max_len=140)

    print("\n" + "="*80)
    print("STAGE 2: SFT")
    print("="*80)
    sft_losses = train_lm(lm, stoi, sft_texts, epochs=5, lr=2e-3, batch_size=16, max_len=220)
    torch.save(lm.state_dict(), CKPT / "sft_lm.pt")

    print("\n" + "="*80)
    print("STAGE 3: REWARD MODEL")
    print("="*80)
    rm = RewardModel(vocab_size=len(vocab))
    rm_losses, rm_accs, rm_margins = train_rm(rm, stoi, prefs, epochs=5, lr=2e-3)
    torch.save(rm.state_dict(), CKPT / "reward_model.pt")

    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(2,2,1); ax1.plot(pre_losses); ax1.set_title("Pretraining Loss")
    ax2 = fig.add_subplot(2,2,2); ax2.plot(sft_losses); ax2.set_title("SFT Loss")
    ax3 = fig.add_subplot(2,2,3); ax3.plot(rm_losses); ax3.set_title("RM Loss")
    ax4 = fig.add_subplot(2,2,4); ax4.plot(rm_accs, label="Acc%"); ax4.plot(rm_margins, label="Margin"); ax4.legend(); ax4.set_title("RM Metrics")
    fig.tight_layout()
    out_png = OUT_DIR / "program1_saved_metrics.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print("\nSaved checkpoints:")
    print(f" - {CKPT / 'vocab.json'}")
    print(f" - {CKPT / 'sft_lm.pt'}")
    print(f" - {CKPT / 'reward_model.pt'}")
    print(f"Saved plot: {out_png}")

if __name__ == "__main__":
    main()