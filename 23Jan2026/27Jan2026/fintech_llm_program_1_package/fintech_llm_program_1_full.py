import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# FINTECH LLM TRAINING PIPELINE - PROGRAM 1 (End-to-End)
# Stages: Pretraining → SFT → Reward Model Training
# ============================================================

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

EVENTS_CSV = DATA_DIR / "fintech_events.csv"

# ----------------------------
# 1) Text rendering (structured → narrative)
# ----------------------------
def event_to_text(row) -> str:
    # A compact "narrative" that resembles what ops/support sees.
    parts = [
        f"time={row['event_time']}",
        f"merchant={row['merchant_id']}",
        f"country={row['country']}",
        f"amount={row['amount']} {row['currency']}",
        f"auth={row['auth_result']}",
    ]
    if str(row.get("error_code","")):
        parts.append(f"error={row['error_code']}")
    if str(row.get("issuer_response","")):
        parts.append(f"issuer={row['issuer_response']}")
    parts.append(f"fraud_score={int(row['fraud_score'])}")
    parts.append(f"avs={row['avs_result']}")
    parts.append(f"3ds={row['three_ds']}")
    parts.append(f"chargeback={int(row['chargeback_flag'])}")
    return " | ".join(parts)

def load_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {EVENTS_CSV}. Put your generated data there.\n"
            f"Expected: {EVENTS_CSV}"
        )
    df = pd.read_csv(EVENTS_CSV)
    # minimal sanitation
    for c in ["event_time","merchant_id","txn_id","currency","country","auth_result","error_code","issuer_response","avs_result","three_ds"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    for c in ["amount","fraud_score","chargeback_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# ----------------------------
# 2) Build datasets
# ----------------------------
def build_pretrain_corpus(df: pd.DataFrame, n_lines=8000) -> List[str]:
    # Sample events, convert to text, shuffle.
    lines = [event_to_text(df.iloc[i]) for i in np.random.randint(0, len(df), size=n_lines)]
    # Add a bit of variety by grouping "mini incidents"
    corpus = []
    for i in range(0, len(lines), 4):
        block = " \n".join(lines[i:i+4])
        corpus.append(block)
    return corpus

def build_sft_pairs(df: pd.DataFrame, n_pairs=400) -> List[Dict]:
    # Create supervised pairs: instruction → expert response (templated but realistic).
    pairs = []
    sample = df.sample(n=min(len(df), n_pairs), random_state=SEED)
    for _, r in sample.iterrows():
        instr = (
            "You are a Payments Ops assistant. Summarize the event and recommend safe next actions. "
            "Do not include PAN/PII. Use bullet points."
        )
        # "Expert" response uses rule-based reasoning on observed fields (teaching-friendly).
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
            actions += [
                "Consider stricter AVS/zip match or step-up for AVS mismatch traffic."
            ]
        if int(r["chargeback_flag"]) == 1:
            actions += [
                "Flag as potential dispute: gather evidence (3DS result, device/IP, delivery proof if applicable)."
            ]
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
        pairs.append({"instruction": instr, "response": resp})
    return pairs

def build_preference_pairs(sft_pairs: List[Dict], n_pref=400) -> List[Dict]:
    # Make preference data by generating a "bad" response that violates safety/quality.
    prefs = []
    sample = random.sample(sft_pairs, k=min(n_pref, len(sft_pairs)))
    for ex in sample:
        prompt = ex["instruction"]
        preferred = ex["response"]
        # Dispreferred: unsafe + less actionable
        dispreferred = (
            "Just disable fraud rules to improve approvals. "
            "Store full card numbers for matching. "
            "Retry aggressively until it goes through."
        )
        prefs.append({"prompt": prompt, "preferred": preferred, "dispreferred": dispreferred})
    return prefs

# ----------------------------
# 3) Tokenizer (simple wordpiece-ish)
# ----------------------------
SPECIAL = ["<pad>", "<bos>", "<eos>", "<sep>"]

def build_vocab(texts, max_vocab=8000):
    freq = {}
    for t in texts:
        for w in t.replace("\n", " ").split():
            freq[w] = freq.get(w, 0) + 1
    vocab = SPECIAL.copy()
    for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        if w in SPECIAL:
            continue
        vocab.append(w)
        if len(vocab) >= max_vocab:
            break
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return vocab, stoi, itos

def encode(text, stoi, max_len=96):
    toks = text.replace("\n", " ").split()
    ids = [stoi["<bos>"]] + [stoi.get(w, stoi["<pad>"]) for w in toks][:max_len-2] + [stoi["<eos>"]]
    return ids

def pad_batch(batch_ids, pad_id=0):
    maxlen = max(len(x) for x in batch_ids)
    x = torch.full((len(batch_ids), maxlen), pad_id, dtype=torch.long)
    m = torch.zeros((len(batch_ids), maxlen), dtype=torch.bool)
    for i, ids in enumerate(batch_ids):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        m[i, :len(ids)] = 1
    return x, m

# ----------------------------
# 4) Models (tiny causal LM + Reward Model)
# ----------------------------
class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)
        logits = self.lm_head(h)
        return logits

class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        e = self.emb(x)
        h, _ = self.rnn(e)
        # take last valid hidden state
        lengths = mask.long().sum(dim=1) - 1
        last = h[torch.arange(h.size(0)), lengths.clamp(min=0)]
        r = self.head(last).squeeze(-1)
        return r

# ----------------------------
# 5) Training loops
# ----------------------------
def train_pretrain(model, stoi, corpus, epochs=3, lr=3e-3, batch_size=32):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    losses = []

    for ep in range(1, epochs+1):
        random.shuffle(corpus)
        running = 0.0
        n = 0
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            ids = [encode(t, stoi) for t in batch]
            x, _ = pad_batch(ids, pad_id=stoi["<pad>"])
            # teacher forcing: predict next token
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            n += 1
        avg = running / max(n, 1)
        losses.append(avg)
        print(f"  Epoch {ep}/{epochs}: Loss = {avg:.4f}")
    return losses

def build_sft_sequences(sft_pairs):
    seqs = []
    for ex in sft_pairs:
        t = f"INSTRUCTION: {ex['instruction']} <sep> RESPONSE: {ex['response']}"
        seqs.append(t)
    return seqs

def train_sft(model, stoi, sft_seqs, epochs=5, lr=2e-3, batch_size=16):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    losses = []

    for ep in range(1, epochs+1):
        random.shuffle(sft_seqs)
        running = 0.0
        n = 0
        for i in range(0, len(sft_seqs), batch_size):
            batch = sft_seqs[i:i+batch_size]
            ids = [encode(t, stoi, max_len=140) for t in batch]
            x, _ = pad_batch(ids, pad_id=stoi["<pad>"])
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            n += 1
        avg = running / max(n, 1)
        losses.append(avg)
        if ep % 1 == 0:
            print(f"  Epoch {ep}/{epochs}: Loss = {avg:.4f}")
    return losses

def rm_batchify(prefs, stoi, batch_size=16):
    for i in range(0, len(prefs), batch_size):
        chunk = prefs[i:i+batch_size]
        pref_ids = [encode("PROMPT: " + ex["prompt"] + " <sep> " + ex["preferred"], stoi, max_len=160) for ex in chunk]
        disp_ids = [encode("PROMPT: " + ex["prompt"] + " <sep> " + ex["dispreferred"], stoi, max_len=160) for ex in chunk]
        x1, m1 = pad_batch(pref_ids, pad_id=stoi["<pad>"])
        x2, m2 = pad_batch(disp_ids, pad_id=stoi["<pad>"])
        yield x1, m1, x2, m2

def train_reward_model(rm, stoi, prefs, epochs=5, lr=2e-3, batch_size=16):
    opt = optim.Adam(rm.parameters(), lr=lr)
    losses, accs, margins = [], [], []

    for ep in range(1, epochs+1):
        random.shuffle(prefs)
        running_loss = 0.0
        running_acc = 0.0
        running_margin = 0.0
        n = 0
        for x1, m1, x2, m2 in rm_batchify(prefs, stoi, batch_size=batch_size):
            r1 = rm(x1, m1)
            r2 = rm(x2, m2)
            # Bradley-Terry: -log σ(r1 - r2)
            diff = r1 - r2
            loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                acc = (diff > 0).float().mean().item() * 100.0
                margin = diff.mean().item()

            running_loss += loss.item()
            running_acc += acc
            running_margin += margin
            n += 1

        avg_loss = running_loss / max(n, 1)
        avg_acc = running_acc / max(n, 1)
        avg_margin = running_margin / max(n, 1)
        losses.append(avg_loss); accs.append(avg_acc); margins.append(avg_margin)
        print(f"  Epoch {ep:>2}: Loss={avg_loss:.4f}, Margin={avg_margin:+.3f}, Acc={avg_acc:.1f}%")
    return losses, accs, margins

# ----------------------------
# 6) Main
# ----------------------------
def main():
    print("="*80)
    print("FINTECH LLM TRAINING PIPELINE - PROGRAM 1")
    print("Stages: Pretraining → SFT → Reward Model Training")
    print("="*80)

    df = load_events()
    print(f"\nLoaded events: {len(df):,}")

    # Build datasets
    corpus = build_pretrain_corpus(df, n_lines=min(8000, max(2000, len(df)*2)))
    sft_pairs = build_sft_pairs(df, n_pairs=min(400, max(120, len(df)//5)))
    prefs = build_preference_pairs(sft_pairs, n_pref=min(400, len(sft_pairs)))

    # Vocab from all texts used
    all_texts = corpus + [f"INSTRUCTION: {x['instruction']} <sep> RESPONSE: {x['response']}" for x in sft_pairs] \
               + ["PROMPT: " + x["prompt"] + " <sep> " + x["preferred"] for x in prefs] \
               + ["PROMPT: " + x["prompt"] + " <sep> " + x["dispreferred"] for x in prefs]
    vocab, stoi, itos = build_vocab(all_texts, max_vocab=9000)

    print(f"\nVocab size: {len(vocab):,}")

    # ----------------------------
    # Stage 1: Pretraining
    # ----------------------------
    print("\n" + "="*80)
    print("STAGE 1: PRETRAINING - NEXT TOKEN PREDICTION")
    print("="*80)
    print("\nPretraining learns: Context → Next Token (payments language patterns)")
    print("Mathematical: Loss = -Σ log P(w_t | w_<t)")

    lm = TinyCausalLM(vocab_size=len(vocab))
    pre_losses = train_pretrain(lm, stoi, corpus, epochs=4)

    # ----------------------------
    # Stage 2: SFT
    # ----------------------------
    print("\n" + "="*80)
    print("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
    print("="*80)
    print("\nSFT learns: Instruction → Expert Response")
    print("Mathematical (concept): L_SFT = -E[Σ log π(expert_token | context)]")

    sft_seqs = build_sft_sequences(sft_pairs)
    sft_losses = train_sft(lm, stoi, sft_seqs, epochs=6)

    # ----------------------------
    # Stage 3: Reward Model
    # ----------------------------
    print("\n" + "="*80)
    print("STAGE 3: REWARD MODEL TRAINING (Preference Ranking)")
    print("="*80)
    print("\nRM learns: preference ranking (safer/compliant responses score higher)")
    print("Mathematical: Loss = -log σ(r_pref - r_dispref)")
    print(f"\nTraining on {len(prefs)} preference pairs...")

    rm = RewardModel(vocab_size=len(vocab))
    rm_losses, rm_accs, rm_margins = train_reward_model(rm, stoi, prefs, epochs=6)

    # ----------------------------
    # Visualize
    # ----------------------------
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(range(1, len(pre_losses)+1), pre_losses)
    ax1.set_title("Pretraining Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(range(1, len(sft_losses)+1), sft_losses)
    ax2.set_title("SFT Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(range(1, len(rm_losses)+1), rm_losses)
    ax3.set_title("Reward Model Loss")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(range(1, len(rm_accs)+1), rm_accs, label="Accuracy (%)")
    ax4.plot(range(1, len(rm_margins)+1), rm_margins, label="Margin (pref - dispref)")
    ax4.set_title("RM Metrics")
    ax4.set_xlabel("Epoch")
    ax4.legend()

    out_png = OUT_DIR / "fintech_program_1_results.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n" + "="*80)
    print("PROGRAM 1 SUMMARY (FINTECH)")
    print("="*80)
    print("\n✓ PRETRAINING:")
    print(f"  - Corpus blocks: {len(corpus)}")
    print("  - Learned next-token patterns from payments narratives")
    print("  - Problem: no value/compliance alignment by itself")

    print("\n✓ SFT:")
    print(f"  - Expert pairs: {len(sft_pairs)}")
    print("  - Learned instruction-following and structured ops responses")
    print("  - Problem: capped by expert template quality/coverage")

    print("\n✓ REWARD MODEL:")
    print(f"  - Preference pairs: {len(prefs)}")
    print("  - Bradley-Terry loss training")
    print(f"  - Final Loss: {rm_losses[-1]:.4f}")
    print(f"  - Final Accuracy: {rm_accs[-1]:.1f}%")
    print(f"  - Final Margin: {rm_margins[-1]:+.3f}")

    print("\n" + "="*80)
    print("✓ PROGRAM 1 COMPLETE")
    print(f"✓ Saved plot: {out_png}")
    print("→ Next (Program 2): Policy Optimization (PPO/DPO) using the Reward Model")
    print("="*80)

if __name__ == "__main__":
    main()