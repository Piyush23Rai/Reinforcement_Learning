# FinTech LLM Training — Program 2 (DPO) + Streamlit UI

This package continues from Program 1 and demonstrates **Policy Optimization** using **DPO** (Direct Preference Optimization)
driven by a Reward Model.

## What you get
- `program1_train_and_save.py` — trains a toy Pretrain → SFT → Reward Model and **saves checkpoints**
- `program2_dpo_train.py` — generates candidate answers, scores them with RM, and runs **DPO**
- `app.py` — Streamlit UI to explore events, generate candidates, view RM scores, and compare **SFT vs DPO policy**

> Note: This is a teaching-grade GRU LM (not a transformer). The purpose is to make the RLHF pipeline tangible.

---

## 0) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Put your CSV
Place your file at:
- `data/fintech_events.csv`

Expected columns:
event_time,merchant_id,txn_id,amount,currency,country,payment_method,auth_result,error_code,issuer_response,fraud_score,avs_result,three_ds,chargeback_flag

## 2) Train Program 1 models and save checkpoints
```bash
python program1_train_and_save.py
```

Saves:
- `out/checkpoints/vocab.json`
- `out/checkpoints/sft_lm.pt`
- `out/checkpoints/reward_model.pt`

## 3) Run Program 2 (DPO)
```bash
python program2_dpo_train.py --steps 200 --k 6
```

Saves:
- `out/checkpoints/dpo_policy.pt`
- `out/dpo_metrics.png`

## 4) Launch UI
```bash
streamlit run app.py
```

---

## Teaching storyline
1) Generate **multiple candidate answers** for the same payments incident prompt.
2) Score them with the **Reward Model** (higher = preferred: safer, more actionable, compliance-aware).
3) Run **DPO** to nudge the policy toward preferred answers while staying near the SFT reference.

If students ask “Where is PPO?”:
- PPO needs rollouts + advantage estimation + explicit KL control.
- DPO is a simpler offline alternative that still demonstrates: **optimize policy with preferences while staying close to a reference**.