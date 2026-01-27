# FinTech LLM Training Pipeline (Program 1) — End-to-End Modeling

This package turns your generated fintech events into:
1) **Pretraining** data (next-token prediction on "event narratives")
2) **SFT** data (instruction → expert response)
3) **Reward Model (RM)** data (preference pairs)

It trains **toy-but-real** neural models so students can see the full pipeline.

## 0) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt
```

## 1) Put your generated data
Place your events file as:
- `data/fintech_events.csv`

(Use the CSV columns: event_time, merchant_id, txn_id, amount, currency, country, auth_result, error_code,
issuer_response, fraud_score, avs_result, three_ds, chargeback_flag)

## 2) Run end-to-end pipeline
```bash
python fintech_llm_program_1_full.py
```

Outputs:
- `out/fintech_program_1_results.png`  (loss curves + RM metrics)
- prints a stage-by-stage summary similar to your pharma Program 1

## 3) Teaching flow
- Pretraining: teaches "payments language" and patterns (timeouts, do_not_honor, 3DS, AVS mismatch)
- SFT: teaches structured incident response + safe actions (no PAN/PII, compliance-safe guidance)
- RM: learns to prefer safer, more actionable answers (vs risky shortcuts)

## Notes
- This is a **toy GRU-based causal LM**, not a production transformer.
- The goal is pedagogy: show how the pipeline pieces connect.