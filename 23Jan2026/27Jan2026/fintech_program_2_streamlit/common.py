import math
from typing import List, Dict

import torch
import torch.nn as nn

SPECIAL = ["<pad>", "<bos>", "<eos>", "<sep>"]

def build_vocab(texts: List[str], max_vocab: int = 9000):
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
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos

def encode(text: str, stoi: Dict[str, int], max_len: int = 180) -> List[int]:
    toks = text.replace("\n", " ").split()
    ids = [stoi["<bos>"]] + [stoi.get(w, stoi["<pad>"]) for w in toks][: max_len - 2] + [stoi["<eos>"]]
    return ids

def pad_batch(batch_ids: List[List[int]], pad_id: int):
    maxlen = max(len(x) for x in batch_ids)
    x = torch.full((len(batch_ids), maxlen), pad_id, dtype=torch.long)
    m = torch.zeros((len(batch_ids), maxlen), dtype=torch.bool)
    for i, ids in enumerate(batch_ids):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        m[i, : len(ids)] = 1
    return x, m

class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, hidden: int = 192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor):
        e = self.emb(x)
        h, _ = self.rnn(e)
        logits = self.lm_head(h)
        return logits

class RewardModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, hidden: int = 192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        e = self.emb(x)
        h, _ = self.rnn(e)
        lengths = mask.long().sum(dim=1) - 1
        last = h[torch.arange(h.size(0)), lengths.clamp(min=0)]
        r = self.head(last).squeeze(-1)
        return r

def event_to_text(row: dict) -> str:
    parts = [
        f"time={row['event_time']}",
        f"merchant={row['merchant_id']}",
        f"country={row['country']}",
        f"amount={row['amount']} {row['currency']}",
        f"auth={row['auth_result']}",
    ]
    if str(row.get("error_code", "")):
        parts.append(f"error={row['error_code']}")
    if str(row.get("issuer_response", "")):
        parts.append(f"issuer={row['issuer_response']}")
    parts.append(f"fraud_score={int(row['fraud_score'])}")
    parts.append(f"avs={row['avs_result']}")
    parts.append(f"3ds={row['three_ds']}")
    parts.append(f"chargeback={int(row['chargeback_flag'])}")
    return " | ".join(parts)

DEFAULT_INSTRUCTION = (
    "You are a Payments Ops assistant. Summarize the event and recommend safe next actions. "
    "Do not include PAN/PII. Use bullet points."
)

def build_prompt_from_event(row: dict) -> str:
    ev = event_to_text(row)
    return f"INSTRUCTION: {DEFAULT_INSTRUCTION} <sep> EVENT: {ev} <sep> RESPONSE:"

@torch.no_grad()
def sample_response(model: TinyCausalLM, prompt_ids: List[int], stoi, itos,
                    max_new_tokens: int = 60, temperature: float = 0.9, top_k: int = 30, device: str = "cpu"):
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = prompt_ids[:]
    for _ in range(max_new_tokens):
        inp = x[:, :-1] if x.size(1) > 1 else x
        logits = model(inp)[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = torch.softmax(vals, dim=-1)
            next_rel = torch.multinomial(probs, 1).item()
            next_id = idx[0, next_rel].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
        x = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        if next_id == stoi["<eos>"]:
            break
    toks = [itos.get(i, "<unk>") for i in generated]
    return " ".join(toks)

def split_prompt_response_tokens(full_text: str):
    if "RESPONSE:" in full_text:
        i = full_text.index("RESPONSE:")
        prompt = full_text[: i + len("RESPONSE:")].strip()
        resp = full_text[i + len("RESPONSE:"):].strip()
    else:
        prompt, resp = full_text, ""
    return prompt, resp

def logprob_of_response(model: TinyCausalLM, stoi, prompt_text: str, response_text: str,
                        max_len: int = 220, device: str = "cpu"):
    model.eval()
    full = f"{prompt_text} {response_text}".strip()
    ids = encode(full, stoi, max_len=max_len)
    prompt_ids = encode(prompt_text, stoi, max_len=max_len)
    boundary = min(len(prompt_ids) - 1, len(ids) - 1)

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    inp = x[:, :-1]
    tgt = x[:, 1:]
    logits = model(inp)
    logp = torch.log_softmax(logits, dim=-1)

    start = max(boundary - 1, 0)
    lp = 0.0
    count = 0
    for t in range(start, tgt.size(1)):
        tok = tgt[0, t].item()
        if tok == stoi["<pad>"]:
            continue
        lp += logp[0, t, tok].item()
        count += 1
    return lp, count