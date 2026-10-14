"""
Regenerates js/gpt2-perplexity-data.js: runs a real GPT-2 (124M, "gpt2" on
Hugging Face) forward pass over three sample sentences and extracts the
model's actual next-token probability at every position, plus its top-3
guesses — no simulation, no reverse-engineered averages. Used by the
evaluation.html perplexity demo, so a reader can see for themselves why an
iconic phrase, a technical sentence, and a word-salad sentence land at very
different perplexities.

Requires torch + transformers. Run inside an isolated venv:
  python3 -m venv tools/.venv
  tools/.venv/bin/pip install torch transformers
  tools/.venv/bin/python tools/gen_gpt2_perplexity.py
"""
import json
import math
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"
TOP_K = 3
OUT_PATH = Path(__file__).resolve().parent.parent / "js" / "gpt2-perplexity-data.js"

SENTENCES = [
    {
        "id": "iconic",
        "label": "iconic phrase",
        "text": "The quick brown fox jumps over the lazy dog.",
    },
    {
        "id": "technical",
        "label": "technical sentence",
        "text": "The transformer stores a key and value vector for every token.",
    },
    {
        "id": "nonsense",
        "label": "word salad",
        "text": "Purple elephants calculate Tuesday inside the mathematics of jazz.",
    },
]

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

results = []
for spec in SENTENCES:
    enc = tokenizer(spec["text"], return_tensors="pt")
    ids = enc["input_ids"][0]
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0]  # (seq, vocab)
    probs = torch.softmax(logits, dim=-1)  # (seq, vocab)

    tokens = [tokenizer.decode([t]) for t in ids.tolist()]
    n = len(tokens)

    steps = [None]  # first token has no preceding context to predict it
    surprisal_sum = 0.0
    surprisal_n = 0
    for i in range(n - 1):
        pos_probs = probs[i]  # distribution predicting token i+1
        actual_id = ids[i + 1].item()
        actual_p = pos_probs[actual_id].item()
        topk = torch.topk(pos_probs, TOP_K)
        rank = int((pos_probs > actual_p).sum().item()) + 1
        surprisal = -math.log(actual_p)
        surprisal_sum += surprisal
        surprisal_n += 1
        steps.append({
            "surprisal": round(surprisal, 4),
            "actualProb": round(actual_p, 6),
            "rank": rank,
            "top": [
                {"tok": tokenizer.decode([tid.item()]), "p": round(p.item(), 6)}
                for tid, p in zip(topk.indices, topk.values)
            ],
        })

    mean_surprisal = surprisal_sum / surprisal_n
    results.append({
        "id": spec["id"],
        "label": spec["label"],
        "text": spec["text"],
        "tokens": tokens,
        "steps": steps,
        "perplexity": round(math.exp(mean_surprisal), 4),
    })


def js_str(s):
    return json.dumps(s)


lines = [
    "/*",
    f" * Real next-token probabilities from GPT-2 (124M params, \"{MODEL_NAME}\" on",
    " * Hugging Face) — an actual forward pass over three sentences below, not",
    " * simulated. For every token, `steps[i]` gives the model's real",
    " * probability for the token that actually came next, its rank among all",
    " * ~50K vocab entries, and its top-3 guesses. `perplexity` is",
    " * exp(mean(-ln actualProb)) computed directly from those numbers — not",
    " * reverse-engineered from a published constant. Regenerate with",
    " * tools/gen_gpt2_perplexity.py.",
    " */",
    "const GPT2_PPX_SENTENCES = " + json.dumps(results, indent=2) + ";",
    "",
]

OUT_PATH.write_text("\n".join(lines))
for r in results:
    print(f"{r['id']:>10}: perplexity={r['perplexity']:.2f}  \"{r['text']}\"")
print(f"wrote {len(results)} sentences to {OUT_PATH}")
