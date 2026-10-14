"""
Regenerates js/gpt2-niah-data.js: runs a real GPT-2 (124M, "gpt2" on
Hugging Face) forward pass over three filler passages and extracts, per
token, the actual signal each real eviction policy scores it by:

  - KNorm (Devoto et al., 2024): L2 norm of the token's real key vector at
    a middle layer. The real algorithm keeps the LOWEST-norm keys.
  - SnapKV (Li et al., 2024): real attention weight each token receives
    from the last `window_size` queries, averaged over heads and query
    positions at that same layer. The real algorithm always keeps the
    observation window itself, plus the top-scoring tokens outside it.

No per-trial randomness, no reverse-engineered curve — the eviction mask
evaluation.html computes client-side from this file's numbers is the
actual KNorm/SnapKV selection rule applied to real GPT-2 internals.
Only the layer choice (one representative middle layer, matching
js/gpt2-basics-data.js's convention) and SnapKV's real smoothing max-pool
(skipped for a cleaner single-number-per-token demo) are simplified.

Requires torch + transformers. Run inside an isolated venv:
  python3 -m venv tools/.venv
  tools/.venv/bin/pip install torch transformers
  tools/.venv/bin/python tools/gen_gpt2_niah.py
"""
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"
LAYER = 5  # middle layer, same convention as js/gpt2-basics-data.js
SNAPKV_WINDOW = 8
TOKEN_COUNT = 48  # fixed length so the needle-depth slider means the same thing across samples
OUT_PATH = Path(__file__).resolve().parent.parent / "js" / "gpt2-niah-data.js"

PASSAGES = [
    {
        "id": "novel",
        "text": (
            "It was a cold and grey November morning when the letter arrived "
            "at Baker Street, carrying with it the unmistakable scent of "
            "trouble that Holmes had come to recognize instantly, folding his "
            "newspaper and setting down his pipe before the envelope had even "
            "been fully opened by the maid who brought it up the narrow stairs."
        ),
    },
    {
        "id": "manual",
        "text": (
            "To configure the cache eviction policy, first determine the "
            "maximum sequence length supported by the model, then select a "
            "compression ratio appropriate for the available GPU memory, "
            "keeping in mind that overly aggressive ratios can silently "
            "degrade retrieval quality without producing any visible warning "
            "during ordinary validation runs on short held-out prompts."
        ),
    },
    {
        "id": "news",
        "text": (
            "Officials announced on Tuesday that the new transit line "
            "connecting the eastern suburbs to the downtown corridor would "
            "begin operations ahead of schedule, following months of testing "
            "delays caused primarily by unexpected supply chain disruptions "
            "affecting the signaling equipment used throughout the network's "
            "older stations."
        ),
    },
]

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
model.eval()

captured = {}

def make_hook(layer_idx):
    def hook(module, inputs, output):
        captured[layer_idx] = output.detach()
    return hook

handle = model.transformer.h[LAYER].attn.c_attn.register_forward_hook(make_hook(LAYER))

results = []
for spec in PASSAGES:
    enc = tokenizer(spec["text"], return_tensors="pt")
    ids = enc["input_ids"][0][:TOKEN_COUNT]
    enc = {"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones(1, len(ids))}

    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    n_embd = model.config.n_embd
    qkv = captured[LAYER][0]  # (seq, 3*n_embd)
    _, k_all, _ = qkv.split(n_embd, dim=1)  # (seq, n_embd), full concatenated key vector per token
    knorm_scores = k_all.norm(dim=-1).tolist()  # real L2 norm per token

    attn = out.attentions[LAYER][0]  # (heads, seq, seq)
    seq_len = attn.shape[-1]
    window = min(SNAPKV_WINDOW, seq_len)
    # mean attention paid to each key position by the last `window` queries, averaged over heads
    obs = attn[:, seq_len - window:, :]  # (heads, window, seq)
    snapkv_scores = obs.mean(dim=(0, 1)).tolist()  # real attention-weight importance per token

    tokens = [tokenizer.decode([t]) for t in ids.tolist()]
    results.append({
        "id": spec["id"],
        "tokens": tokens,
        "knormScore": [round(s, 5) for s in knorm_scores],
        "snapkvScore": [round(s, 6) for s in snapkv_scores],
    })

handle.remove()

lines = [
    "/*",
    f" * Real per-token eviction signal from GPT-2 (124M params, \"{MODEL_NAME}\" on",
    " * Hugging Face) — an actual forward pass over three filler passages, not",
    " * simulated. `knormScore` is the real L2 norm of each token's key vector",
    f" * at layer {LAYER} (KNorm keeps the LOWEST-norm tokens). `snapkvScore` is",
    f" * the real attention weight each token received from the last",
    f" * {SNAPKV_WINDOW} queries' attention, averaged over heads (SnapKV always keeps",
    " * that observation window, plus the top-scoring tokens outside it).",
    " * Regenerate with tools/gen_gpt2_niah.py.",
    " */",
    f"const GPT2_NIAH_WINDOW = {SNAPKV_WINDOW};",
    "const GPT2_NIAH_HAYSTACKS = " + json.dumps(results, indent=2) + ";",
    "",
]

OUT_PATH.write_text("\n".join(lines))
for r in results:
    print(f"{r['id']:>8}: {len(r['tokens'])} tokens")
print(f"wrote {len(results)} haystacks to {OUT_PATH}")
