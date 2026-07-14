"""
Regenerates js/gpt2-basics-data.js: runs a real GPT-2 (124M, "gpt2" on
Hugging Face) forward pass over the site's sample sentence and extracts the
actual Q/K/V vectors computed at a curated set of layers and heads for each
real BPE token — no simulation, no hashing. Used by the basics.html
cache-growth demo's layer/head switcher.

Requires torch + transformers (not part of the site itself, which stays
dependency-free — this only runs offline to produce a static data file).

Usage: python3 tools/gen_gpt2_qkv.py
"""
import json
from pathlib import Path

import torch
from transformers import GPT2Model, GPT2Tokenizer

MODEL_NAME = "gpt2"
SENTENCE = (
    "the transformer stores a key and value vector for every token it has "
    "already processed so it never has to recompute attention over the full "
    "prompt again during generation"
)
DIMS_SHOWN = 4  # how many of head_dim's 64 components to keep for display
LAYERS = [0, 5, 11]  # early / middle / late (gpt2 small has 12 blocks, 0-11)
HEADS = [0, 4, 8]  # a spread across the 12 heads per layer
OUT_PATH = Path(__file__).resolve().parent.parent / "js" / "gpt2-basics-data.js"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2Model.from_pretrained(MODEL_NAME)
model.eval()

captured = {}  # layer_idx -> raw c_attn output (seq, 3*n_embd)

def make_hook(layer_idx):
    def hook(module, inputs, output):
        captured[layer_idx] = output.detach()
    return hook

handles = [model.h[l].attn.c_attn.register_forward_hook(make_hook(l)) for l in LAYERS]

enc = tokenizer(SENTENCE, return_tensors="pt")
with torch.no_grad():
    model(**enc)

for h in handles:
    h.remove()

n_embd = model.config.n_embd  # 768
head_dim = n_embd // model.config.n_head  # 64

token_ids = enc["input_ids"][0].tolist()
tokens = [tokenizer.decode([tid]) for tid in token_ids]

# qkv_by_key["layer:head"] = [ {q,k,v} per token ]
qkv_by_key = {}
for layer in LAYERS:
    qkv = captured[layer][0]  # (seq, 3*n_embd)
    q_all, k_all, v_all = qkv.split(n_embd, dim=1)  # each (seq, n_embd)
    for head in HEADS:
        # GPT-2 reshapes n_embd into (num_heads, head_dim) preserving column
        # order, so head h occupies columns [h*head_dim : (h+1)*head_dim].
        start = head * head_dim
        end = start + DIMS_SHOWN
        records = []
        for i in range(len(token_ids)):
            records.append({
                "q": [round(x, 4) for x in q_all[i, start:end].tolist()],
                "k": [round(x, 4) for x in k_all[i, start:end].tolist()],
                "v": [round(x, 4) for x in v_all[i, start:end].tolist()],
            })
        qkv_by_key[f"{layer}:{head}"] = records


def fmt_nums(nums):
    return "[" + ", ".join(f"{n:.4f}" for n in nums) + "]"


lines = [
    "/*",
    f" * Real Q/K/V vectors from GPT-2 (124M params, \"{MODEL_NAME}\" on Hugging",
    f" * Face) — extracted from an actual forward pass over the sentence below,",
    f" * not simulated, at a curated set of layers ({LAYERS}) and heads",
    f" * ({HEADS}) out of the model's 12 layers x 12 heads ({DIMS_SHOWN} of",
    f" * {head_dim} head_dim components shown per vector). Regenerate with",
    " * tools/gen_gpt2_qkv.py. Tokens are GPT-2's real BPE tokens, which is",
    " * why \"recompute\" splits into two pieces below.",
    " *",
    " * sentence: \"the transformer stores a key and value vector for every",
    " * token it has already processed so it never has to recompute attention",
    " * over the full prompt again during generation\"",
    " */",
    "const GPT2_TOKENS = [" + ", ".join(json.dumps(t) for t in tokens) + "];",
    "",
    f"const GPT2_LAYERS = {LAYERS};",
    f"const GPT2_HEADS = {HEADS};",
    "",
    "// keyed \"layer:head\" -> one { q, k, v } record per token, same order as GPT2_TOKENS",
    "const GPT2_QKV = {",
]
for layer in LAYERS:
    for head in HEADS:
        key = f"{layer}:{head}"
        lines.append(f'  "{key}": [')
        for rec in qkv_by_key[key]:
            lines.append(f'    {{ q: {fmt_nums(rec["q"])}, k: {fmt_nums(rec["k"])}, v: {fmt_nums(rec["v"])} }},')
        lines.append("  ],")
lines.append("};")
lines.append("")
lines.append(
    f"const GPT2_META = {{ model: {json.dumps(MODEL_NAME)}, headDim: {head_dim}, dimsShown: {DIMS_SHOWN} }};"
)
lines.append("")

OUT_PATH.write_text("\n".join(lines))
print(f"wrote {len(LAYERS) * len(HEADS)} layer/head combos x {len(tokens)} tokens to {OUT_PATH}")
print("tokens:", tokens)
