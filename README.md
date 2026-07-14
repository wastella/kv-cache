# kv/cache

An interactive, minimalist explainer of KV caches and KV cache compression
algorithms, built with plain HTML/CSS/JS — no build step, no server.

## Structure

```
index.html              home
about.html
contact.html            update the placeholder email/GitHub links before publishing
basics.html             what a KV cache is, with a prefill/decode growth demo
compression/
  index.html            method list, grouped into subsections:
                         inference-time, architecture-level, systems, evaluation
  streaming-llm.html    inference-time
  h2o.html               inference-time
  snapkv.html            inference-time
  knorm.html             inference-time
  pyramidkv.html         inference-time
  tova.html              inference-time
  think.html             inference-time
  quantized-cache.html   inference-time
  gqa-mqa.html           architecture-level
  mla.html               architecture-level
  paged-attention.html   systems
  evaluation.html        evaluation
css/style.css           shared design system
js/sandbox.js           shared interactive demo component + scoring logic
```

## Run locally

Any static file server works, e.g.:

```
python3 -m http.server 8000
```

then open `http://localhost:8000`.

## Deploy to GitHub Pages (free)

1. Push this directory to a GitHub repo.
2. Repo Settings → Pages → Source: "Deploy from a branch" → `main` → `/ (root)`.
3. Site publishes at `https://<username>.github.io/<repo>/`.

`.nojekyll` is already included so GitHub doesn't run its Jekyll build step
(not needed for a plain static site, and avoids it mangling files that start
with an underscore).

## Adding a new inference-time method

Copy `compression/streaming-llm.html` as a starting point. Each page:

1. Explains the method's intuition in prose.
2. Defines a `compute(tokens, params)` function returning `{ kept, scores, tags }`.
3. Calls `createCompressionDemo({...})` from `js/sandbox.js`.
4. Links to the real source file in NVIDIA/kvpress and the original paper.

Then add a row for it under the "Inference-time methods" subsection on
`compression/index.html` and on `index.html`.
