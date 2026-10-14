# kv/cache

An interactive, minimalist explainer of KV caches and KV cache compression
algorithms, built with plain HTML/CSS/JS — no build step, no server.

## Structure

```
index.html              home
about.html
contact.html
basics.html             what a KV cache is, with a prefill/decode growth demo
compression/
  index.html            method list, grouped into subsections:
                         architecture, inference-time, combinators & wrappers,
                         systems, evaluation

  # architecture (decided at training time, not applied to an existing model)
  gqa-mqa.html
  mla.html

  # inference-time (28 kvpress methods, applied to an already-trained model)
  streaming-llm.html     h2o.html               tova.html
  expected-attention.html critical-kv.html      capkv.html
  duo-attention.html     snapkv.html            finch.html
  chunkkv.html           lagkv.html             keydiff.html
  knorm.html             cur.html               kvzip.html
  restorekv.html         fast-kvzip.html        qfilter.html
  pyramidkv.html         simlayerkv.html        lukv.html
  dms.html               think.html             quantized-cache.html
  compactor.html         cam.html               merging.html
  kvcompose.html

  # combinators & wrappers: one grouped note on index.html, no dedicated
  # pages (AdaKV, ComposedPress, ChunkPress/BlockPress, KeyRerotationPress,
  # PerLayerCompressionPress, the DecodingPress family, RandomPress)

  paged-attention.html   systems
  evaluation.html        evaluation
css/style.css            shared design system
js/sandbox.js            shared interactive demo component + scoring logic
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

Copy `compression/knorm.html` (a plain slider-grid demo) or a bespoke page
like `compression/tova.html` as a starting point, depending on whether the
method fits the generic scoring-and-eviction pattern. Each page:

1. Explains the method's intuition in prose, citing the real paper (arXiv ID)
   and, where relevant, the real kvpress source file it's based on.
2. Defines a `compute(tokens, params)` function returning `{ kept, scores, tags }`,
   written to mirror kvpress's real scoring logic (verify against the actual
   `.py` source, don't trust a docstring or summary).
3. Calls `createCompressionDemo({...})` from `js/sandbox.js`, or a bespoke
   demo builder if the method doesn't fit that generic shape.
4. Flags any simulated/illustrative value with an explicit `.note` callout
   (see `compression/knorm.html` for the canonical phrasing).

Then add a row for it under the appropriate subsection on
`compression/index.html`. If it's one of kvpress's pure wrapper/infrastructure
classes (takes a `press:`/`base_press:` param rather than scoring anything
itself), it belongs in the "Combinators & wrappers" grouped note instead of
getting its own page — see `HANDOFF.md` for how that split was decided.
