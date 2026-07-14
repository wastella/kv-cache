# Status — kv/cache project

All planned content and QA work is complete as of this session. This file is kept as a
record of what was done and what's still a deliberate open item — not an active punch list.

## What exists

12 topic pages under `compression/` (8 kvpress inference-time presses + 4
architecture/systems/evaluation pages, unified into one nav tab and index page
with subsections — previously split across separate `catalogue/`/`concepts/`
directories), `index.html`, `about.html`, `basics.html`, `contact.html`,
shared `css/style.css` + `js/sandbox.js`. Full structure and design-system
notes are in `README.md`.

## What was done this session

1. Added 5 catalogue pages (Knorm, PyramidKV, TOVA, ThinK, quantized cache) and 4 concepts
   pages (GQA/MQA, MLA, PagedAttention, evaluation methodology), each independently verified
   against real kvpress source / papers / model configs while being written.
2. Site-wide QA pass: mobile nav overflow, slider accessibility, heading hierarchy, dead
   "three methods" copy — fixed.
3. Wired a new `concepts/` nav item into all 13 pre-existing pages; added an `about.html`
   section explaining why concepts pages exist outside the catalogue.
4. **Per-entry independent review pass** (12 reviewer agents, one per page, each fixing what
   it found rather than just reporting): caught and corrected multiple real errors —
   a rounding-mode bug in PyramidKV's budget-formula JS (Python round-half-to-even vs JS
   round-half-up), a mischaracterized scoring mechanism in TOVA, a misattributed quantization
   scheme in the quantized-cache page, a wrong baseline architecture claim in MLA (DeepSeek
   67B already used GQA, not plain MHA) plus a fabricated code-comment quote, a
   misstated RULER headline finding in evaluation.html, a dead GitHub source link and a
   demo that silently exceeded its own stated capacity in PagedAttention, plus assorted
   AI-sounding prose tightened throughout. Also fixed a shared bug in `js/sandbox.js`
   (slider value label didn't apply its `format()` function until first interaction).
5. Mobile-responsiveness spot check (375px) across all custom demos (PyramidKV's bar chart,
   ThinK's channel grid, GQA/MLA's boxes, PagedAttention's memory blocks, evaluation's SVG
   charts) — all render without horizontal overflow; the evaluation charts are compact but
   legible at that width.
6. Found and fixed a pre-existing (not from this session's new pages) HTML structural bug in
   `basics.html`: a `<div class="note">` was closed with `</p>` instead of `</div>`. Also
   updated its stale "three different methods" copy to "eight."
7. Final integration check: every internal link/script reference across the whole site
   resolves, and every page's HTML tags are now balanced (verified programmatically).

## Later session: merged catalogue/concepts into one `compression/` section

The `catalogue`/`concepts` nav split read as arbitrary (site owner's call —
both are ways to shrink the KV cache, just at different stages). Merged both
into a single `compression/` directory and one `compression/index.html` with
four `<h2>` subsections (inference-time, architecture-level, systems,
evaluation) instead of two separate index pages. All 12 detail pages moved
with `git mv` (content unchanged, only nav/breadcrumb/meta URLs updated).
Old `catalogue/*`/`concepts/*` URLs now 404 — intentional, no redirects added
(site owner confirmed breaking old links was fine).

## Repo and hosting

Live at git repo `wastella/kv-cache` (public), hosted on GitHub Pages at
`https://wastella.github.io/kv-cache/`. Pushing to `main` deploys automatically.

## Deliberately not done — these are calls for the site owner, not open bugs

- **kvpress has 40+ presses total; this catalogue covers 8**, spanning the major families
  (position, attention×2, prediction, norm, layer-wise, dimension-pruning, quantization).
  Adding more is a content decision, not a defect — see `about.html`'s "Why these methods"
  section for the current rationale, and `compression/index.html`'s note for what's explicitly
  flagged as still missing (merging/low-rank methods like CaM/KVzip).

## Conventions to preserve if extending this site later

- Every claim cites a primary source (arXiv ID, real repo file, real model config) — no
  invented statistics. Where a number can't be verified, say so or hedge, don't state it flatly.
- Every simulated/illustrative demo value gets an explicit `.note` callout admitting it's
  simulated, not measured (canonical phrasing in `compression/knorm.html`).
- New CSS for bespoke demos goes at the end of `css/style.css` with a scoped class prefix
  (`.layer-*`, `.channel-*`, `.mem-*`, `.chart-*`) — never edit/restyle existing rules other
  pages depend on.
- When several agents touch shared files concurrently, either give them non-overlapping
  sections explicitly, or have the orchestrating session make that one shared edit itself
  after their work lands — don't trust two agents not to collide on the same file.
