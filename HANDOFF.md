# Status — kv/cache project

This file is kept as a record of what was done and what's still a deliberate open item —
not an active punch list.

## What exists

32 topic pages under `compression/` (28 kvpress inference-time methods + 4
architecture/systems/evaluation pages, unified into one nav tab and index page
with subsections — previously split across separate `catalogue/`/`concepts/`
directories), `index.html`, `about.html`, `basics.html`, `contact.html`,
shared `css/style.css` + `js/sandbox.js`. Full structure and design-system
notes are in `README.md`.

## 2026-08-15 session: kvpress catalogue expanded from 8 to 28 methods

Pulled the current full list of exported classes from `NVIDIA/kvpress` (44 as of this
session, up from the ~40 the site previously assumed) via `gh api`, classified each one
against its actual source (not just its docstring/paper abstract), and split them: 20
were genuine standalone methods not yet on the site, 10 were pure wrappers/infrastructure
(take a `press:`/`base_press:` param, no independent scoring idea of their own), and one
already-covered wrapper (AdaKV) plus the already-covered core 8 rounded out the count.

Added 20 new pages via 5 parallel subagents (one wave, non-overlapping file sets, matching
the established "content manager" pattern): critical-kv, cur, expected-attention,
duo-attention, chunkkv, qfilter, finch, lagkv, keydiff, kvzip, restorekv, cam, compactor
(covers `LeverageScorePress`+`NonCausalAttnPress` too), fast-kvzip (covers both `KVzapPress`
and `FastKVzipPress` as sibling learned approximations of KVzip), dms, simlayerkv,
kvcompose, merging, capkv, lukv. Each agent verified its assigned presses against the real
`.py` source (fetched locally first, not trusted from a summary) and against the cited
papers via WebFetch, introduced new family tags where the existing 7 didn't fit
(head-wise, partition-based, similarity-based, reconstruction-based, learned, threshold-based,
low-rank, merging, structured), and left CSS-merge requests instead of touching the shared
stylesheet directly. Two pages needed new scoped CSS (CaM's merge-vs-evict token states,
LagKV's partition-box layout) — merged into `css/style.css` by the orchestrating session
afterward, along with a small unrelated fix (`ul.rows a.row-link` → `ul.rows .row-link` so
the new non-clickable wrapper-note rows get the same padding as real linked rows).

The 10 pure wrappers (AdaKV, ComposedPress, ChunkPress, BlockPress, KeyRerotationPress,
PerLayerCompressionPress, DecodingPress/PrefillDecodingPress/CompressionRatioDecodingPress
as one family, RandomPress) got one grouped "Combinators & wrappers" section on
`compression/index.html` instead of individual pages — a scoping decision the site owner
made explicitly (asked via AskUserQuestion before generating 20+ pages) rather than one
inferred silently, given the size of the undertaking.

`about.html`'s "why these methods" section and the stale "eight methods... CaM/KVzip not
covered yet" framing were rewritten to match (CaM and KVzip are exactly the two examples
that section used to name as the known gap). `compression/index.html`'s "Inference-time"
list grew from 8 rows to 28, ordered by family adjacency (not alphabetical) so related
methods sit near each other, matching the existing single-flat-list pattern rather than
introducing new `<h3>` subheadings — a deliberate call to avoid a new structural pattern
the rest of the site doesn't use, even though 28 rows is a lot to scroll.

Follow-up: ran the same read-only-agents + single-fix-pass pattern as the 2026-08-14
stress test (5 parallel audits: humanization, demo correctness, cross-page consistency,
shared-file regression, citation/link accuracy) across the 20 new pages, then applied
every confirmed finding directly. Real fixes: a repeated budget-clamp bug
(`Math.max(nSink, Math.floor(...))` instead of `Math.max(1, Math.floor(...))`) in
cur.html, expected-attention.html, finch.html, and capkv.html silently over-guaranteed
sink-token survival past what the real ranking promises, plus matching prose overclaims
softened to match; an em-dash-as-clause-separator regression and a repeated
"worth ___ing:" caveat-opener (compactor.html, chunkkv.html); two `<h2>` that should've
been `<h3>` (critical-kv.html, expected-attention.html); a meta-description missing the
site's ", interactively explained." suffix (simlayerkv.html); `README.md`'s Structure
section, stale since before the catalogue/concepts merge, rewritten to list all 32 pages.

The 5th audit (citation/link accuracy) failed twice on background-agent session-limit
errors, so it was done directly instead: fetched every arXiv abstract page and every
`github.com/NVIDIA/kvpress/blob/main/kvpress/presses/*.py` link cited across the 20 new
pages and diffed the real title/authors against what each page claims. One real bug
found: `simlayerkv.html` cited `arxiv.org/abs/2410.13846` as SimLayerKV, which was true
at submission (verified via the `v1` snapshot) but the same arXiv ID now serves a
different, later paper by an overlapping author set (`LightTransfer`) — arXiv allows an
id to be replaced with a revised submission, so an unversioned link can silently start
pointing at the wrong paper months or years later. Fixed by pinning the link to
`.../abs/2410.13846v1` and adding a one-line note explaining why. Every other arXiv ID
and every GitHub source link across the 20 pages checked out exactly (title, authors,
and file existence all verified directly, not just spot-checked). Worth remembering for
future citation audits: don't trust an unversioned arXiv link to keep meaning the same
paper forever — pin to `vN` when citing a specific claim/result from that paper.

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

- As of the 2026-08-15 expansion, the catalogue covers all 28 of kvpress's genuinely
  distinct methods; the remaining ~10 exported classes are pure wrappers, covered as one
  grouped note rather than individual pages (see `about.html`'s "Why these methods" section
  and `compression/index.html#combinators`). If kvpress adds new presses later, re-run the
  same classification (`gh api repos/NVIDIA/kvpress/contents/kvpress/presses` + read each
  docstring to decide "wrapper" vs. "method") rather than assuming the count is still current.

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
