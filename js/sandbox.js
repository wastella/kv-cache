/*
 * Shared "compression sandbox" component used across the catalogue pages.
 *
 * Each method page supplies a `compute(tokens, params)` function that
 * returns which token indices survive compression. The scoring formulas
 * mirror the real logic in NVIDIA's kvpress presses (see the linked source
 * on each page), but the attention/importance numbers themselves are
 * simulated locally in the browser — there is no model running here. That's
 * called out on every page that uses it.
 */

// Shared sample sequence so the three catalogue pages are directly
// comparable against the same input.
const SAMPLE_TOKENS = (
  "the transformer stores a key and value vector for every token it has " +
  "already processed so it never has to recompute attention over the full " +
  "prompt again during generation"
).split(" ");

// deterministic pseudo-random in [0, 1) from a string seed, so demos don't
// jitter between recomputes unless tokens/params actually change.
function seeded(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 15;
  return ((h >>> 0) % 100000) / 100000;
}

function normalize(scores) {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min || 1e-9;
  return scores.map((s) => (s - min) / range);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(c);
  return node;
}

let demoInstanceCount = 0;

/**
 * @param {Object} opts
 * @param {HTMLElement} opts.root - container to render into
 * @param {string[]} opts.tokens - the token sequence to visualize
 * @param {Array} opts.controls - [{key, label, min, max, step, default, format}]
 * @param {Function} opts.compute - (tokens, params) => {kept: Set<number>, scores: number[0..1], tags?: Map<number,string[]>}
 * @param {string} [opts.label] - small caption above the grid
 */
function createCompressionDemo(opts) {
  const { root, tokens, controls, compute, label } = opts;
  const params = {};
  controls.forEach((c) => (params[c.key] = c.default));

  demoInstanceCount += 1;
  const controlIdBase = `demo${demoInstanceCount}-control`;

  root.innerHTML = "";
  root.className = "demo";

  if (label) root.appendChild(el("div", { class: "demo-label" }, [document.createTextNode(label)]));

  const legend = el("div", { class: "legend" });
  legend.appendChild(
    el("span", {}, [el("span", { class: "swatch" }), document.createTextNode("kept in cache")])
  );
  legend.appendChild(
    el("span", {}, [el("span", { class: "swatch evicted" }), document.createTextNode("evicted")])
  );
  root.appendChild(legend);

  const grid = el("div", { class: "token-grid" });
  root.appendChild(grid);

  const tokenEls = tokens.map((tok, i) => {
    const t = el("span", { class: "token" }, [document.createTextNode(tok)]);
    grid.appendChild(t);
    return t;
  });

  const controlsWrap = el("div", { class: "demo-controls" });
  controls.forEach((c, idx) => {
    const inputId = `${controlIdBase}-${idx}`;
    const valSpan = el("span", { class: "val" }, [document.createTextNode(c.format ? c.format(c.default) : String(c.default))]);
    const labelRow = el("label", { for: inputId }, [
      document.createTextNode(c.label + " "),
      valSpan,
    ]);
    const input = el("input", {
      type: "range",
      id: inputId,
      min: c.min,
      max: c.max,
      step: c.step,
      value: c.default,
    });
    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      params[c.key] = v;
      valSpan.textContent = c.format ? c.format(v) : String(v);
      render();
    });
    const control = el("div", { class: "control" }, [labelRow, input]);
    controlsWrap.appendChild(control);
  });
  root.appendChild(controlsWrap);

  const statsWrap = el("div", { class: "demo-stats" });
  const statKept = el("div", { class: "stat" }, [
    el("div", { class: "stat-num" }, [document.createTextNode("—")]),
    el("div", { class: "stat-label" }, [document.createTextNode("tokens kept")]),
  ]);
  const statRatio = el("div", { class: "stat" }, [
    el("div", { class: "stat-num" }, [document.createTextNode("—")]),
    el("div", { class: "stat-label" }, [document.createTextNode("cache size")]),
  ]);
  statsWrap.appendChild(statKept);
  statsWrap.appendChild(statRatio);
  root.appendChild(statsWrap);

  function render() {
    const result = compute(tokens, params);
    const kept = result.kept;
    const scores = result.scores || tokens.map(() => 1);
    const tags = result.tags || new Map();

    tokenEls.forEach((tEl, i) => {
      tEl.classList.toggle("evicted", !kept.has(i));
      tEl.classList.remove("sink", "window");
      (tags.get(i) || []).forEach((tag) => tEl.classList.add(tag));
      if (kept.has(i)) {
        tEl.style.background = `rgba(23, 22, 15, ${0.03 + scores[i] * 0.12})`;
      } else {
        tEl.style.background = "";
      }
    });

    statKept.firstChild.textContent = `${kept.size} / ${tokens.length}`;
    const pct = Math.round((kept.size / tokens.length) * 100);
    statRatio.firstChild.textContent = `${pct}%`;
  }

  render();
  return { render, params };
}

/**
 * Simple prefill/decode cache-growth demo used on basics.html.
 */
function createGrowthDemo({ root, tokenCount = 40 }) {
  root.innerHTML = "";
  root.className = "demo";

  root.appendChild(
    el("div", { class: "demo-label" }, [document.createTextNode("prefill → decode, one token at a time")])
  );

  const phaseLine = el("div", { class: "muted", style: "margin-bottom:10px;font-size:13px;" });
  root.appendChild(phaseLine);

  const track = el("div", { class: "membar-track" });
  const fill = el("div", { class: "membar-fill" });
  track.appendChild(fill);
  root.appendChild(track);

  const readout = el("div", { class: "muted", style: "font-size:12.5px;margin-bottom:16px;" });
  root.appendChild(readout);

  const buttons = el("div", { class: "demo-buttons" });
  const playBtn = el("button", { class: "btn" }, [document.createTextNode("play")]);
  const resetBtn = el("button", { class: "btn ghost" }, [document.createTextNode("reset")]);
  buttons.appendChild(playBtn);
  buttons.appendChild(resetBtn);
  root.appendChild(buttons);

  let i = 0;
  let timer = null;

  function draw() {
    const pct = Math.round((i / tokenCount) * 100);
    fill.style.width = pct + "%";
    const phase = i === 0 ? "idle" : i <= 8 ? "prefill (prompt)" : "decode (generating)";
    phaseLine.textContent = `step ${i} / ${tokenCount} — ${phase}`;
    readout.textContent = `cache holds key+value vectors for ${i} token${i === 1 ? "" : "s"} across every layer and head`;
  }

  function step() {
    if (i >= tokenCount) {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = "play";
      return;
    }
    i += 1;
    draw();
  }

  playBtn.addEventListener("click", () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = "play";
      return;
    }
    if (i >= tokenCount) i = 0;
    playBtn.textContent = "pause";
    timer = setInterval(step, 180);
  });

  resetBtn.addEventListener("click", () => {
    clearInterval(timer);
    timer = null;
    playBtn.textContent = "play";
    i = 0;
    draw();
  });

  draw();
}
