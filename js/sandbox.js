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
 * Prefill/decode cache-growth demo used on basics.html. Shows GPT-2's real
 * token sequence filling the cache: the whole prompt lands in one step
 * (prefill happens in parallel), then one token appends per step after
 * that (decode is sequential) — the two-groups layout and the "all at
 * once vs. one at a time" pacing are the point, not just a percentage.
 * Clicking any cached token reveals its real Q/K/V vectors at the chosen
 * layer/head (see js/gpt2-basics-data.js — an actual GPT-2 forward pass,
 * not simulated), and makes concrete which of the three actually stays
 * in the cache. Switching layer/head shows the real vectors are actually
 * different at each one, not just a label.
 */
function createGrowthDemo({
  root,
  tokens = GPT2_TOKENS,
  qkvByKey = GPT2_QKV,
  layers = GPT2_LAYERS,
  heads = GPT2_HEADS,
  prefillCount = 8,
}) {
  root.innerHTML = "";
  root.className = "demo";

  root.appendChild(
    el("div", { class: "demo-label" }, [document.createTextNode("prefill → decode, watch the cache fill")])
  );

  root.appendChild(
    el("div", { class: "token-group-label" }, [document.createTextNode(`prompt: prefill (${prefillCount} tokens, all at once)`)])
  );
  const prefillGrid = el("div", { class: "token-grid" });
  root.appendChild(prefillGrid);

  root.appendChild(
    el("div", { class: "token-group-label" }, [document.createTextNode(`generated: decode (${tokens.length - prefillCount} tokens, one per step)`)])
  );
  const decodeGrid = el("div", { class: "token-grid" });
  root.appendChild(decodeGrid);

  let selected = null; // index of the token currently shown in the QKV panel
  let userPicked = false; // true once the visitor clicks a token themselves
  let currentLayer = layers[0];
  let currentHead = heads[0];

  const tokenEls = tokens.map((tok, idx) => {
    const t = el("span", { class: "token pending", tabindex: "0", role: "button" }, [document.createTextNode(tok.trim())]);
    const pick = () => {
      if (t.classList.contains("pending")) return;
      selected = idx;
      userPicked = true;
      drawPanel();
      tokenEls.forEach((e, i) => e.classList.toggle("selected", i === selected));
    };
    t.addEventListener("click", pick);
    t.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pick();
      }
    });
    (idx < prefillCount ? prefillGrid : decodeGrid).appendChild(t);
    return t;
  });

  function pillRow(caption, values, formatLabel, isActive, onPick) {
    const wrap = el("div", { class: "qkv-selector-row" });
    wrap.appendChild(el("span", { class: "qkv-selector-caption" }, [document.createTextNode(caption)]));
    const pills = el("span", { class: "qkv-selector-pills" });
    values.forEach((v) => {
      const btn = el("button", { class: isActive(v) ? "btn" : "btn ghost" }, [document.createTextNode(formatLabel(v))]);
      btn.addEventListener("click", () => onPick(v));
      pills.appendChild(btn);
    });
    wrap.appendChild(pills);
    return wrap;
  }

  const selectorWrap = el("div", { class: "qkv-selector" });
  root.appendChild(selectorWrap);

  function drawSelector() {
    selectorWrap.innerHTML = "";
    selectorWrap.appendChild(
      pillRow("layer", layers, (l) => `layer ${l}`, (l) => l === currentLayer, (l) => {
        currentLayer = l;
        drawSelector();
        drawPanel();
      })
    );
    selectorWrap.appendChild(
      pillRow("head", heads, (h) => `head ${h}`, (h) => h === currentHead, (h) => {
        currentHead = h;
        drawSelector();
        drawPanel();
      })
    );
  }

  const panel = el("div", { class: "qkv-panel" });
  root.appendChild(panel);

  function numsRow(label, nums, discarded, badgeText, explain) {
    const row = el("div", { class: discarded ? "qkv-row discarded" : "qkv-row" });
    const main = el("div", { class: "qkv-row-main" });
    main.appendChild(el("span", { class: "qkv-label" }, [document.createTextNode(label)]));
    const numsWrap = el("span", { class: "qkv-nums" });
    nums.forEach((n) => numsWrap.appendChild(el("span", { class: "qkv-num" }, [document.createTextNode(n.toFixed(2))])));
    numsWrap.appendChild(el("span", { class: "qkv-ellipsis" }, [document.createTextNode("…")]));
    main.appendChild(numsWrap);
    main.appendChild(el("span", { class: discarded ? "qkv-badge discarded" : "qkv-badge" }, [document.createTextNode(badgeText)]));
    row.appendChild(main);
    row.appendChild(el("div", { class: "qkv-explain" }, [document.createTextNode(explain)]));
    return row;
  }

  function drawPanel() {
    panel.innerHTML = "";
    if (selected === null) {
      panel.appendChild(
        el("div", { class: "qkv-title" }, [document.createTextNode("click any cached token above to inspect its Q, K, V vectors")])
      );
      return;
    }
    const vec = qkvByKey[`${currentLayer}:${currentHead}`][selected];
    panel.appendChild(
      el("div", { class: "qkv-title" }, [
        document.createTextNode(
          `"${tokens[selected].trim()}": real GPT-2 vectors (layer ${currentLayer}, head ${currentHead}; 4 of 64 head_dim components shown)`
        ),
      ])
    );
    panel.appendChild(
      el("div", { class: "qkv-lead" }, [
        document.createTextNode(
          "All three are computed for every token, but only K and V get written to the cache. That's why it's called a KV cache, not a QKV cache."
        ),
      ])
    );
    panel.appendChild(
      numsRow(
        "Q",
        vec.q,
        true,
        "used once",
        "Compared against every key right now to produce this token's own attention output at this step, then never touched again."
      )
    );
    panel.appendChild(
      numsRow(
        "K",
        vec.k,
        false,
        "cached",
        "Every future token's query gets compared against this to score how much that future token should attend to this one."
      )
    );
    panel.appendChild(
      numsRow(
        "V",
        vec.v,
        false,
        "cached",
        "Blended into the output of every future token whose query scores highly against this token's key."
      )
    );
  }

  drawSelector();

  const phaseLine = el("div", { class: "muted", style: "margin:14px 0 10px;font-size:13px;" });
  root.appendChild(phaseLine);

  const buttons = el("div", { class: "demo-buttons" });
  const stepBtn = el("button", { class: "btn ghost" }, [document.createTextNode("step")]);
  const playBtn = el("button", { class: "btn" }, [document.createTextNode("play")]);
  const resetBtn = el("button", { class: "btn ghost" }, [document.createTextNode("reset")]);
  buttons.appendChild(stepBtn);
  buttons.appendChild(playBtn);
  buttons.appendChild(resetBtn);
  root.appendChild(buttons);

  const statsWrap = el("div", { class: "demo-stats" });
  const statCached = el("div", { class: "stat" }, [
    el("div", { class: "stat-num" }, [document.createTextNode("0 / " + tokens.length)]),
    el("div", { class: "stat-label" }, [document.createTextNode("tokens cached")]),
  ]);
  const statPhase = el("div", { class: "stat" }, [
    el("div", { class: "stat-num" }, [document.createTextNode("idle")]),
    el("div", { class: "stat-label" }, [document.createTextNode("phase")]),
  ]);
  statsWrap.appendChild(statCached);
  statsWrap.appendChild(statPhase);
  root.appendChild(statsWrap);

  let i = 0;
  let timer = null;

  function stopTimer() {
    clearTimeout(timer);
    timer = null;
    playBtn.textContent = "play";
  }

  function draw() {
    tokenEls.forEach((tEl, idx) => tEl.classList.toggle("pending", idx >= i));
    const phase = i === 0 ? "idle" : i < prefillCount ? "prefill (prompt)" : i < tokens.length ? "decode (generating)" : "done";
    statCached.firstChild.textContent = `${i} / ${tokens.length}`;
    statPhase.firstChild.textContent = phase;
    phaseLine.textContent = `cache holds key+value vectors for ${i} token${i === 1 ? "" : "s"} across every layer and head`;

    // follow the newest cached token unless the visitor picked one manually
    if (!userPicked) selected = i > 0 ? i - 1 : null;
    if (selected !== null && selected >= i) selected = i > 0 ? i - 1 : null;
    tokenEls.forEach((tEl, idx) => tEl.classList.toggle("selected", idx === selected));
    drawPanel();
  }

  // Prefill happens "all at once, in parallel" — jump straight to the end
  // of the prompt on the first step instead of counting up through it.
  function step() {
    if (i >= tokens.length) {
      stopTimer();
      return;
    }
    i = i === 0 ? prefillCount : i + 1;
    draw();
    if (i >= tokens.length) stopTimer();
  }

  stepBtn.addEventListener("click", () => {
    stopTimer();
    step();
  });

  // Re-picks the delay before every tick so autoplay actually speeds up
  // from prefill's 700ms to decode's 550ms once i crosses prefillCount,
  // instead of locking in whichever delay applied when play was clicked.
  function scheduleTick() {
    timer = setTimeout(() => {
      step();
      if (timer) scheduleTick();
    }, i < prefillCount ? 700 : 550);
  }

  playBtn.addEventListener("click", () => {
    if (timer) {
      stopTimer();
      return;
    }
    if (i >= tokens.length) i = 0;
    playBtn.textContent = "pause";
    scheduleTick();
  });

  resetBtn.addEventListener("click", () => {
    stopTimer();
    i = 0;
    selected = null;
    userPicked = false;
    draw();
  });

  draw();
}
