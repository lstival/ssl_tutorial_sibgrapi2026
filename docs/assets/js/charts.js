/* ------------------------------------------------------------
   Hand-rolled SVG charts driven by the tutorial's real result
   JSON (assets/data/*.json). No chart library.
   All text uses currentColor-derived theme tokens.
   ------------------------------------------------------------ */

const NS = "http://www.w3.org/2000/svg";
const K = { con:"var(--c-con)", mask:"var(--c-mask)", dist:"var(--c-dist)", base:"var(--ink-3)",
            scratch:"var(--ink-2)" };
/* Baseline curves are drawn thin, dashed and without the draw-in animation; "scratch" gets a
   dotted pattern so it stays distinguishable from "base" when both are on one chart. */
const DASH = { base: "5 4", scratch: "2 4" };

const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  return e;
};
const pct = (v) => `${(v * 100).toFixed(1)}%`;

/* ---------- 1. grouped bars: paradigm accuracy per modality ---------- */
function barChart(host, series, opts = {}) {
  const W = 680, H = 300;
  const m = { t: 18, r: 16, b: 54, l: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lo = opts.min ?? 0.7, hi = opts.max ?? 1.0;
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || "Bar chart" });

  /* y grid + ticks */
  for (let v = lo; v <= hi + 1e-9; v += 0.05) {
    const yy = y(v);
    svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy,
      stroke: "var(--rule)", "stroke-width": 1 }));
    const t = el("text", { x: m.l - 9, y: yy + 3.5, "text-anchor": "end",
      fill: "var(--ink-3)", "font-size": 10, "font-family": "var(--ff-mono)" });
    t.textContent = `${Math.round(v * 100)}%`;
    svg.appendChild(t);
  }

  const bw = iw / series.length;
  const barW = Math.min(64, bw * 0.54);

  series.forEach((s, i) => {
    const cx = m.l + bw * i + bw / 2;
    const yy = y(s.v);
    const h = Math.max(1, m.t + ih - yy);

    const r = el("rect", { x: cx - barW / 2, y: yy, width: barW, height: h,
      rx: 3, fill: K[s.k] || K.base });
    if (s.k === "base") { r.setAttribute("fill", "var(--ink-3)"); r.setAttribute("opacity", ".45"); }
    /* grow-in */
    const anim = el("animate", { attributeName: "height", from: 0, to: h,
      dur: "0.75s", begin: `${i * 0.08}s`, fill: "freeze",
      calcMode: "spline", keySplines: ".2 .7 .2 1", keyTimes: "0;1" });
    const anim2 = el("animate", { attributeName: "y", from: m.t + ih, to: yy,
      dur: "0.75s", begin: `${i * 0.08}s`, fill: "freeze",
      calcMode: "spline", keySplines: ".2 .7 .2 1", keyTimes: "0;1" });
    if (!window.matchMedia("(prefers-reduced-motion:reduce)").matches) {
      r.appendChild(anim); r.appendChild(anim2);
    }
    svg.appendChild(r);

    /* value above bar */
    const vt = el("text", { x: cx, y: yy - 7, "text-anchor": "middle",
      fill: s.best ? (K[s.k] || "var(--ink)") : "var(--ink-2)",
      "font-size": 11.5, "font-weight": s.best ? 700 : 400,
      "font-family": "var(--ff-mono)" });
    vt.textContent = pct(s.v);
    svg.appendChild(vt);

    /* category label (wraps to 2 lines) */
    const words = s.label.split(" ");
    words.forEach((w, j) => {
      const lt = el("text", { x: cx, y: m.t + ih + 19 + j * 12, "text-anchor": "middle",
        fill: "var(--ink-2)", "font-size": 11, "font-family": "var(--ff-mono)" });
      lt.textContent = w;
      svg.appendChild(lt);
    });
    if (s.best) {
      const bt = el("text", { x: cx, y: m.t + ih + 19 + words.length * 12 + 1,
        "text-anchor": "middle", fill: K[s.k], "font-size": 9.5,
        "font-family": "var(--ff-mono)", "letter-spacing": ".08em" });
      bt.textContent = "BEST";
      svg.appendChild(bt);
    }
  });

  /* axis */
  svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih,
    stroke: "var(--rule-2)", "stroke-width": 1 }));

  host.innerHTML = "";
  host.appendChild(svg);
}

/* ---------- 2. line chart: few-label curves / scaling ---------- */
function lineChart(host, lines, xs, opts = {}) {
  const W = 680, H = 300;
  const m = { t: 18, r: 16, b: 48, l: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lo = opts.min ?? 0.3, hi = opts.max ?? 1.0;
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const x = (i) => m.l + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || "Line chart" });

  const step = opts.step ?? 0.1;
  for (let v = lo; v <= hi + 1e-9; v += step) {
    const yy = y(v);
    svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy,
      stroke: "var(--rule)", "stroke-width": 1 }));
    const t = el("text", { x: m.l - 9, y: yy + 3.5, "text-anchor": "end",
      fill: "var(--ink-3)", "font-size": 10, "font-family": "var(--ff-mono)" });
    t.textContent = `${Math.round(v * 100)}%`;
    svg.appendChild(t);
  }

  xs.forEach((lbl, i) => {
    const t = el("text", { x: x(i), y: m.t + ih + 20, "text-anchor": "middle",
      fill: "var(--ink-2)", "font-size": 11, "font-family": "var(--ff-mono)" });
    t.textContent = lbl;
    svg.appendChild(t);
  });
  if (opts.xlabel) {
    const t = el("text", { x: m.l + iw / 2, y: H - 8, "text-anchor": "middle",
      fill: "var(--ink-3)", "font-size": 10.5, "font-family": "var(--ff-mono)",
      "letter-spacing": ".06em" });
    t.textContent = opts.xlabel;
    svg.appendChild(t);
  }

  const ends = [];
  lines.forEach((ln, li) => {
    const color = K[ln.k] || K.base;
    /* a reference line: a single value drawn flat across the axis, labelled on the left */
    if (ln.ref !== undefined) {
      const yy = y(ln.ref);
      svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: color,
        "stroke-width": 1.6, "stroke-dasharray": "5 4", opacity: ln.k === "base" ? .6 : .85 }));
      const t = el("text", { x: m.l + 6, y: yy - 6, fill: color, "font-size": 10.5,
        "font-weight": 600, "font-family": "var(--ff-mono)",
        stroke: "var(--surface)", "stroke-width": 3, "paint-order": "stroke" });
      t.textContent = `${ln.tag} · ${pct(ln.ref)}`;
      svg.appendChild(t);
      return;
    }
    const pts = ln.v.map((v, i) => [x(i), y(v)]);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    const dash = DASH[ln.k];
    const path = el("path", { d, fill: "none", stroke: color,
      "stroke-width": dash ? 1.6 : 2.2,
      "stroke-dasharray": dash || "none",
      "stroke-linecap": "round", "stroke-linejoin": "round",
      opacity: ln.k === "base" ? .6 : 1 });
    svg.appendChild(path);

    if (!window.matchMedia("(prefers-reduced-motion:reduce)").matches) {
      const L = 900;
      path.setAttribute("stroke-dasharray", dash || L);
      if (!dash) {
        path.setAttribute("stroke-dashoffset", L);
        path.appendChild(el("animate", { attributeName: "stroke-dashoffset",
          from: L, to: 0, dur: "1s", begin: `${li * 0.12}s`, fill: "freeze" }));
      }
    }

    pts.forEach((p, i) => {
      svg.appendChild(el("circle", { cx: p[0], cy: p[1], r: DASH[ln.k] ? 2.5 : 3.4,
        fill: color, opacity: ln.k === "base" ? .6 : 1 }));
    });

    /* endpoint value, placed below once every line is known */
    const last = pts[pts.length - 1];
    ends.push({ x: last[0] - 4, y: last[1] - 10, color, text: pct(ln.v[ln.v.length - 1]) });
  });

  /* Stack the endpoint values top to bottom, pushing each one down just enough that it does
     not overlap the one above: curves that finish close together stay readable. */
  ends.sort((a, b) => a.y - b.y);
  ends.forEach((e, i) => {
    if (i > 0) e.y = Math.max(e.y, ends[i - 1].y + 12);
    const vt = el("text", { x: e.x, y: e.y, "text-anchor": "end",
      fill: e.color, "font-size": 11, "font-weight": 600, "font-family": "var(--ff-mono)",
      stroke: "var(--surface)", "stroke-width": 3, "paint-order": "stroke" });
    vt.textContent = e.text;
    svg.appendChild(vt);
  });

  svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih,
    stroke: "var(--rule-2)", "stroke-width": 1 }));

  host.innerHTML = "";
  host.appendChild(svg);
}

/* ---------- legend ---------- */
function legend(host, items) {
  host.innerHTML = items.map(i =>
    `<span class="lg"><i style="background:${K[i.k] || "var(--ink-3)"};${i.k === "base" ? "opacity:.5" : ""}"></i>${i.label}</span>`
  ).join("");
}

/* ---------- data wiring ---------- */
const DATA = {};

async function loadData() {
  const files = {
    rs: "assets/data/rs_eval_results.json",
    sup: "assets/data/supervised_results.json",
    seco: "assets/data/seco_scaling_results.json",
    /* optional until the MAE / DINO scaling runs land (run_seco_scaling.py --mechanism ...) */
    seco_mae: "assets/data/seco_scaling_mae_results.json",
    seco_dino: "assets/data/seco_scaling_dino_results.json",
    ts: "assets/data/ts_eval_results.json"
  };
  await Promise.all(Object.entries(files).map(async ([k, p]) => {
    try {
      const r = await fetch(p);
      DATA[k] = r.ok ? await r.json() : null;
    } catch (e) { DATA[k] = null; }
  }));
}

function renderModality(mode) {
  const barHost = document.getElementById("chart-bars");
  const barLg = document.getElementById("lg-bars");
  const fewHost = document.getElementById("chart-few");
  const fewLg = document.getElementById("lg-few");
  const barSub = document.getElementById("sub-bars");
  const fewSub = document.getElementById("sub-few");
  if (!barHost) return;

  if (mode === "rs" && DATA.rs) {
    const d = DATA.rs;
    barChart(barHost, [
      { label: "Random init", v: d.random_init.full_label, k: "base" },
      { label: "Contrastive", v: d.contrastive.full_label, k: "con", best: true },
      { label: "Masking MAE", v: d.mae.full_label, k: "mask" },
      { label: "DINO", v: d.dino.full_label, k: "dist" },
      { label: "Supervised", v: DATA.sup ? DATA.sup.supervised_fromscratch_full : 0.9307, k: "base" }
    ], { min: 0.75, max: 0.95, aria: "Linear-probe accuracy on EuroSAT by pretraining paradigm" });
    barSub.textContent = "EuroSAT · frozen encoder + linear probe · full label budget";
    legend(barLg, [
      { k: "con", label: "Contrastive" }, { k: "mask", label: "Masking (MAE)" },
      { k: "dist", label: "Distillation (DINO)" }, { k: "base", label: "Baselines" }
    ]);

    const budgets = ["5", "10", "25", "50", "100"];
    if (DATA.sup) {
      lineChart(fewHost, [
        { k: "con", v: budgets.map(b => DATA.sup.low_label.ssl_ft[b]) },
        { k: "mask", v: budgets.map(b => DATA.sup.low_label.ssl_probe[b]) },
        { k: "base", v: budgets.map(b => DATA.sup.low_label.scratch_ft[b]) }
      ], budgets, { min: 0.45, max: 0.9, step: 0.1,
        xlabel: "LABELS PER CLASS",
        aria: "Accuracy versus labels per class, SSL versus training from scratch" });
      fewSub.textContent = "EuroSAT · accuracy as the label budget shrinks";
      legend(fewLg, [
        { k: "con", label: "SSL + fine-tune" }, { k: "mask", label: "SSL + linear probe" },
        { k: "base", label: "From scratch" }
      ]);
    }
  }

  if (mode === "ts" && DATA.ts) {
    const d = DATA.ts;
    /* eval_ts_encoders.py writes each mechanism as an object ({accuracy, pool, by_readout,
       checkpoint_meta}) and the random-init baseline at the top level. Accept either that or
       a bare number, so an older results file still renders. */
    const acc = (x) => (x && typeof x === "object" ? x.accuracy : x);
    barChart(barHost, [
      { label: "Random init", v: acc(d.full_label.random_init ?? d.random_init), k: "base" },
      { label: "Contrastive", v: acc(d.full_label.contrastive), k: "con" },
      { label: "Masking MAE", v: acc(d.full_label.mae), k: "mask" },
      { label: "DINO", v: acc(d.full_label.dino), k: "dist", best: true },
      { label: "Supervised", v: d.supervised, k: "base" }
    ], { min: 0.75, max: 0.96, aria: "Linear-probe accuracy on UCR SwedishLeaf by pretraining paradigm" });
    barSub.textContent = `UCR · ${d.target} · ${d.n_classes} classes · frozen encoder + linear probe`;
    legend(barLg, [
      { k: "con", label: "Contrastive" }, { k: "mask", label: "Masking (MAE)" },
      { k: "dist", label: "Distillation (DINO)" }, { k: "base", label: "Baselines" }
    ]);

    const budgets = d.label_budgets.map(String);
    const fewLines = [
      { k: "dist", v: budgets.map(b => d.few_label.dino[b]) },
      { k: "con", v: budgets.map(b => d.few_label.contrastive[b]) },
      { k: "mask", v: budgets.map(b => d.few_label.mae[b]) },
      { k: "base", v: budgets.map(b => (d.few_label.random_init || d.random_init.few_label)[b]) }
    ];
    const fewItems = [
      { k: "dist", label: "DINO" }, { k: "con", label: "Contrastive" },
      { k: "mask", label: "MAE" }, { k: "base", label: "Random init" }
    ];
    /* trained from scratch on the same k labels per class; older results files lack it */
    if (d.few_label.supervised) {
      fewLines.push({ k: "scratch", v: budgets.map(b => d.few_label.supervised[b]) });
      fewItems.push({ k: "scratch", label: "From scratch" });
    }
    lineChart(fewHost, fewLines, budgets, { min: 0.3, max: 0.95, step: 0.1,
      xlabel: "LABELS PER CLASS",
      aria: "Accuracy versus labels per class for each paradigm on time series" });
    fewSub.textContent = `UCR · ${d.target} · accuracy as the label budget shrinks`;
    legend(fewLg, fewItems);
  }
}

function renderScaling() {
  const host = document.getElementById("chart-scaling");
  const lg = document.getElementById("lg-scaling");
  if (!host || !DATA.seco) return;
  const keys = ["f002", "f005", "f010", "f030", "f100"];
  const labels = keys.map(k => {
    const f = DATA.seco[k].meta.fraction;
    return f >= 1 ? "100%" : `${(f * 100).toFixed(f < 0.1 ? 0 : 0)}%`;
  });
  const lines = [{ k: "con", v: keys.map(k => DATA.seco[k].full_label) }];
  const items = [{ k: "con", label: "Contrastive" }];
  const pending = [];

  /* MAE / DINO: a full curve once their scaling runs exist, otherwise the single measured
     full-corpus value as a flat reference -- never interpolated points. */
  [["mae", "mask", "MAE (norm-pix)", "mae"], ["dino", "dist", "DINO", "dino"]]
    .forEach(([name, k, label, fullKey]) => {
      const curve = DATA[`seco_${name}`];
      if (curve && keys.every(f => curve[f])) {
        lines.push({ k, v: keys.map(f => curve[f].full_label) });
        items.push({ k, label });
      } else if (DATA.rs && DATA.rs[fullKey]) {
        lines.push({ k, ref: DATA.rs[fullKey].full_label, tag: `${label}, 100% only` });
        items.push({ k, label: `${label} (100% SeCo only)` });
        pending.push(label);
      }
    });

  if (DATA.sup) {
    lines.push({ k: "base", ref: DATA.sup.supervised_fromscratch_full, tag: "Supervised, no pretraining" });
    items.push({ k: "base", label: "Supervised from scratch (all labels)" });
  }

  const vals = lines.flatMap(l => (l.ref !== undefined ? [l.ref] : l.v));
  const lo = Math.floor((Math.min(...vals) - 0.005) * 50) / 50;
  const hi = Math.ceil((Math.max(...vals) + 0.005) * 50) / 50;
  lineChart(host, lines, labels, { min: lo, max: hi, step: 0.02,
    xlabel: "SHARE OF THE SeCo PRETRAINING CORPUS",
    aria: "EuroSAT linear-probe accuracy as the unlabeled pretraining corpus grows, per SSL method, with a supervised reference" });
  legend(lg, items);

  const sub = document.getElementById("sub-scaling");
  if (sub) {
    sub.textContent = "Encoders pretrained on growing fractions of the SeCo corpus, each for a fixed 20 000 steps, " +
      "then linear-probed on EuroSAT. Dashed: supervised training from scratch, which uses no unlabeled data" +
      (pending.length ? `, and ${pending.join(" / ")} measured on the full corpus only (smaller fractions pending).` : ".");
  }
}
