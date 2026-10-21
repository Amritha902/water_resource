/* Minimal, dependency-free SVG charts for the project site.
 *
 *   Charts.line(el, spec)   time series with crosshair tooltip
 *   Charts.hbar(el, spec)   horizontal bars with per-bar tooltip
 *
 * Colours are CSS variable names ("--series-1"), so light/dark mode swaps
 * them automatically.  Labels are inserted with textContent, never innerHTML.
 */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const cssVar = (name) => (name && name.startsWith("--") ? `var(${name})` : name);

  function niceTicks(lo, hi, n) {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (hi === lo) { hi = lo + 1; lo = lo - 1; }
    const span = hi - lo;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = (span / n) / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmtCompact(v) {
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/, "") + "K";
    if (a === 0) return "0";
    if (a < 0.01) return v.toExponential(0);
    return (+v.toPrecision(3)).toString();
  }
  function bisect(xs, x) {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < x) lo = m; else hi = m; }
    return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
  }

  /* ------------------------------------------------------------------ */
  function line(container, spec) {
    container.textContent = "";
    container.classList.add("chart");
    if (spec.title) h("div", "chart-title", container, spec.title);
    if (spec.subtitle) h("div", "chart-sub", container, spec.subtitle);
    const series = spec.series.filter((s) => s && s.x && s.x.length);
    const bands = spec.bands || [];

    if (series.length + bands.length >= 2 || spec.legend) {
      const lg = h("div", "legend", container);
      for (const s of series) {
        const k = h("span", "key", lg);
        const sw = h("span", "swatch-line" + (s.dash ? " dash" : ""), k);
        sw.style.borderTopColor = cssVar(s.color);
        h("span", "", k, s.name);
      }
      for (const b of bands) {
        const k = h("span", "key", lg);
        const sw = h("span", "swatch-band", k);
        sw.style.background = cssVar(b.color);
        h("span", "", k, b.name);
      }
    }

    const wrap = h("div", "", container);
    wrap.style.position = "relative";
    const height = spec.height || 240;
    const m = { t: 10, r: spec.directLabels ? 92 : 16, b: spec.xLabel ? 38 : 26, l: 54 };
    const svg = el("svg", { height, role: "img", "aria-label": spec.ariaLabel || spec.title || "chart" }, wrap);
    const tip = h("div", "tooltip", wrap);
    wrap.tabIndex = 0;
    wrap.classList.add("chart");

    // domains
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    const scan = (xs, ys) => {
      for (let i = 0; i < xs.length; i++) {
        const x = xs[i], y = ys[i];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y == null || !isFinite(y)) continue;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      }
    };
    series.forEach((s) => scan(s.x, s.y));
    bands.forEach((b) => { scan(b.x, b.lo); scan(b.x, b.hi); });
    (spec.refLines || []).forEach((r) => { if (r.y < ymin) ymin = r.y; if (r.y > ymax) ymax = r.y; });
    if (spec.xMin !== undefined) xmin = spec.xMin;
    if (spec.xMax !== undefined) xmax = spec.xMax;
    if (spec.yMin !== undefined) ymin = spec.yMin;
    if (spec.yMax !== undefined) ymax = spec.yMax;
    if (spec.yMin === undefined && spec.zeroBased) ymin = Math.min(0, ymin);
    const pad = (ymax - ymin) * 0.06 || 1;
    if (spec.yMin === undefined && !spec.zeroBased) ymin -= pad;
    if (spec.yMax === undefined) ymax += pad;
    const yt = niceTicks(ymin, ymax, spec.yTicks || 4);
    if (spec.yMin === undefined) ymin = Math.min(ymin, yt[0]);
    if (spec.yMax === undefined) ymax = Math.max(ymax, yt[yt.length - 1]);

    let markerEl = null, Xcur = null, markerX = spec.marker;
    const xFmt = spec.xFmt || fmtCompact;
    const yFmt = spec.yFmt || fmtCompact;
    const vFmt = spec.valueFmt || ((v) => (v == null ? "–" : (+v).toPrecision(4)));

    function draw() {
      const W = Math.max(280, wrap.clientWidth);
      svg.setAttribute("width", W);
      svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const iw = W - m.l - m.r, ih = height - m.t - m.b;
      const X = (x) => m.l + ((x - xmin) / (xmax - xmin || 1)) * iw;
      const Y = (y) => m.t + ih - ((y - ymin) / (ymax - ymin || 1)) * ih;
      Xcur = X;

      const ax = el("g", { class: "axis" }, svg);
      for (const v of yt) {
        if (v < ymin - 1e-9 || v > ymax + 1e-9) continue;
        el("line", { class: "gridline", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }, ax);
        const t = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, ax);
        t.textContent = yFmt(v);
      }
      const xt = spec.xTicksAt || niceTicks(xmin, xmax, Math.max(3, Math.floor(iw / 90)));
      for (const v of xt) {
        if (v < xmin - 1e-9 || v > xmax + 1e-9) continue;
        const t = el("text", { x: X(v), y: m.t + ih + 16, "text-anchor": "middle" }, ax);
        t.textContent = xFmt(v);
      }
      el("line", { class: "gridline", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, ax);
      if (spec.xLabel) {
        const t = el("text", { class: "axis-label", x: m.l + iw / 2, y: height - 4, "text-anchor": "middle" }, svg);
        t.textContent = spec.xLabel;
      }
      if (spec.yLabel) {
        const t = el("text", { class: "axis-label", x: 12, y: m.t + ih / 2, "text-anchor": "middle",
          transform: `rotate(-90 12 ${m.t + ih / 2})` }, svg);
        t.textContent = spec.yLabel;
      }

      // shaded spans (e.g. storm events, violations)
      for (const sp of spec.spans || []) {
        el("rect", { x: X(sp.x0), y: m.t, width: Math.max(1, X(sp.x1) - X(sp.x0)), height: ih,
          fill: cssVar(sp.color || "--surface-2"), opacity: sp.opacity || 0.6 }, svg);
      }
      // bands
      for (const b of bands) {
        let d = "";
        for (let i = 0; i < b.x.length; i++) d += (i ? "L" : "M") + X(b.x[i]).toFixed(1) + "," + Y(b.hi[i]).toFixed(1);
        for (let i = b.x.length - 1; i >= 0; i--) d += "L" + X(b.x[i]).toFixed(1) + "," + Y(b.lo[i]).toFixed(1);
        el("path", { d: d + "Z", fill: cssVar(b.color), opacity: 0.16, stroke: "none" }, svg);
      }
      // reference lines
      for (const r of spec.refLines || []) {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(r.y), y2: Y(r.y), stroke: cssVar(r.color || "--reference"),
          "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
        if (r.label) {
          const t = el("text", { class: "ref-label", x: m.l + 4, y: Y(r.y) - 5 }, svg);
          t.textContent = r.label;
        }
      }
      // series
      const ends = [];
      for (const s of series) {
        let d = "", pen = false;
        for (let i = 0; i < s.x.length; i++) {
          const y = s.y[i];
          if (y == null || !isFinite(y) || s.x[i] < xmin || s.x[i] > xmax) { pen = false; continue; }
          d += (pen ? "L" : "M") + X(s.x[i]).toFixed(1) + "," + Y(Math.max(ymin, Math.min(ymax, y))).toFixed(1);
          pen = true;
        }
        el("path", { d, fill: "none", stroke: cssVar(s.color), "stroke-width": s.width || 2,
          "stroke-linejoin": "round", "stroke-linecap": "round",
          "stroke-dasharray": s.dash ? "6 4" : "none", opacity: s.opacity || 1 }, svg);
        if (spec.directLabels) {
          let i = s.x.length - 1;
          while (i > 0 && (s.y[i] == null || s.x[i] > xmax)) i--;
          ends.push({ s, y: Y(s.y[i]) });
        }
      }
      // direct labels with leader lines when they would collide
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) { const prev = ends[i - 1].ly ?? ends[i - 1].y; if (ends[i].y - prev < 16) ends[i].ly = prev + 16; }
      for (const e of ends) {
        const ly = e.ly || e.y;
        e.ly = ly;
        if (Math.abs(ly - e.y) > 2) el("line", { x1: m.l + iw + 2, x2: m.l + iw + 8, y1: e.y, y2: ly, stroke: cssVar("--text-muted"), "stroke-width": 1 }, svg);
        const t = el("text", { class: "direct-label", x: m.l + iw + 10, y: ly + 4 }, svg);
        t.textContent = e.s.short || e.s.name;
      }

      // crosshair
      const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: cssVar("--text-muted"), "stroke-width": 1, opacity: 0 }, svg);
      const dots = series.map((s) => el("circle", { r: 4, fill: cssVar(s.color), stroke: cssVar("--surface-1"), "stroke-width": 2, opacity: 0 }, svg));
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      const ref = series[0] || (bands[0] && { x: bands[0].x });
      let cur = -1;

      function show(i) {
        if (!ref) return;
        cur = Math.max(0, Math.min(ref.x.length - 1, i));
        const x = ref.x[cur];
        cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("opacity", 1);
        tip.textContent = "";
        h("div", "tt-head", tip, (spec.xTip || xFmt)(x));
        series.forEach((s, k) => {
          const j = s.x === ref.x ? cur : bisect(s.x, x);
          const y = s.y[j];
          if (y == null) { dots[k].setAttribute("opacity", 0); return; }
          dots[k].setAttribute("cx", X(s.x[j])); dots[k].setAttribute("cy", Y(Math.max(ymin, Math.min(ymax, y))));
          dots[k].setAttribute("opacity", 1);
          const row = h("div", "tt-row", tip);
          const key = h("span", "tt-key", row); key.style.borderTopColor = cssVar(s.color);
          h("b", "", row, vFmt(y, s));
          h("span", "", row, s.name);
        });
        for (const b of bands) {
          const j = bisect(b.x, x);
          const row = h("div", "tt-row", tip);
          h("span", "tt-key", row).style.borderTopColor = cssVar(b.color);
          h("b", "", row, `${vFmt(b.lo[j])} – ${vFmt(b.hi[j])}`);
          h("span", "", row, b.name);
        }
        tip.style.opacity = 1;
        const tx = X(x), tw = tip.offsetWidth;
        tip.style.left = (tx + 14 + tw > W ? tx - tw - 14 : tx + 14) + "px";
        tip.style.top = m.t + 4 + "px";
        if (spec.onHover) spec.onHover(x);
      }
      function hide() {
        cross.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0));
        tip.style.opacity = 0;
      }
      hit.addEventListener("pointermove", (ev) => {
        const r = svg.getBoundingClientRect();
        const x = xmin + ((ev.clientX - r.left - m.l) / iw) * (xmax - xmin);
        show(bisect(ref.x, x));
      });
      hit.addEventListener("pointerleave", hide);
      wrap.onkeydown = (ev) => {
        if (ev.key === "ArrowRight") { show(cur < 0 ? 0 : cur + Math.max(1, ref.x.length >> 7)); ev.preventDefault(); }
        if (ev.key === "ArrowLeft") { show(cur < 0 ? 0 : cur - Math.max(1, ref.x.length >> 7)); ev.preventDefault(); }
        if (ev.key === "Escape") hide();
      };
      wrap.onblur = hide;

      // playback cursor
      markerEl = el("line", { y1: m.t, y2: m.t + ih, stroke: cssVar("--text-primary"),
        "stroke-width": 1.2, opacity: 0 }, svg);
      setMarker(markerX);
    }
    function setMarker(x) {
      markerX = x;
      if (!markerEl) return;
      if (x === undefined || x === null) { markerEl.setAttribute("opacity", 0); return; }
      markerEl.setAttribute("x1", Xcur(x)); markerEl.setAttribute("x2", Xcur(x));
      markerEl.setAttribute("opacity", 0.55);
    }
    draw();
    if (window.ResizeObserver) {
      let last = wrap.clientWidth;
      new ResizeObserver(() => { if (wrap.clientWidth !== last) { last = wrap.clientWidth; draw(); } }).observe(wrap);
    }
    return { redraw: draw, setMarker };
  }

  /* ------------------------------------------------------------------ */
  function hbar(container, spec) {
    container.textContent = "";
    container.classList.add("chart");
    if (spec.title) h("div", "chart-title", container, spec.title);
    if (spec.subtitle) h("div", "chart-sub", container, spec.subtitle);
    const wrap = h("div", "", container);
    wrap.style.position = "relative";
    const rows = spec.rows;
    const bh = 18, gap = 12;
    const height = rows.length * (bh + gap) + 30;
    const svg = el("svg", { height, role: "img", "aria-label": spec.title || "bar chart" }, wrap);
    const tip = h("div", "tooltip", wrap);
    const fmt = spec.fmt || ((v) => (+v).toFixed(3));
    function draw() {
      const W = Math.max(280, wrap.clientWidth);
      svg.setAttribute("width", W); svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const lw = Math.min(170, W * 0.36), rw = 64;
      const iw = W - lw - rw;
      let lo = Math.min(0, ...rows.map((r) => r.value)), hi = Math.max(0, ...rows.map((r) => r.value));
      if (spec.max !== undefined) hi = spec.max;
      const X = (v) => lw + ((v - lo) / (hi - lo || 1)) * iw;
      const ax = el("g", { class: "axis" }, svg);
      for (const v of niceTicks(lo, hi, 4)) {
        el("line", { class: "gridline", x1: X(v), x2: X(v), y1: 0, y2: height - 22 }, ax);
        el("text", { x: X(v), y: height - 6, "text-anchor": "middle" }, ax).textContent = fmtCompact(v);
      }
      rows.forEach((r, i) => {
        const y = i * (bh + gap) + 4;
        el("text", { class: "direct-label", x: lw - 8, y: y + bh / 2 + 4, "text-anchor": "end" }, svg).textContent = r.label;
        const x0 = X(Math.min(0, r.value)), x1 = X(Math.max(0, r.value));
        const w = Math.max(1, x1 - x0);
        const rad = Math.min(4, w / 2);
        // 4px rounded data end, square at the baseline
        const pos = r.value >= 0;
        const d = pos
          ? `M${x0},${y}H${x1 - rad}Q${x1},${y} ${x1},${y + rad}V${y + bh - rad}Q${x1},${y + bh} ${x1 - rad},${y + bh}H${x0}Z`
          : `M${x1},${y}H${x0 + rad}Q${x0},${y} ${x0},${y + rad}V${y + bh - rad}Q${x0},${y + bh} ${x0 + rad},${y + bh}H${x1}Z`;
        const bar = el("path", { d, fill: cssVar(r.color || spec.color || "--series-1") }, svg);
        // negative bars: value sits right of the zero line so it never collides with the row label
        el("text", { class: "bar-label", x: pos ? x1 + 6 : X(0) + 6, y: y + bh / 2 + 4, "text-anchor": "start" }, svg).textContent = fmt(r.value);
        const hitR = el("rect", { x: lw, y: y - gap / 2, width: iw + rw, height: bh + gap, fill: "transparent" }, svg);
        const on = () => {
          bar.setAttribute("opacity", 0.8);
          tip.textContent = "";
          h("div", "tt-head", tip, r.label);
          const row = h("div", "tt-row", tip);
          h("b", "", row, fmt(r.value));
          if (r.note) h("span", "", row, r.note);
          tip.style.opacity = 1;
          tip.style.left = Math.min(W - 200, x1 + 10) + "px";
          tip.style.top = y + "px";
        };
        hitR.addEventListener("pointermove", on);
        hitR.addEventListener("pointerleave", () => { bar.setAttribute("opacity", 1); tip.style.opacity = 0; });
      });
      if (spec.refAt !== undefined) {
        el("line", { x1: X(spec.refAt), x2: X(spec.refAt), y1: 0, y2: height - 22, stroke: cssVar("--reference"), "stroke-dasharray": "5 4" }, svg);
      }
    }
    draw();
    if (window.ResizeObserver) new ResizeObserver(draw).observe(wrap);
  }


  /* ------------------------------------------------------------------ */
  /* scatter: spec.groups = [{name, color, shape, points:[{x, y, label}]}],
     optional spec.lines = [{name, color, x[], y[], dash}] drawn underneath. */
  function scatter(container, spec) {
    container.textContent = "";
    container.classList.add("chart");
    if (spec.title) h("div", "chart-title", container, spec.title);
    if (spec.subtitle) h("div", "chart-sub", container, spec.subtitle);
    const groups = spec.groups.filter((g) => g.points.length);
    const lines = spec.lines || [];
    const lg = h("div", "legend", container);
    for (const g of groups) {
      const k = h("span", "key", lg);
      const sw = h("span", "", k);
      sw.style.cssText = `width:10px;height:10px;display:inline-block;background:${cssVar(g.color)};border-radius:${g.shape === "square" ? "2px" : "50%"}`;
      h("span", "", k, g.name);
    }
    for (const l of lines) {
      const k = h("span", "key", lg);
      const sw = h("span", "swatch-line" + (l.dash ? " dash" : ""), k);
      sw.style.borderTopColor = cssVar(l.color);
      h("span", "", k, l.name);
    }
    const wrap = h("div", "", container);
    wrap.style.position = "relative";
    const height = spec.height || 300;
    const m = { t: 12, r: 18, b: 40, l: 58 };
    const svg = el("svg", { height, role: "img", "aria-label": spec.title || "scatter" }, wrap);
    const tip = h("div", "tooltip", wrap);
    const all = groups.flatMap((g) => g.points).concat(lines.flatMap((l) => l.x.map((x, i) => ({ x, y: l.y[i] }))));
    let xmin = Math.min(...all.map((p) => p.x)), xmax = Math.max(...all.map((p) => p.x));
    let ymin = Math.min(...all.map((p) => p.y)), ymax = Math.max(...all.map((p) => p.y));
    const px = (xmax - xmin) * 0.08 || 1, py = (ymax - ymin) * 0.1 || 1;
    xmin = spec.xMin ?? xmin - px; xmax = spec.xMax ?? xmax + px;
    ymin = spec.yMin ?? ymin - py; ymax = spec.yMax ?? ymax + py;
    const xFmt = spec.xFmt || fmtCompact, yFmt = spec.yFmt || fmtCompact;
    function draw() {
      const W = Math.max(280, wrap.clientWidth);
      svg.setAttribute("width", W); svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const iw = W - m.l - m.r, ih = height - m.t - m.b;
      const X = (x) => m.l + ((x - xmin) / (xmax - xmin)) * iw;
      const Y = (y) => m.t + ih - ((y - ymin) / (ymax - ymin)) * ih;
      const ax = el("g", { class: "axis" }, svg);
      for (const v of niceTicks(ymin, ymax, 5)) {
        if (v < ymin || v > ymax) continue;
        el("line", { class: "gridline", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }, ax);
        el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, ax).textContent = yFmt(v);
      }
      for (const v of niceTicks(xmin, xmax, Math.max(3, Math.floor(iw / 90)))) {
        if (v < xmin || v > xmax) continue;
        el("line", { class: "gridline", x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih }, ax);
        el("text", { x: X(v), y: m.t + ih + 16, "text-anchor": "middle" }, ax).textContent = xFmt(v);
      }
      if (spec.xLabel) el("text", { class: "axis-label", x: m.l + iw / 2, y: height - 4, "text-anchor": "middle" }, svg).textContent = spec.xLabel;
      if (spec.yLabel) el("text", { class: "axis-label", x: 12, y: m.t + ih / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + ih / 2})` }, svg).textContent = spec.yLabel;
      for (const r of spec.refLines || []) {
        if (r.x !== undefined) el("line", { x1: X(r.x), x2: X(r.x), y1: m.t, y2: m.t + ih, stroke: cssVar("--reference"), "stroke-dasharray": "5 4" }, svg);
        if (r.y !== undefined) el("line", { x1: m.l, x2: m.l + iw, y1: Y(r.y), y2: Y(r.y), stroke: cssVar("--reference"), "stroke-dasharray": "5 4" }, svg);
        if (r.label) el("text", { class: "ref-label", x: r.x !== undefined ? X(r.x) + 4 : m.l + 4, y: r.y !== undefined ? Y(r.y) - 5 : m.t + 10 }, svg).textContent = r.label;
      }
      for (const l of lines) {
        let d = "";
        l.x.forEach((x, i) => { d += (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(l.y[i]).toFixed(1); });
        el("path", { d, fill: "none", stroke: cssVar(l.color), "stroke-width": 1.6, "stroke-dasharray": l.dash ? "6 4" : "none" }, svg);
      }
      for (const g of groups) {
        for (const p of g.points) {
          const cx = X(p.x), cy = Y(p.y);
          const mark = g.shape === "square"
            ? el("rect", { x: cx - 5, y: cy - 5, width: 10, height: 10, rx: 2, fill: cssVar(g.color), stroke: cssVar("--surface-1"), "stroke-width": 2 }, svg)
            : el("circle", { cx, cy, r: 5.5, fill: cssVar(g.color), stroke: cssVar("--surface-1"), "stroke-width": 2 }, svg);
          if (p.tag) el("text", { class: "direct-label", x: cx + 9, y: cy + 4 }, svg).textContent = p.tag;
          const hit = el("circle", { cx, cy, r: 13, fill: "transparent" }, svg);
          hit.addEventListener("pointerenter", () => {
            mark.setAttribute("opacity", 0.75);
            tip.textContent = "";
            h("div", "tt-head", tip, g.name + (p.label ? " · " + p.label : ""));
            const r1 = h("div", "tt-row", tip); h("b", "", r1, xFmt(p.x)); h("span", "", r1, spec.xName || "x");
            const r2 = h("div", "tt-row", tip); h("b", "", r2, yFmt(p.y)); h("span", "", r2, spec.yName || "y");
            tip.style.opacity = 1;
            tip.style.left = Math.min(W - 180, cx + 14) + "px"; tip.style.top = Math.max(0, cy - 30) + "px";
          });
          hit.addEventListener("pointerleave", () => { mark.setAttribute("opacity", 1); tip.style.opacity = 0; });
        }
      }
    }
    draw();
    if (window.ResizeObserver) new ResizeObserver(draw).observe(wrap);
  }

  /* ---------------- shared page helpers ---------------- */
  function themeToggle(btn) {
    const root = document.documentElement;
    const saved = (() => { try { return localStorage.getItem("theme"); } catch (e) { return null; } })();
    if (saved) root.setAttribute("data-theme", saved);
    const label = () => {
      const dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      btn.textContent = dark ? "Light mode" : "Dark mode";
      return dark;
    };
    label();
    btn.addEventListener("click", () => {
      const next = label() ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
      label();
      window.dispatchEvent(new Event("themechange"));
    });
  }
  function segmented(container, options, value, onChange) {
    container.textContent = "";
    container.classList.add("seg");
    container.setAttribute("role", "group");
    const btns = options.map((o) => {
      const b = h("button", "", container, o.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(o.value === value));
      b.addEventListener("click", () => {
        btns.forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        onChange(o.value);
      });
      return b;
    });
  }
  function table(container, head, rows, numericFrom) {
    container.textContent = "";
    const wrap = h("div", "table-wrap", container);
    const t = h("table", "", wrap);
    const tr = h("tr", "", h("thead", "", t));
    head.forEach((c, i) => { const th = h("th", i >= (numericFrom ?? 1) ? "num" : "", tr, c); });
    const tb = h("tbody", "", t);
    for (const r of rows) {
      const row = h("tr", "", tb);
      r.forEach((c, i) => {
        const td = h("td", i >= (numericFrom ?? 1) ? "num" : "", row);
        if (c && typeof c === "object") { td.textContent = c.text; if (c.best) td.classList.add("best"); if (c.chip) { td.textContent = ""; h("span", "chip " + c.chip, td, c.text); } }
        else td.textContent = c;
      });
    }
    return t;
  }

  window.Charts = { line, hbar, scatter, themeToggle, segmented, table, fmtCompact };
})();
