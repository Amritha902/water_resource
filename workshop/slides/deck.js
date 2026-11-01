// Slide engine and animations for workshop/slides/index.html.  No libraries.
"use strict";

// ───────────────────────────── colours and helpers
const C = {ink:"#1d2433", ink2:"#4b5563", muted:"#8a919c", line:"#e3e1db", bg:"#fbfaf7", soft:"#f1efe9",
  blue:"#2563eb", orange:"#ea580c", green:"#15803d", purple:"#7c3aed", red:"#dc2626", teal:"#0d9488",
  lblue:"#dbeafe", lorange:"#ffedd5", lgreen:"#dcfce7", lred:"#fee2e2"};
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
const fmt = (x, d = 2) => (x >= 0 ? " " : "") + x.toFixed(d);
const thousands = x => Math.round(x).toLocaleString("en-US").replace(/,/g, " ");

function seeded(seed) {           // small deterministic RNG, so the slides look the same every time
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
function gauss(rnd) { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }

function canvas(id) {             // HiDPI canvas; draw in CSS pixels
  const cv = document.getElementById(id);
  const w = +cv.getAttribute("width"), h = +cv.getAttribute("height");
  if (!cv._ready) {
    const r = Math.max(2, window.devicePixelRatio || 1);
    cv.width = w * r; cv.height = h * r; cv.style.width = w + "px"; cv.style.height = h + "px";
    cv._ready = true; cv._r = r;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(cv._r, 0, 0, cv._r, 0, 0);
  ctx.w = w; ctx.h = h;
  return ctx;
}
function clear(ctx, col = C.bg) { ctx.fillStyle = col; ctx.fillRect(0, 0, ctx.w, ctx.h); }
function text(ctx, s, x, y, o = {}) {
  ctx.font = `${o.weight || 400} ${o.size || 14}px ${o.mono ? "Consolas,Menlo,monospace" : "Segoe UI,system-ui,sans-serif"}`;
  ctx.fillStyle = o.color || C.ink; ctx.textAlign = o.align || "left"; ctx.textBaseline = o.base || "middle";
  ctx.fillText(s, x, y);
}
function rrect(ctx, x, y, w, h, r, fill, stroke, lw = 1.5) {
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}
function line(ctx, pts, col, lw = 2, dash) {
  ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
}
function arrow(ctx, x1, y1, x2, y2, col, lw = 2, head = 9) {
  line(ctx, [[x1, y1], [x2, y2]], col, lw);
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(a - .4), y2 - head * Math.sin(a - .4));
  ctx.lineTo(x2 - head * Math.cos(a + .4), y2 - head * Math.sin(a + .4));
  ctx.closePath(); ctx.fillStyle = col; ctx.fill();
}
function dot(ctx, x, y, r, fill, stroke) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
}
// a scaled plot area: returns mappers
function axes(x0, y0, w, h, xr, yr) {
  return {x: v => x0 + (v - xr[0]) / (xr[1] - xr[0]) * w, y: v => y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h, x0, y0, w, h};
}
function frame(ctx, ax, o = {}) {
  ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  ctx.strokeRect(ax.x0, ax.y0, ax.w, ax.h);
  if (o.xlabel) text(ctx, o.xlabel, ax.x0 + ax.w / 2, ax.y0 + ax.h + 22, {align: "center", color: C.ink2, size: 13});
  if (o.ylabel) { ctx.save(); ctx.translate(ax.x0 - 40, ax.y0 + ax.h / 2); ctx.rotate(-Math.PI / 2);
    text(ctx, o.ylabel, 0, 0, {align: "center", color: C.ink2, size: 13}); ctx.restore(); }
}
// animation loop that stops itself when the slide is left
function runner(draw) {
  let id = 0, t0 = performance.now(), alive = true;
  const tick = now => { if (!alive) return; draw((now - t0) / 1000); id = requestAnimationFrame(tick); };
  id = requestAnimationFrame(tick);
  return {stop() { alive = false; cancelAnimationFrame(id); }, restart() { t0 = performance.now(); }};
}

// ───────────────────────────── engine
const slides = $$(".slide");
const A = {};                     // animations by name: {start(el), stop(), frag(n)}
let cur = 0, frag = 0, live = null;

function fitStage() {
  const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
  $("#stage").style.transform = `translate(-50%,-50%) scale(${s})`;
}
function frags() { return $$(".frag", slides[cur]); }
function show(i, f = 0) {
  if (live && live.stop) live.stop();
  cur = clamp(i, 0, slides.length - 1);
  slides.forEach(s => s.classList.toggle("active", s === slides[cur]));
  const fr = frags(); frag = f === "all" ? fr.length : f;
  fr.forEach((el, k) => el.classList.toggle("on", k < frag));
  const name = slides[cur].dataset.anim;
  live = name && A[name] ? A[name] : null;
  if (live) live.start(slides[cur]);
  if (live && live.frag) live.frag(frag);
  $("#bar").style.width = (100 * (cur + 1) / slides.length) + "%";
  $("#count").textContent = `${cur + 1} / ${slides.length}`;
  history.replaceState(null, "", "#" + (cur + 1));
}
function next() {
  const fr = frags();
  if (frag < fr.length) { fr[frag++].classList.add("on"); if (live && live.frag) live.frag(frag); }
  else show(cur + 1);
}
function prev() {
  if (frag > 0) { frags()[--frag].classList.remove("on"); if (live && live.frag) live.frag(frag); }
  else if (cur > 0) show(cur - 1, "all");
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); next(); }
  else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); prev(); }
  else if (e.key === "Home") show(0);
  else if (e.key === "End") show(slides.length - 1, "all");
  else if (e.key === "r" || e.key === "R") show(cur, frag);
  else if (e.key === "f" || e.key === "F") { try { const p = !document.fullscreenElement ? document.documentElement.requestFullscreen() : document.exitFullscreen(); if (p && p.catch) p.catch(() => {}); } catch (_) {} }
});
window.addEventListener("resize", fitStage);

// ═════════════════════════════ 1. the plant
A.plant = {
  start() {
    const sl = $("#kla-slider"), rd = $("#kla-read");
    const rnd = seeded(3);
    // geometry
    const Y = 150, H = 110, tanks = [];
    let x = 120;
    const widths = [120, 120, 140, 140, 140];
    for (let i = 0; i < 5; i++) { tanks.push({x, w: widths[i], aer: i >= 2}); x += widths[i] + 18; }
    const st = {x: x + 30, w: 150};
    const main = [[20, Y + 60], [tanks[0].x + 20, Y + 60], [tanks[4].x + tanks[4].w - 10, Y + 60], [st.x + 20, Y + 60], [st.x + st.w / 2, Y + 50], [st.x + st.w + 90, Y + 20]];
    const recyc = [[tanks[4].x + tanks[4].w - 30, Y], [tanks[4].x + tanks[4].w - 30, Y - 50], [tanks[0].x + 30, Y - 50], [tanks[0].x + 30, Y]];
    const sludge = [[st.x + st.w / 2, Y + H + 20], [st.x + st.w / 2, Y + H + 60], [tanks[0].x + 50, Y + H + 60], [tanks[0].x + 50, Y + H]];
    const pathLen = p => p.slice(1).reduce((s, q, i) => s + Math.hypot(q[0] - p[i][0], q[1] - p[i][1]), 0);
    const at = (p, s) => { for (let i = 1; i < p.length; i++) { const l = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); if (s <= l) return [lerp(p[i - 1][0], p[i][0], s / l), lerp(p[i - 1][1], p[i][1], s / l)]; s -= l; } return p[p.length - 1]; };
    const paths = [[main, pathLen(main), 40, C.orange], [recyc, pathLen(recyc), 16, C.purple], [sludge, pathLen(sludge), 12, "#92400e"]];
    const parts = paths.map(([p, L, n]) => Array.from({length: n}, () => rnd() * L));
    let bubbles = [];
    let so = 2;
    this.r = runner(t => {
      const kla = +sl.value;
      // tank-5 oxygen at a fixed demand: KLa (8 − S) = OUR  →  S = 8 − OUR/KLa
      const target = Math.max(0, 8 - 700 / kla);
      so += (target - so) * 0.05;
      const energy = 8 / 1800 * 1333 * (240 + 240 + kla);
      rd.textContent = `KLa5 = ${kla} 1/d · oxygen ≈ ${so.toFixed(2)} mg/L · aeration ≈ ${thousands(energy)} kWh/d`;
      const ctx = canvas("cv-plant"); clear(ctx, "rgba(0,0,0,0)"); ctx.clearRect(0, 0, ctx.w, ctx.h);
      // pipes
      line(ctx, main, "#cbd5e1", 8); line(ctx, recyc, "#ddd6fe", 6); line(ctx, sludge, "#e7d3bd", 6);
      // tanks
      tanks.forEach((tk, i) => {
        rrect(ctx, tk.x, Y, tk.w, H, 8, tk.aer ? "#dbeafe" : "#efe6d8", tk.aer ? "#93c5fd" : "#d6c7ae", 2);
        text(ctx, `tank ${i + 1}`, tk.x + tk.w / 2, Y + H + 14, {align: "center", size: 13, color: C.ink2});
        text(ctx, tk.aer ? "aerated" : "anoxic", tk.x + tk.w / 2, Y + 16, {align: "center", size: 12, color: C.muted});
      });
      // settler
      ctx.beginPath(); ctx.moveTo(st.x, Y); ctx.lineTo(st.x + st.w, Y); ctx.lineTo(st.x + st.w - 40, Y + H + 20); ctx.lineTo(st.x + 40, Y + H + 20); ctx.closePath();
      ctx.fillStyle = "#f3f4f6"; ctx.fill(); ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 2; ctx.stroke();
      text(ctx, "settler", st.x + st.w / 2, Y + H + 36, {align: "center", size: 13, color: C.ink2});
      // bubbles
      tanks.forEach((tk, i) => {
        if (!tk.aer) return;
        const rate = i === 4 ? kla / 240 : 1;
        if (rnd() < 0.5 * rate) bubbles.push({x: tk.x + 10 + rnd() * (tk.w - 20), y: Y + H - 6, v: 0.6 + rnd() * 0.8});
      });
      bubbles = bubbles.filter(b => b.y > Y + 24);
      bubbles.forEach(b => { b.y -= b.v; dot(ctx, b.x + Math.sin(b.y / 6) * 1.5, b.y, 2.4, "rgba(255,255,255,.9)", "#60a5fa"); });
      // particles
      paths.forEach(([p, L, n, col], j) => {
        parts[j] = parts[j].map(s => (s + 1.4) % L);
        parts[j].forEach(s => { const q = at(p, s); dot(ctx, q[0], q[1], 3, col); });
      });
      // labels
      text(ctx, "influent", 22, Y + 40, {size: 14, weight: 600, color: C.orange});
      text(ctx, "Q, ammonium, COD", 22, Y + 84, {size: 12, color: C.ink2});
      text(ctx, "effluent", st.x + st.w + 30, Y - 4, {size: 14, weight: 600, color: C.green});
      text(ctx, "internal recycle Qa (pump)", (recyc[1][0] + recyc[2][0]) / 2, Y - 64, {align: "center", size: 13, color: C.purple, weight: 600});
      text(ctx, "returned sludge", (sludge[1][0] + sludge[2][0]) / 2, Y + H + 74, {align: "center", size: 13, color: "#92400e"});
      // sensor tags
      const t5 = tanks[4], t2 = tanks[1];
      rrect(ctx, t5.x + 12, Y + 34, t5.w - 24, 46, 6, "#fff", C.blue, 1.5);
      text(ctx, "sensor S_O,5", t5.x + t5.w / 2, Y + 47, {align: "center", size: 12, color: C.blue, weight: 600});
      text(ctx, `${so.toFixed(2)} mg/L`, t5.x + t5.w / 2, Y + 66, {align: "center", size: 15, mono: true, color: Math.abs(so - 2) < .15 ? C.green : C.red});
      rrect(ctx, t2.x + 12, Y + 40, t2.w - 24, 30, 6, "#fff", C.purple, 1.5);
      text(ctx, "sensor S_NO,2", t2.x + t2.w / 2, Y + 55, {align: "center", size: 12, color: C.purple, weight: 600});
      text(ctx, "air blower → KLa5", t5.x + t5.w / 2, Y + H + 34, {align: "center", size: 12.5, color: C.blue, weight: 600});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 4. what is a model: fit a line by gradient descent
A.fit = {
  start() {
    const rnd = seeded(11);
    const pts = Array.from({length: 26}, () => { const x = 0.1 + rnd() * 0.85; return [x, 0.15 + 0.7 * x + 0.07 * gauss(rnd)]; });
    let w = -0.2, b = 0.75, step = 0, playing = false, hist = [];
    const lr = 0.6;
    const loss = () => pts.reduce((s, [x, y]) => s + (w * x + b - y) ** 2, 0) / pts.length;
    const gd = () => {
      let gw = 0, gb = 0;
      pts.forEach(([x, y]) => { const e = w * x + b - y; gw += 2 * e * x; gb += 2 * e; });
      w -= lr * gw / pts.length; b -= lr * gb / pts.length; step++; hist.push(loss());
    };
    hist.push(loss());
    const panel = $("#fit-panel");
    $("#fit-play").onclick = () => { playing = !playing; $("#fit-play").textContent = playing ? "❚❚ pause" : "▶ train"; };
    $("#fit-step").onclick = () => gd();
    $("#fit-reset").onclick = () => { w = -0.2; b = 0.75; step = 0; hist = [loss()]; };
    let acc = 0;
    this.r = runner(t => {
      if (playing && step < 300) { acc++; if (acc % 4 === 0) gd(); }
      const ctx = canvas("cv-fit"); clear(ctx);
      const ax = axes(60, 20, 380, 360, [0, 1], [0, 1.05]);
      frame(ctx, ax, {xlabel: "x: inflow (scaled)", ylabel: "y: ammonium load (scaled)"});
      // residuals
      pts.forEach(([x, y]) => line(ctx, [[ax.x(x), ax.y(y)], [ax.x(x), ax.y(w * x + b)]], "rgba(220,38,38,.55)", 1.5));
      line(ctx, [[ax.x(0), ax.y(b)], [ax.x(1), ax.y(w + b)]], C.blue, 3);
      pts.forEach(([x, y]) => dot(ctx, ax.x(x), ax.y(y), 4.5, C.ink));
      text(ctx, `ŷ = ${w.toFixed(2)}·x + ${b.toFixed(2)}`, ax.x(0.04), ax.y(1.0), {size: 16, mono: true, color: C.blue, weight: 600});
      // loss curve
      const lx = axes(480, 40, 160, 150, [0, Math.max(40, hist.length)], [0, hist[0] * 1.05]);
      frame(ctx, lx);
      text(ctx, "loss over steps", 560, 28, {align: "center", size: 13, color: C.ink2});
      line(ctx, hist.map((v, i) => [lx.x(i), lx.y(v)]), C.red, 2);
      // knobs
      const knob = (cx, cy, val, lab) => {
        dot(ctx, cx, cy, 30, "#fff", C.line);
        const a = -Math.PI * 1.25 + clamp((val + 1) / 2, 0, 1) * Math.PI * 1.5;
        line(ctx, [[cx, cy], [cx + 24 * Math.cos(a), cy + 24 * Math.sin(a)]], C.blue, 4);
        text(ctx, lab, cx, cy + 46, {align: "center", size: 14, mono: true});
      };
      knob(520, 280, w, `w = ${w.toFixed(2)}`); knob(605, 280, b, `b = ${b.toFixed(2)}`);
      text(ctx, "the knobs (parameters)", 562, 230, {align: "center", size: 13, color: C.ink2});
      panel.innerHTML = `<div>step</div><div class="mono">${step}</div><div>w, b</div><div class="mono">${w.toFixed(3)}, ${b.toFixed(3)}</div><div>loss (MSE)</div><div class="mono">${loss().toFixed(5)}</div>`;
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 5. collect: series drawn as samples arrive
A.collect = {
  start() {
    const tb = $("#collect-table");
    $$("tr", tb).slice(1).forEach(r => r.remove());
    const n = SERIES.dry.length, dur = 12;
    let lastRow = -1;
    const rows = [["dry", C.blue], ["rain", C.teal], ["storm", C.purple]];
    this.r = runner(t => {
      const k = Math.min(n, Math.floor((t % (dur + 3)) / dur * n));
      const ctx = canvas("cv-collect"); clear(ctx);
      rows.forEach(([wth, col], j) => {
        const ax = axes(70, 14 + j * 134, 670, 104, [0, n], [0, 40000]);
        frame(ctx, ax);
        ctx.fillStyle = "rgba(37,99,235,.05)"; ctx.fillRect(ax.x(0), ax.y0, ax.x(n / 2) - ax.x(0), ax.h);
        if (j === 0) { text(ctx, "week 1 (same in all files)", ax.x(n / 4), ax.y0 + 12, {align: "center", size: 12, color: C.muted});
          text(ctx, "week 2", ax.x(3 * n / 4), ax.y0 + 12, {align: "center", size: 12, color: C.muted}); }
        text(ctx, wth, 12, ax.y0 + 20, {size: 15, weight: 700, color: col});
        text(ctx, "40k", 64, ax.y(40000) + 6, {align: "right", size: 11, color: C.muted});
        text(ctx, "0", 64, ax.y(0) - 6, {align: "right", size: 11, color: C.muted});
        const s = SERIES[wth];
        line(ctx, s.slice(0, k).map((v, i) => [ax.x(i), ax.y(v)]), col, 1.8);
        if (k > 0 && k < n) dot(ctx, ax.x(k - 1), ax.y(s[k - 1]), 4, col);
        for (let d = 0; d <= 14; d += 2) text(ctx, `${d}`, ax.x(d * 24), ax.y0 + ax.h + 10, {align: "center", size: 11, color: C.muted});
      });
      text(ctx, "day", 400, 414, {align: "center", size: 12, color: C.ink2});
      text(ctx, "inflow Q [m³/d]", 70, 6, {size: 12, color: C.ink2, base: "top"});
      if (k !== lastRow && k > 0 && k < n && k % 2 === 0) {
        lastRow = k;
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${(k / 24).toFixed(3)}</td><td>${thousands(SERIES.storm[k])}</td><td>${SERIES.storm_nh[k].toFixed(1)}</td>`;
        tr.style.background = "#fef9c3"; setTimeout(() => tr.style.background = "", 400);
        tb.insertBefore(tr, tb.rows[1] || null);
        while (tb.rows.length > 7) tb.deleteRow(tb.rows.length - 1);
      }
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 7. clean: highlight problem rows with each fragment
A.clean = {
  start(el) { this.el = el; },
  frag(n) {
    const tb = $("#clean-table");
    $$("tr", tb).forEach(r => r.className = "");
    const fr = $$(".frag", this.el);
    fr.forEach((f, i) => {
      if (i >= n || !f.dataset.hl) return;
      f.dataset.hl.split(",").forEach(k => {
        const r = $(`tr[data-r="${k}"]`, tb);
        if (r) r.className = i === n - 1 ? "bad" : "fixed";
      });
    });
  },
  stop() {}
};

// ═════════════════════════════ 8. zero-order hold
A.zoh = {
  start() {
    const rnd = seeded(5);
    const ns = 9, samples = Array.from({length: ns}, (_, i) => 20000 + 6000 * Math.sin(i / 2.2) + 1800 * gauss(rnd));
    this.r = runner(t => {
      const ctx = canvas("cv-zoh"); clear(ctx);
      const ax = axes(80, 20, 1040, 280, [0, (ns - 1) * 20], [8000, 34000]);
      frame(ctx, ax, {xlabel: "control step (45 s each); one sensor sample every 20 steps = 15 min", ylabel: "Q [m³/d]"});
      const ph = (t % 9) / 7, kmax = Math.floor(clamp(ph, 0, 1) * (ns - 1) * 20);
      // held values
      for (let k = 0; k <= kmax; k++) {
        const v = samples[Math.floor(k / 20)];
        ctx.fillStyle = "rgba(37,99,235,.18)"; ctx.fillRect(ax.x(k) - 1.6, ax.y(v), 3.2, ax.y(8000) - ax.y(v));
        dot(ctx, ax.x(k), ax.y(v), 2, C.blue);
      }
      // step line
      const pts = [];
      for (let i = 0; i < ns - 1 && i * 20 <= kmax; i++) { pts.push([ax.x(i * 20), ax.y(samples[i])], [ax.x(Math.min((i + 1) * 20, kmax)), ax.y(samples[i])]); }
      if (pts.length) line(ctx, pts, C.blue, 2);
      samples.forEach((v, i) => { dot(ctx, ax.x(i * 20), ax.y(v), 6.5, C.orange, "#fff"); });
      text(ctx, "● sensor sample (15 min)", 900, 40, {color: C.orange, size: 14, weight: 600});
      text(ctx, "▮ value used at each 45-s step", 900, 62, {color: C.blue, size: 14, weight: 600});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 10. transform and scale: histograms
A.scale = {
  start(el) {
    const raw = [...SERIES.dry, ...SERIES.rain, ...SERIES.storm];
    const lg = raw.map(Math.log);
    const m = lg.reduce((a, b) => a + b, 0) / lg.length, sd = Math.sqrt(lg.reduce((a, b) => a + (b - m) ** 2, 0) / lg.length);
    const z = lg.map(v => (v - m) / sd);
    const states = [
      {vals: raw, lab: "Q [m³/d]", fmt: v => thousands(v)},
      {vals: lg, lab: "log Q", fmt: v => v.toFixed(2)},
      {vals: z, lab: "z = (log Q − mean) / std", fmt: v => v.toFixed(1)}];
    const nb = 30;
    const hist = s => { const lo = Math.min(...s.vals), hi = Math.max(...s.vals), h = new Array(nb).fill(0);
      s.vals.forEach(v => h[Math.min(nb - 1, Math.floor((v - lo) / (hi - lo) * nb))]++);
      const mean = s.vals.reduce((a, b) => a + b, 0) / s.vals.length, std = Math.sqrt(s.vals.reduce((a, b) => a + (b - mean) ** 2, 0) / s.vals.length);
      return {h, lo, hi, mean, std}; };
    const H = states.map(hist);
    let target = 0, from = 0, t0 = 0, now = 0;
    $$("button[data-s]", el).forEach(b => b.onclick = () => {
      $$("button[data-s]", el).forEach(x => x.classList.remove("on")); b.classList.add("on");
      from = target; target = +b.dataset.s; t0 = now;
    });
    const hmax = Math.max(...H.flatMap(x => x.h));
    this.r = runner(t => {
      now = t;
      const p = ease(clamp((t - t0) / 0.8, 0, 1));
      const A_ = H[from], B = H[target];
      const ctx = canvas("cv-scale"); clear(ctx);
      const ax = axes(50, 20, 640, 250, [0, nb], [0, hmax * 1.1]);
      frame(ctx, ax, {xlabel: states[target].lab});
      for (let i = 0; i < nb; i++) {
        const v = lerp(A_.h[i], B.h[i], p);
        rrect(ctx, ax.x(i) + 1, ax.y(v), ax.w / nb - 2, ax.y(0) - ax.y(v), 2, target === 2 ? C.green : target === 1 ? C.teal : C.blue);
      }
      const S = states[target];
      text(ctx, S.fmt(B.lo), ax.x(0), ax.y0 + ax.h + 10, {size: 12, color: C.muted});
      text(ctx, S.fmt(B.hi), ax.x(nb), ax.y0 + ax.h + 10, {size: 12, color: C.muted, align: "right"});
      text(ctx, `mean ${S.fmt(B.mean)}   std ${S.fmt(B.std)}`, ax.x0 + ax.w - 10, ax.y0 + 18, {align: "right", size: 15, mono: true, weight: 600});
      text(ctx, target === 0 ? "long right tail: storms" : target === 1 ? "tail pulled in" : "centred at 0, spread 1", ax.x0 + ax.w - 10, ax.y0 + 40, {align: "right", size: 13, color: C.ink2});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 11. sliding windows
A.window = {
  start() {
    const s = SERIES.dry, n = s.length, L = 24, Hh = 4;   // hourly here: 24 h back, 4 h ahead (repo: 96 and 8 at 15 min)
    this.r = runner(t => {
      const ctx = canvas("cv-window"); clear(ctx);
      const ax = axes(60, 18, 1060, 130, [0, n], [5000, 36000]);
      frame(ctx, ax);
      text(ctx, "inflow, dry weather, 14 days (hourly)", ax.x0 + 6, ax.y0 + 12, {size: 12, color: C.muted});
      line(ctx, s.map((v, i) => [ax.x(i), ax.y(v)]), C.ink2, 1.4);
      const stride = 6, maxW = Math.floor((n - L - Hh) / stride);
      const k = Math.floor(t * 2.2) % (maxW + 6);
      const st = Math.min(k, maxW) * stride;
      // window boxes
      ctx.fillStyle = "rgba(37,99,235,.14)"; ctx.fillRect(ax.x(st), ax.y0, ax.x(st + L) - ax.x(st), ax.h);
      ctx.fillStyle = "rgba(234,88,12,.2)"; ctx.fillRect(ax.x(st + L), ax.y0, ax.x(st + L + Hh) - ax.x(st + L), ax.h);
      line(ctx, s.slice(st, st + L + 1).map((v, i) => [ax.x(st + i), ax.y(v)]), C.blue, 2.6);
      line(ctx, s.slice(st + L, st + L + Hh + 1).map((v, i) => [ax.x(st + L + i), ax.y(v)]), C.orange, 2.6);
      text(ctx, "input x: the past", ax.x(st + L / 2), ax.y0 - 8, {align: "center", size: 12.5, color: C.blue, weight: 600});
      text(ctx, "target y", ax.x(st + L + Hh) + 34, ax.y0 - 8, {align: "center", size: 12.5, color: C.orange, weight: 600});
      // the stack of examples
      const N = Math.min(k, maxW) + 1;
      const show = Math.min(N, 7);
      text(ctx, "examples so far", 60, 180, {size: 13, color: C.ink2, weight: 600});
      for (let j = 0; j < show; j++) {
        const idx = N - 1 - j, s0 = idx * stride, y0 = 196 + j * 25;
        const wx = axes(60, y0, 300, 20, [0, L + Hh], [5000, 36000]);
        rrect(ctx, 56, y0 - 2, 420, 24, 4, j === 0 ? "#fef9c3" : "#fff", C.line, 1);
        line(ctx, s.slice(s0, s0 + L + 1).map((v, i) => [wx.x(i), wx.y(v)]), C.blue, 1.5);
        line(ctx, s.slice(s0 + L, s0 + L + Hh + 1).map((v, i) => [wx.x(L + i), wx.y(v)]), C.orange, 1.5);
        text(ctx, `#${idx}`, 380, y0 + 10, {size: 12, mono: true, color: C.muted});
      }
      // tensor shapes
      rrect(ctx, 560, 180, 560, 186, 10, "#fff", C.line);
      text(ctx, "what the model receives", 580, 202, {size: 14, color: C.ink2, weight: 600});
      text(ctx, `N = ${N} examples (stride ${stride} h here)`, 580, 234, {size: 16, mono: true});
      text(ctx, "X: (N, 96, 3)", 580, 270, {size: 22, mono: true, color: C.blue, weight: 700});
      text(ctx, "N windows × 96 steps × 3 channels (Q, NH, COD)", 780, 270, {size: 12.5, color: C.ink2});
      text(ctx, "y: (N, 8)", 580, 310, {size: 22, mono: true, color: C.orange, weight: 700});
      text(ctx, "N windows × the next 8 values of Q", 780, 310, {size: 12.5, color: C.ink2});
      text(ctx, "Animation: 24 h back, 4 h ahead, hourly. Repo: 96 × 15 min back, 8 × 15 min ahead.", 580, 346, {size: 12, color: C.muted});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 12. split by time vs random
A.split = {
  start(el) {
    let mode = "time";
    const bR = $("#split-random"), bT = $("#split-time");
    bR.onclick = () => { mode = "random"; bR.classList.add("on"); bT.classList.remove("on"); };
    bT.onclick = () => { mode = "time"; bT.classList.add("on"); bR.classList.remove("on"); };
    const rnd = seeded(9), nW = 46, assign = Array.from({length: nW}, () => rnd() < 0.7 ? 0 : rnd() < 0.5 ? 1 : 2);
    const cols = [C.blue, C.orange, C.green], names = ["train", "validation", "test"];
    this.r = runner(t => {
      const ctx = canvas("cv-split"); clear(ctx);
      const ax = axes(60, 30, 1060, 60, [0, 100], [0, 1]);
      // the series strip
      const s = SERIES.storm;
      const sx = axes(60, 20, 1060, 70, [0, s.length], [5000, 40000]);
      line(ctx, s.map((v, i) => [sx.x(i), sx.y(v)]), C.ink2, 1.3);
      // windows
      const p = (t % 6) / 4, shown = Math.floor(clamp(p, 0, 1) * nW);
      let leaks = 0;
      for (let i = 0; i < shown; i++) {
        const g = mode === "time" ? (i < 0.7 * nW ? 0 : i < 0.85 * nW ? 1 : 2) : assign[i];
        const x0 = 60 + i * (1060 - 90) / nW, w = 90;
        const y = 110 + (i % 5) * 22;
        let leak = false;
        if (mode === "random" && g !== 0) { for (let j = Math.max(0, i - 4); j <= Math.min(nW - 1, i + 4); j++) if (j !== i && j < shown && assign[j] === 0) leak = true; }
        if (leak) leaks++;
        rrect(ctx, x0, y, w, 16, 4, cols[g] + (leak ? "" : "cc"), leak ? C.red : null, leak ? 2.5 : 0);
      }
      names.forEach((nm, g) => { rrect(ctx, 60 + g * 150, 240, 16, 16, 3, cols[g]); text(ctx, nm, 82 + g * 150, 248, {size: 14}); });
      if (mode === "random") {
        text(ctx, `${leaks} test/validation windows overlap a training window: the model has already seen most of their samples`, 520, 248, {size: 14, color: C.red, weight: 600});
      } else {
        text(ctx, "train on the past · validate later · test last: no overlap except at the two borders", 520, 248, {size: 14, color: C.green, weight: 600});
      }
      text(ctx, "each bar is one window (input + target); bars overlap in time", 60, 280, {size: 12, color: C.muted});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 14. shapes decide the model
const MODELS = [
  {name: "Influent forecaster", file: "src/wwtp/forecast/", input: "(B, 96, 3): 24 h of flow, NH₄, COD at 15 min", output: "(B, 8, 3, 3): 10/50/90 % quantiles for the next 2 h + (B, 3) weather class", model: "causal TCN (dilated 1-D convolutions)", loss: "pinball + 0.2 × cross-entropy", params: 171211,
    why: "Sequence input → convolution over time. Uncertainty wanted → quantile outputs. Must be causal: only past samples."},
  {name: "Digital twin", file: "src/wwtp/twin/", input: "(B, 64, 7): oxygen, nitrate, air, recycle, 3 inlet signals, per 6-min step", output: "(B, 64, 2) next oxygen and nitrate + (B, 64, 1) effluent NH₄", model: "GRU, residual: s(t+1) = s(t) + g(h)", loss: "smooth L1, scheduled sampling", params: 70851,
    why: "Hidden plant state (biomass) → recurrent memory. Must roll out many steps → train on its own predictions too."},
  {name: "Ammonium early warning", file: "src/wwtp/warning/", input: "(B, 60, 17): 6 h of 17 plant signals", output: "(B, 5): P(violation within 12…120 min) + (B, 5, 3) NH₄ quantiles", model: "TCN + survival (hazard) head; GBM baseline on features", loss: "binary cross-entropy + pinball", params: 134932,
    why: "Yes/no event → probabilities. 'Within 2 h' can't be less likely than 'within 30 min' → monotone head by construction."},
  {name: "DDPG actor (paper 2)", file: "src/wwtp/rl/ddpg.py", input: "(B, 3): S_S, S_O, S_NH in tank 5", output: "(B, 1): change of KLa5, in [−1, 1]", model: "MLP 3 → 256 → 256 → 1, tanh", loss: "maximise the critic's Q (RL)", params: 67073,
    why: "Small state vector → MLP. Bounded action → tanh. No correct action to copy → reinforcement learning."},
  {name: "SAC-IL actor (ours)", file: "src/wwtp/sacil/sacil.py", input: "(B, 11): last 3 errors of both loops, both inputs, flow, flow change", output: "(B, 2+2): mean and log-std of the correction", model: "MLP 11 → 128 → 128 → 4", loss: "behaviour cloning (MSE), then SAC", params: 18564,
    why: "An expert exists → copy it first (supervised), then improve with reward."},
  {name: "MAACC as printed (paper 1)", file: "src/wwtp/maacc/exact.py", input: "1 number: the loop's own error", output: "1 number: change of the input", model: "1 hidden layer (10 or 12 units), only the outer weights learn", loss: "adaptive critic, one step at a time", params: 10,
    why: "Only 10–14 numbers learn per agent, from one sample each step. It never moved away from its PID."}];
A.shapes = {
  start() {
    const box = $("#shape-btns"), card = $("#shape-card");
    box.innerHTML = "";
    const pick = i => {
      $$("button", box).forEach((b, j) => b.classList.toggle("on", i === j));
      const m = MODELS[i];
      card.innerHTML = `<h3 style="margin-bottom:10px">${m.name}</h3>
      <div class="kv" style="grid-template-columns:96px 1fr;font-size:15.5px">
        <div>input</div><div class="mono" style="color:${C.blue}">${m.input}</div>
        <div>output</div><div class="mono" style="color:${C.orange}">${m.output}</div>
        <div>model</div><div>${m.model}</div>
        <div>loss</div><div>${m.loss}</div>
        <div>parameters</div><div class="mono">${m.params === 10 ? "10–14 trainable per agent" : thousands(m.params)}</div>
        <div>why</div><div>${m.why}</div>
      </div><p class="file" style="margin-top:10px">${m.file}</p>`;
    };
    MODELS.forEach((m, i) => { const b = document.createElement("button"); b.className = "taskbtn"; b.textContent = m.name; b.onclick = () => pick(i); box.appendChild(b); });
    pick(0);
  },
  stop() {}
};

// ═════════════════════════════ 15. forward pass, neuron by neuron
A.forward = {
  start() {
    const sizes = [3, 4, 4, 1], names = ["inflow Q", "ammonium", "COD"];
    let rnd = seeded(21);
    const W = [], B = [];
    for (let l = 1; l < sizes.length; l++) {
      W.push(Array.from({length: sizes[l]}, () => Array.from({length: sizes[l - 1]}, () => +(gauss(rnd) * 0.8).toFixed(2))));
      B.push(Array.from({length: sizes[l]}, () => +(gauss(rnd) * 0.3).toFixed(2)));
    }
    let x = [1.2, -0.4, 0.7];
    // the sequence of steps: (layer, neuron, k inputs summed so far)
    const steps = [];
    for (let l = 1; l < sizes.length; l++) for (let j = 0; j < sizes[l]; j++) { for (let k = 0; k <= sizes[l - 1]; k++) steps.push([l, j, k]); steps.push([l, j, "act"]); }
    let si = -1, playing = false, acc = 0;
    const act = () => { const a = [x]; for (let l = 1; l < sizes.length; l++) { a.push(W[l - 1].map((row, j) => { const z = row.reduce((s, w, k) => s + w * a[l - 1][k], B[l - 1][j]); return l === sizes.length - 1 ? z : Math.max(0, z); })); } return a; };
    const zval = (a, l, j, k) => { let z = B[l - 1][j]; for (let i = 0; i < k; i++) z += W[l - 1][j][i] * a[l - 1][i]; return z; };
    const done = (l, j) => { if (si < 0) return false; const [L, J, K] = steps[si]; return L > l || (L === l && (J > j || (J === j && K === "act"))); };
    $("#fw-play").onclick = () => { playing = !playing; $("#fw-play").textContent = playing ? "❚❚ pause" : "▶ play"; if (si >= steps.length - 1) si = -1; };
    $("#fw-step").onclick = () => { si = Math.min(steps.length - 1, si + 1); };
    $("#fw-reset").onclick = () => { si = -1; };
    $("#fw-new").onclick = () => { x = x.map(() => +(gauss(rnd) * 1.1).toFixed(2)); si = -1; };
    const panel = $("#fw-panel");
    this.r = runner(t => {
      if (playing) { acc++; if (acc % 22 === 0) { if (si < steps.length - 1) si++; else { playing = false; $("#fw-play").textContent = "▶ play"; } } }
      const a = act();
      const ctx = canvas("cv-forward"); clear(ctx);
      const X = l => 105 + l * 165, Yn = (l, j) => 235 + (j - (sizes[l] - 1) / 2) * 92;
      const labels = ["input layer", "hidden layer 1 (ReLU)", "hidden layer 2 (ReLU)", "output"];
      labels.forEach((s, l) => text(ctx, s, X(l), 24, {align: "center", size: 13, color: C.ink2, weight: 600}));
      text(ctx, `${sizes[1]}×${sizes[0]} + ${sizes[1]}`, (X(0) + X(1)) / 2, 452, {align: "center", size: 12, mono: true, color: C.muted});
      text(ctx, `${sizes[2]}×${sizes[1]} + ${sizes[2]}`, (X(1) + X(2)) / 2, 452, {align: "center", size: 12, mono: true, color: C.muted});
      text(ctx, `${sizes[3]}×${sizes[2]} + ${sizes[3]}`, (X(2) + X(3)) / 2, 452, {align: "center", size: 12, mono: true, color: C.muted});
      const cur_ = si >= 0 ? steps[si] : null;
      // edges
      for (let l = 1; l < sizes.length; l++) for (let j = 0; j < sizes[l]; j++) for (let k = 0; k < sizes[l - 1]; k++) {
        const w = W[l - 1][j][k];
        let alpha = 0.18;
        if (done(l, j)) alpha = 0.55;
        if (cur_ && cur_[0] === l && cur_[1] === j && (cur_[2] === "act" || k < cur_[2])) alpha = 1;
        ctx.globalAlpha = alpha;
        line(ctx, [[X(l - 1) + 22, Yn(l - 1, k)], [X(l) - 22, Yn(l, j)]], w >= 0 ? C.blue : C.red, 0.8 + Math.abs(w) * 3);
        ctx.globalAlpha = 1;
        if (cur_ && cur_[0] === l && cur_[1] === j && k === cur_[2] - 1) {
          // a pulse travelling along the edge being added
          const p = (acc % 22) / 22;
          dot(ctx, lerp(X(l - 1) + 22, X(l) - 22, p), lerp(Yn(l - 1, k), Yn(l, j), p), 5, C.orange);
          text(ctx, `w=${w.toFixed(2)}`, lerp(X(l - 1), X(l), .45), lerp(Yn(l - 1, k), Yn(l, j), .45) - 12, {align: "center", size: 12, mono: true, color: w >= 0 ? C.blue : C.red, weight: 700});
        }
      }
      // nodes
      for (let l = 0; l < sizes.length; l++) for (let j = 0; j < sizes[l]; j++) {
        const known = l === 0 || done(l, j);
        const v = a[l][j];
        const isCur = cur_ && cur_[0] === l && cur_[1] === j;
        const fill = !known ? "#fff" : l === 0 ? C.lblue : v > 0 ? `rgba(21,128,61,${0.15 + Math.min(.6, v / 2)})` : "#f3f4f6";
        dot(ctx, X(l), Yn(l, j), 22, fill, isCur ? C.orange : C.ink2);
        if (isCur) { ctx.lineWidth = 3; ctx.strokeStyle = C.orange; ctx.beginPath(); ctx.arc(X(l), Yn(l, j), 25, 0, 7); ctx.stroke(); }
        if (known) text(ctx, v.toFixed(2), X(l), Yn(l, j), {align: "center", size: 13, mono: true, weight: 700});
        if (l === 0) text(ctx, names[j], X(l) - 32, Yn(l, j), {align: "right", size: 13, color: C.ink2});
      }
      text(ctx, "flow in 1 h", X(3), Yn(3, 0) + 40, {align: "center", size: 13, color: C.ink2});
      text(ctx, "(scaled)", X(3), Yn(3, 0) + 58, {align: "center", size: 12, color: C.muted});
      // panel
      if (!cur_) { panel.innerHTML = `Input (already scaled):<br>x = [${x.map(v => v.toFixed(2)).join(", ")}]<br><br>Press <b>play</b> or <b>next step</b>.`; return; }
      const [l, j, k] = cur_;
      const terms = W[l - 1][j].map((w, i) => `${w.toFixed(2)}·${a[l - 1][i].toFixed(2)}`);
      const nk = k === "act" ? sizes[l - 1] : k;
      const z = zval(a, l, j, nk);
      const lname = l === sizes.length - 1 ? "output" : `layer ${l}, neuron ${j + 1}`;
      let html = `<b>${lname}</b><br>z = ${terms.slice(0, nk).join(" + ")}${nk ? " + " : ""}<span style="color:${C.purple}">b(${B[l - 1][j].toFixed(2)})</span><br>z = <b>${z.toFixed(3)}</b>`;
      if (k === "act") html += l === sizes.length - 1 ? `<br>output layer: no activation<br>ŷ = <b style="color:${C.orange}">${z.toFixed(3)}</b>` : `<br>ReLU(z) = max(0, z) = <b style="color:${C.green}">${Math.max(0, z).toFixed(3)}</b>${z < 0 ? "<br><span style='color:#8a919c'>negative → this neuron is silent</span>" : ""}`;
      panel.innerHTML = html;
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 16. parameter counter
A.params = {
  start() {
    const inp = $("#par-input"), nIn = $("#par-n"), tb = $("#par-table"), verdict = $("#par-verdict");
    const presets = [["our MLP (notebook 2)", "288,64,64,8"], ["linear model", "288,8"], ["DDPG actor", "3,256,256,1"], ["SAC-IL actor", "11,128,128,4"], ["too big", "288,1024,1024,8"]];
    const box = $("#par-presets"); box.innerHTML = "";
    presets.forEach(([n, v]) => { const b = document.createElement("button"); b.textContent = n; b.onclick = () => { inp.value = v; draw(); }; box.appendChild(b); });
    const draw = () => {
      const s = inp.value.split(/[,\s→x-]+/).map(Number).filter(v => v > 0);
      if (s.length < 2) return;
      let tot = 0, rows = `<tr><th>layer</th><th>weights</th><th>biases</th><th>total</th></tr>`;
      for (let i = 1; i < s.length; i++) { const w = s[i - 1] * s[i], b = s[i]; tot += w + b;
        rows += `<tr><td>${s[i - 1]} → ${s[i]}</td><td class="mono">${s[i - 1]}×${s[i]} = ${thousands(w)}</td><td class="mono">${s[i]}</td><td class="mono">${thousands(w + b)}</td></tr>`; }
      rows += `<tr><td><b>all</b></td><td></td><td></td><td class="mono"><b>${thousands(tot)}</b></td></tr>`;
      tb.innerHTML = rows;
      const n = +nIn.value || 1, r = tot / n;
      verdict.className = "note " + (r < 1 ? "green" : r < 20 ? "blue" : "red");
      verdict.innerHTML = `${thousands(tot)} parameters for ${thousands(n)} examples: <b>${r.toFixed(1)} per example</b>. ` + (r < 1 ? "Few knobs: it may underfit, but it can't memorise." : r < 20 ? "Common for neural networks: watch the validation curve, use weight decay and early stopping." : "Far more knobs than examples: it can memorise the training set. Expect overfitting unless you regularise hard or get more data.");
      const ctx = canvas("cv-params"); clear(ctx);
      const maxN = 10, X = i => 60 + i * (580 / (s.length - 1));
      const ys = (n, j) => { const m = Math.min(n, maxN); return 145 + (j - (m - 1) / 2) * 24; };
      for (let i = 1; i < s.length; i++) { const a = Math.min(s[i - 1], maxN), b = Math.min(s[i], maxN);
        for (let j = 0; j < a; j++) for (let k = 0; k < b; k++) line(ctx, [[X(i - 1), ys(s[i - 1], j)], [X(i), ys(s[i], k)]], "rgba(37,99,235,.13)", 1); }
      s.forEach((n, i) => { const m = Math.min(n, maxN);
        for (let j = 0; j < m; j++) dot(ctx, X(i), ys(n, j), 7, i === 0 ? C.lblue : i === s.length - 1 ? C.lorange : "#fff", C.ink2);
        if (n > maxN) text(ctx, "⋮", X(i), ys(n, m - 1) + 20, {align: "center", size: 18, color: C.muted});
        text(ctx, `${n}`, X(i), 268, {align: "center", size: 16, mono: true, weight: 700});
      });
      text(ctx, "units per layer", 340, 286, {align: "center", size: 12, color: C.muted});
    };
    inp.oninput = draw; nIn.oninput = draw;
    inp.value = "288,64,64,8"; draw();
  },
  stop() {}
};

// ═════════════════════════════ 17. causal dilated convolution
A.conv = {
  start() {
    const T = 30, dil = [1, 2, 4], rows = 4;
    this.r = runner(t => {
      const ctx = canvas("cv-conv"); clear(ctx);
      const X = i => 165 + i * 32.5, Y = r => 320 - r * 92;
      const names = ["input (time →)", "conv, dilation 1", "conv, dilation 2", "conv, dilation 4"];
      const out = 14 + Math.floor(t * 1.2) % (T - 14);     // the output neuron moves along time
      // receptive field, layer by layer
      const field = [new Set([out])];
      for (let r = rows - 1; r >= 1; r--) {
        const s = new Set(); field[0].forEach(i => { for (let k = 0; k < 3; k++) { const j = i - k * dil[r - 1]; if (j >= 0) s.add(j); } });
        field.unshift(s);
      }
      // field[0] = input positions … field[3] = {out}
      for (let r = 0; r < rows; r++) {
        text(ctx, names[r], 145, Y(r), {align: "right", size: 13, color: C.ink2});
        for (let i = 0; i < T; i++) {
          const on = field[r].has(i);
          if (r > 0 && on) for (let k = 0; k < 3; k++) { const j = i - k * dil[r - 1]; if (j >= 0) line(ctx, [[X(j), Y(r - 1) - 9], [X(i), Y(r) + 9]], "rgba(234,88,12,.75)", 2); }
        }
        for (let i = 0; i < T; i++) {
          const on = field[r].has(i);
          dot(ctx, X(i), Y(r), 8, on ? (r === rows - 1 ? C.orange : C.lorange) : "#fff", on ? C.orange : "#cbd5e1");
        }
      }
      text(ctx, `this output sees ${field[0].size} past inputs (receptive field 1 + 2·(1+2+4) = 15), and none from the future`, 600, 20, {align: "center", size: 15, weight: 600, color: C.orange});
      line(ctx, [[X(out) + 17, 40], [X(out) + 17, 350]], "rgba(220,38,38,.5)", 2, [6, 5]);
      text(ctx, "now", X(out) + 22, 350, {size: 12, color: C.red});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 18. recurrent unrolling
A.gru = {
  start() {
    const T = 8, H = 6, rnd = seeded(4);
    const xs = Array.from({length: T}, (_, i) => 0.5 + 0.4 * Math.sin(i / 1.3));
    const hs = [Array(H).fill(0)];
    for (let t = 0; t < T; t++) hs.push(hs[t].map((h, j) => Math.tanh(0.7 * h + (j % 2 ? -1 : 1) * xs[t] * (0.4 + 0.15 * j) + 0.2 * gauss(rnd))));
    const ys = hs.slice(1).map(h => 0.5 + 0.35 * h.reduce((a, b) => a + b, 0) / H);
    this.r = runner(t => {
      const ctx = canvas("cv-gru"); clear(ctx);
      const k = Math.floor(t * 0.9) % (T + 3), p = (t * 0.9) % 1;
      const X = i => 110 + i * 132;
      for (let i = 0; i < T; i++) {
        const on = i < k, now = i === k;
        rrect(ctx, X(i) - 40, 150, 80, 64, 10, now ? C.lorange : on ? "#fff" : "#fafafa", now ? C.orange : on ? C.ink2 : "#d1d5db", now ? 3 : 1.5);
        text(ctx, "GRU", X(i), 182, {align: "center", size: 15, weight: 700, color: on || now ? C.ink : "#c0c4cc"});
        // input
        arrow(ctx, X(i), 292, X(i), 220, on || now ? C.blue : "#d1d5db", 2);
        text(ctx, `x${i + 1}`, X(i), 308, {align: "center", size: 14, mono: true, color: on || now ? C.blue : "#c0c4cc"});
        text(ctx, i === 0 ? "air, inflow, …" : "", X(i), 328, {align: "center", size: 11.5, color: C.muted});
        // output
        if (on) { arrow(ctx, X(i), 146, X(i), 92, C.orange, 2); text(ctx, `ŷ${i + 1}`, X(i), 78, {align: "center", size: 14, mono: true, color: C.orange}); }
      }
      // hidden state passing
      for (let i = 0; i < Math.min(k, T - 1); i++) arrow(ctx, X(i) + 42, 182, X(i + 1) - 44, 182, C.purple, 2.5);
      if (k > 0 && k < T) {
        const xx = lerp(X(k - 1) + 42, X(k) - 44, ease(Math.min(1, p * 1.6)));
        const h = hs[k];
        for (let j = 0; j < H; j++) {
          const v = h[j];
          ctx.fillStyle = v > 0 ? `rgba(124,58,237,${.25 + .7 * Math.abs(v)})` : `rgba(220,38,38,${.25 + .7 * Math.abs(v)})`;
          ctx.fillRect(xx - 7, 182 - H * 4.5 + j * 9, 14, 8);
        }
        text(ctx, "memory h", xx, 140, {align: "center", size: 12.5, color: C.purple, weight: 600});
      }
      text(ctx, "inputs enter from below, predictions leave from the top", 600, 352, {align: "center", size: 12.5, color: C.muted});
      text(ctx, "same cell, same weights, at every step · the memory h is handed on", 600, 30, {align: "center", size: 15, weight: 600, color: C.purple});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 19. losses
A.loss = {
  start() {
    const box = $("#loss-btns"); box.innerHTML = "";
    const L = [["MSE", e => e * e, C.blue], ["MAE", e => Math.abs(e), C.green], ["pinball τ=0.9", e => Math.max(0.9 * e, -0.1 * e), C.orange], ["pinball τ=0.1", e => Math.max(0.1 * e, -0.9 * e), C.purple]];
    const on = [true, true, false, false];
    L.forEach(([n], i) => { const b = document.createElement("button"); b.textContent = n; b.classList.toggle("on", on[i]); b.onclick = () => { on[i] = !on[i]; b.classList.toggle("on", on[i]); }; box.appendChild(b); });
    this.r = runner(t => {
      const ctx = canvas("cv-loss"); clear(ctx);
      const ax = axes(60, 20, 550, 330, [-2, 2], [0, 4]);
      frame(ctx, ax, {xlabel: "error e = true − predicted   (e > 0: we predicted too low)", ylabel: "loss"});
      line(ctx, [[ax.x(0), ax.y0], [ax.x(0), ax.y0 + ax.h]], C.line, 1);
      const e = 1.8 * Math.sin(t * 0.7);
      L.forEach(([n, f, col], i) => {
        if (!on[i]) return;
        const pts = []; for (let v = -2; v <= 2.001; v += 0.02) pts.push([ax.x(v), ax.y(Math.min(4, f(v)))]);
        line(ctx, pts, col, 3);
        dot(ctx, ax.x(e), ax.y(Math.min(4, f(e))), 6, col, "#fff");
        text(ctx, `${n}: ${f(e).toFixed(2)}`, 76, 40 + i * 22, {size: 14, mono: true, color: col, weight: 600});
      });
      line(ctx, [[ax.x(e), ax.y0], [ax.x(e), ax.y0 + ax.h]], "rgba(29,36,51,.25)", 1, [4, 4]);
      text(ctx, `e = ${e.toFixed(2)}`, ax.x(e), ax.y0 + ax.h - 10, {align: "center", size: 12, mono: true});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 20. gradient descent on a 2-D bowl
A.descent = {
  start(el) {
    let lr = 0.15, path = [], running = true, k = 0;
    const f = (a, b) => 0.5 * (a * a) + 2.5 * (b * b);          // elongated bowl
    const g = (a, b) => [a, 5 * b];
    const reset = () => { path = [[-2.6, 1.15]]; k = 0; };
    reset();
    $$("button[data-lr]", el).forEach(b => b.onclick = () => { $$("button[data-lr]", el).forEach(x => x.classList.remove("on")); b.classList.add("on"); lr = +b.dataset.lr; reset(); running = true; });
    $("#gd-run").onclick = () => { reset(); running = true; };
    const panel = $("#gd-panel");
    let acc = 0;
    this.r = runner(t => {
      acc++;
      if (running && acc % 10 === 0 && path.length < 60) {
        const [a, b] = path[path.length - 1], [ga, gb] = g(a, b);
        const n = [a - lr * ga, b - lr * gb];
        path.push(n); k++;
        if (Math.abs(n[0]) > 6 || Math.abs(n[1]) > 6 || f(...n) < 1e-4) running = false;
      }
      const ctx = canvas("cv-descent"); clear(ctx);
      const ax = axes(30, 20, 600, 380, [-3.2, 3.2], [-2, 2]);
      // contours
      for (let c = 0.05; c < 12; c *= 1.6) {
        ctx.beginPath();
        for (let th = 0; th <= 2 * Math.PI + .01; th += .04) { const a = Math.sqrt(2 * c) * Math.cos(th), b = Math.sqrt(c / 2.5) * Math.sin(th); th ? ctx.lineTo(ax.x(a), ax.y(b)) : ctx.moveTo(ax.x(a), ax.y(b)); }
        ctx.strokeStyle = `rgba(37,99,235,${0.12 + 0.5 / (1 + c)})`; ctx.lineWidth = 1.3; ctx.stroke();
      }
      dot(ctx, ax.x(0), ax.y(0), 5, C.green);
      text(ctx, "minimum", ax.x(0) + 8, ax.y(0) + 14, {size: 12, color: C.green});
      ctx.save(); ctx.beginPath(); ctx.rect(ax.x0, ax.y0, ax.w, ax.h); ctx.clip();
      line(ctx, path.map(([a, b]) => [ax.x(a), ax.y(b)]), C.orange, 2);
      path.forEach(([a, b], i) => dot(ctx, ax.x(a), ax.y(b), i === path.length - 1 ? 8 : 3.5, i === path.length - 1 ? C.orange : C.lorange, C.orange));
      ctx.restore();
      text(ctx, "weight 1", ax.x0 + ax.w - 4, ax.y0 + ax.h - 10, {align: "right", size: 12, color: C.muted});
      text(ctx, "weight 2", ax.x0 + 6, ax.y0 + 12, {size: 12, color: C.muted});
      const [a, b] = path[path.length - 1], L = f(a, b);
      panel.innerHTML = `lr = ${lr} · step ${k} · loss = ${L > 1e4 ? "∞ (diverged)" : L.toFixed(4)}` + (Math.abs(a) > 6 || Math.abs(b) > 6 ? " <b style='color:#dc2626'>diverged</b>" : "");
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 21. the training loop + curves
A.loop = {
  start() {
    const stages = [["1 · batch of data", C.ink2, "x, y"], ["2 · forward", C.blue, "ŷ = model(x)"], ["3 · loss", C.red, "loss(ŷ, y)"], ["4 · backward", C.purple, "∂loss/∂w"], ["5 · update", C.green, "w ← w − lr·g"]];
    const E = 40;
    const tr = e => 0.08 + 0.9 * Math.exp(-e / 7);
    const va = e => 0.15 + 0.85 * Math.exp(-e / 7) + 0.0009 * Math.max(0, e - 14) ** 2;
    let best = 0;
    for (let e = 0; e < E; e++) if (va(e) < va(best)) best = e;
    this.r = runner(t => {
      const ctx = canvas("cv-loop"); clear(ctx);
      const cx = 260, cy = 215, R = 150;
      const pos = i => { const a = -Math.PI / 2 + i * 2 * Math.PI / stages.length; return [cx + R * Math.cos(a), cy + R * Math.sin(a)]; };
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.strokeStyle = C.line; ctx.lineWidth = 6; ctx.stroke();
      const cyc = t * 0.55, ph = cyc % 1, active = Math.floor(ph * stages.length);
      // the pulse
      const a = -Math.PI / 2 + ph * 2 * Math.PI;
      dot(ctx, cx + R * Math.cos(a), cy + R * Math.sin(a), 9, C.orange);
      stages.forEach(([n, col, sub], i) => {
        const [x, y] = pos(i);
        rrect(ctx, x - 80, y - 28, 160, 56, 10, i === active ? col : "#fff", col, 2.5);
        text(ctx, n, x, y - 9, {align: "center", size: 14, weight: 700, color: i === active ? "#fff" : col});
        text(ctx, sub, x, y + 12, {align: "center", size: 13, mono: true, color: i === active ? "#fff" : C.ink2});
      });
      const steps = Math.floor(cyc), epoch = Math.floor(t * 1.4);
      text(ctx, `step ${steps}`, cx, cy - 12, {align: "center", size: 22, mono: true, weight: 700});
      text(ctx, `epoch ${Math.min(epoch % (E + 10), E - 1)}`, cx, cy + 18, {align: "center", size: 15, mono: true, color: C.ink2});
      text(ctx, "one epoch = every batch once", cx, cy + 40, {align: "center", size: 11.5, color: C.muted});
      // curves
      const c2 = canvas("cv-curves"); clear(c2);
      const ax = axes(50, 12, 490, 140, [0, E], [0, 1.05]);
      frame(c2, ax, {xlabel: "epoch"});
      const shown = Math.min(epoch % (E + 10), E - 1);
      line(c2, Array.from({length: shown + 1}, (_, e) => [ax.x(e), ax.y(tr(e))]), C.blue, 2.5);
      line(c2, Array.from({length: shown + 1}, (_, e) => [ax.x(e), ax.y(va(e))]), C.orange, 2.5);
      text(c2, "training loss", ax.x(E) - 4, ax.y(tr(E)) - 10, {align: "right", size: 12, color: C.blue, weight: 600});
      text(c2, "validation loss", ax.x(E) - 4, ax.y(va(E - 1)) - 12, {align: "right", size: 12, color: C.orange, weight: 600});
      if (shown >= best) {
        dot(c2, ax.x(best), ax.y(va(best)), 6, C.green, "#fff");
        text(c2, "best epoch: save this one", ax.x(best) + 10, ax.y(va(best)) + 18, {size: 12, color: C.green, weight: 600});
      }
      if (shown > best + 6) text(c2, "overfitting: train keeps falling, validation rises", ax.x(best) + 40, ax.y0 + 12, {size: 12, color: C.red});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 22. baselines: bars from the README
A.baselines = {
  start() {
    const rows = [["MAACC as printed", [0, 0, 0], C.muted], ["Repaired MAACC", [-85, -52, -67], C.purple], ["PID ×3 + ILC", [-85, -82, -72], C.teal],
      ["Ours: + SAC-IL + gate", [-87, -82, -74], C.blue], ["Tuned PID + ILC", [-90, -89, -80], C.green]];
    this.r = runner(t => {
      const ctx = canvas("cv-base"); clear(ctx);
      const p = ease(clamp(t / 1.6, 0, 1));
      const ws = ["dry", "rain", "storm"];
      const ax = axes(170, 30, 360, 300, [-100, 0], [0, 1]);
      text(ctx, "−100 %", ax.x(-100), 18, {align: "center", size: 12, color: C.muted});
      text(ctx, "0 %", ax.x(0), 18, {align: "center", size: 12, color: C.muted});
      line(ctx, [[ax.x(0), 26], [ax.x(0), 340]], C.ink2, 1);
      rows.forEach(([n, v, col], i) => {
        const y0 = 40 + i * 62;
        text(ctx, n, 160, y0 + 18, {align: "right", size: 13.5, weight: i >= 3 ? 700 : 400});
        v.forEach((val, j) => {
          const w = ax.x(0) - ax.x(val * p);
          rrect(ctx, ax.x(0) - w, y0 + j * 13, Math.max(w, 1), 11, 2, col);
          if (i === 0) text(ctx, ws[j], ax.x(0) + 6, y0 + j * 13 + 6, {size: 11, color: C.muted});
          else text(ctx, `${Math.round(val * p)} %`, ax.x(0) - w - 4, y0 + j * 13 + 6, {align: "right", size: 11, mono: true, color: col});
        });
      });
      text(ctx, "bars per row: dry · rain · storm", 350, 362, {align: "center", size: 12, color: C.ink2});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ 24. the RL loop, on a one-tank oxygen model
A.rl = {
  start() {
    // the same toy environment as notebook 3: S' = Seq + (S − Seq) e^(−KLa dt), Seq = 8 − OUR/KLa
    let S = 1.4, kla = 100, k = 0, ret = 0, e1 = 0;
    const dt = 5 / 1440;
    const our = k => 700 * (1 + 0.35 * Math.sin(2 * Math.PI * (k * dt - 0.3)));
    const hist = [];
    this.r = runner(t => {
      const ph = (t * 0.6) % 1, newStep = Math.floor(t * 0.6) !== k;
      if (newStep) {
        k = Math.floor(t * 0.6);
        const e = 2 - S;                                   // a simple incremental controller as the "agent"
        const da = clamp(40 * (e - e1) + 6 * e, -12, 12); e1 = e;
        kla = clamp(kla + da, 10, 240);
        const Seq = 8 - our(k) / kla; S = Seq + (S - Seq) * Math.exp(-kla * dt);
        const r = -Math.abs(S - 2) - 0.1 * kla / 240; ret += r;
        hist.push(S); if (hist.length > 60) hist.shift();
        this.last = {e, da, r};
      }
      const L = this.last || {e: 0, da: 0, r: 0};
      const ctx = canvas("cv-rl"); clear(ctx);
      rrect(ctx, 190, 30, 220, 96, 14, C.lblue, C.blue, 2.5);
      text(ctx, "Agent", 300, 62, {align: "center", size: 20, weight: 700, color: C.blue});
      text(ctx, "policy: state → action", 300, 92, {align: "center", size: 13, color: C.ink2});
      rrect(ctx, 190, 290, 220, 110, 14, "#e0f2fe", C.teal, 2.5);
      text(ctx, "Environment", 300, 318, {align: "center", size: 20, weight: 700, color: C.teal});
      text(ctx, "tank 5: oxygen dynamics", 300, 346, {align: "center", size: 13, color: C.ink2});
      text(ctx, `S_O = ${S.toFixed(2)} mg/L`, 300, 374, {align: "center", size: 15, mono: true, weight: 700, color: Math.abs(S - 2) < .2 ? C.green : C.red});
      // arrows
      arrow(ctx, 420, 80, 500, 80, C.orange, 3); line(ctx, [[500, 80], [500, 345]], C.orange, 3); arrow(ctx, 500, 345, 420, 345, C.orange, 3);
      text(ctx, "action a", 512, 190, {size: 15, weight: 700, color: C.orange});
      text(ctx, `ΔKLa = ${L.da.toFixed(1)}`, 512, 212, {size: 13, mono: true, color: C.orange});
      line(ctx, [[180, 345], [110, 345], [110, 80]], C.purple, 3); arrow(ctx, 110, 80, 180, 80, C.purple, 3);
      text(ctx, "state s", 8, 175, {size: 15, weight: 700, color: C.purple});
      text(ctx, `error ${L.e.toFixed(2)}`, 8, 197, {size: 13, mono: true, color: C.purple});
      text(ctx, "reward r", 8, 235, {size: 15, weight: 700, color: C.red});
      text(ctx, `${L.r.toFixed(3)}`, 8, 257, {size: 13, mono: true, color: C.red});
      // travelling tokens
      const tok = (pts, col) => { const segs = []; let tot = 0; for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); segs.push(l); tot += l; }
        let s = ph * tot; for (let i = 0; i < segs.length; i++) { if (s <= segs[i]) { dot(ctx, lerp(pts[i][0], pts[i + 1][0], s / segs[i]), lerp(pts[i][1], pts[i + 1][1], s / segs[i]), 7, col); return; } s -= segs[i]; } };
      tok([[420, 80], [500, 80], [500, 345], [420, 345]], C.orange);
      tok([[180, 345], [110, 345], [110, 80], [180, 80]], C.purple);
      text(ctx, `step ${k} · return so far ${ret.toFixed(1)}`, 300, 160, {align: "center", size: 14, mono: true});
      text(ctx, "every 5 min: observe → act → plant moves → reward", 300, 186, {align: "center", size: 12.5, color: C.muted});
      // mini trace
      const tx = axes(190, 412, 400, 22, [0, 60], [1, 3]);
      line(ctx, [[tx.x(0), tx.y(2)], [tx.x(60), tx.y(2)]], C.green, 1, [3, 3]);
      line(ctx, hist.map((v, i) => [tx.x(i), tx.y(clamp(v, 1, 3))]), C.teal, 1.8);
      text(ctx, "oxygen", 184, 422, {align: "right", size: 11, color: C.muted});
    });
  },
  stop() { this.r && this.r.stop(); }
};

// ═════════════════════════════ shared: Python syntax highlighting (no library)
const PY_KW = new Set("def class return for in if elif else while with as import from not and or lambda None True False try except finally yield is break continue pass assert raise global nonlocal del async await".split(" "));
const PY_BUILTIN = new Set("print len range zip enumerate float int str list dict tuple set min max sum abs isinstance super np torch nn F pd plt self".split(" "));
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
// returns one HTML string per source line; tokens that span lines (docstrings) are split per line
function highlight(code) {
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbuf]?"(?:\\.|[^"\\\n])*"|[rbuf]?'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d*)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)|(@\w+)/g;
  const lines = [""]; let last = 0, m;
  const push = (txt, cls) => txt.split("\n").forEach((piece, i) => {
    if (i) lines.push("");
    if (piece) lines[lines.length - 1] += cls ? `<span class="${cls}">${esc(piece)}</span>` : esc(piece);
  });
  while ((m = re.exec(code))) {
    push(code.slice(last, m.index), null);
    if (m[1]) push(m[1], "tc"); else if (m[2]) push(m[2], "ts"); else if (m[3]) push(m[3], "tn");
    else if (m[4]) push(m[4], PY_KW.has(m[4]) ? "tk" : PY_BUILTIN.has(m[4]) ? "tb" : (code[re.lastIndex] === "(" ? "tf" : null));
    else push(m[5], "tk");
    last = re.lastIndex;
  }
  push(code.slice(last), null);
  return lines;
}

// engine hooks: an animation may consume → and ← (players and tours step through their parts)
const _next = next, _prev = prev;
next = function () { if (live && live.next && live.next()) return; _next(); };
prev = function () { if (live && live.prev && live.prev()) return; _prev(); };

// ═════════════════════════════ notebook player: real cells, real outputs, a concept panel
const PSTATE = {};
A.player = {
  start(el) {
    const key = el.dataset.key, P = PLAYERS[key];
    const st = PSTATE[key] || (PSTATE[key] = {ran: 0});
    this.st = st; this.P = P;
    const root = $(".player-root", el);
    root.innerHTML = `
      <div class="nb">
        <div class="nb-bar"><span class="mono tiny">${P.notebook}</span><span style="flex:1"></span>
          <button class="nb-all">▶▶ run all</button><button class="nb-reset">↺ restart kernel</button></div>
        <div class="nb-cells"></div>
      </div>
      <div class="nb-side card">
        <div class="tiny nb-step"></div>
        <h3 class="nb-title"></h3>
        <p class="nb-note"></p>
        <ol class="nb-list"></ol>
        <p class="tiny" style="margin-top:auto">Outputs are from a real run of the notebook. Press → to run the next cell.</p>
      </div>`;
    const cells = $(".nb-cells", root);
    P.cells.forEach((c, i) => {
      const div = document.createElement("div");
      div.className = "nb-cell";
      div.innerHTML = `
        <div class="nb-in"><button class="nb-run" title="run this cell">▶</button><span class="nb-count mono">In [&nbsp;]:</span></div>
        <pre class="nb-code">${highlight(c.code).join("\n")}</pre>
        <div class="nb-out" style="display:none">
          ${c.out.text ? `<pre class="nb-text">${esc(c.out.text)}</pre>` : ""}
          ${c.out.img.map(s => `<img src="${s}" alt="figure">`).join("")}
          ${!c.out.text && !c.out.img.length ? `<div class="tiny">(no output: this cell only defines things)</div>` : ""}
        </div>`;
      $(".nb-run", div).onclick = () => this.runTo(i + 1);
      cells.appendChild(div);
      const li = document.createElement("li");
      li.textContent = c.title; li.onclick = () => this.runTo(i + 1);
      $(".nb-list", root).appendChild(li);
    });
    $(".nb-all", root).onclick = () => this.runTo(P.cells.length);
    $(".nb-reset", root).onclick = () => { st.ran = 0; this.paint(true); };
    this.root = root;
    this.paint(true);
  },
  runTo(n) {
    const from = this.st.ran;
    if (n <= from) { this.st.ran = n; this.paint(); return; }
    // cells run one after the other, with a short busy state, like a kernel
    const step = k => {
      if (k > n) return;
      const cell = $$(".nb-cell", this.root)[k - 1];
      $(".nb-count", cell).textContent = "In [*]:"; cell.classList.add("busy");
      setTimeout(() => { this.st.ran = k; cell.classList.remove("busy"); this.paint(); step(k + 1); }, 450);
    };
    step(from + 1);
  },
  paint(instant) {
    const {P, st, root} = this;
    $$(".nb-cell", root).forEach((cell, i) => {
      const done = i < st.ran;
      $(".nb-count", cell).innerHTML = done ? `In [${P.cells[i].n}]:` : "In [&nbsp;]:";
      const out = $(".nb-out", cell);
      if (done && out.style.display === "none") { out.style.display = ""; out.classList.remove("fresh"); void out.offsetWidth; out.classList.add("fresh"); }
      if (!done) out.style.display = "none";
      cell.classList.toggle("current", i === st.ran - 1);
    });
    $$(".nb-list li", root).forEach((li, i) => { li.classList.toggle("done", i < st.ran); li.classList.toggle("current", i === st.ran - 1); });
    const k = Math.max(0, st.ran - 1), c = P.cells[k];
    $(".nb-step", root).textContent = st.ran ? `cell ${st.ran} of ${P.cells.length}` : `${P.cells.length} cells · nothing run yet`;
    $(".nb-title", root).textContent = st.ran ? c.title : "Press ▶ or →";
    $(".nb-note", root).textContent = st.ran ? c.note : "Each cell runs the real notebook code. The panel explains the idea behind it.";
    const box = $(".nb-cells", root), cur = $$(".nb-cell", root)[k];
    if (st.ran && cur) box.scrollTo({top: cur.offsetTop - box.offsetTop - 8, behavior: instant ? "auto" : "smooth"});
    else box.scrollTop = 0;
  },
  next() { if (this.st.ran < this.P.cells.length) { this.runTo(this.st.ran + 1); return true; } return false; },
  prev() { if (this.st.ran > 0) { this.st.ran--; this.paint(); return true; } return false; },
  stop() {}
};

// ═════════════════════════════ code tour: the project's own code, block by block
const TSTATE = {};
A.tour = {
  start(el) {
    const key = el.dataset.key, T = TOURS[key];
    const st = TSTATE[key] || (TSTATE[key] = {tab: 0, mark: 0});
    Object.assign(this, {T, st, root: $(".tour-root", el)});
    this.root.innerHTML = `
      <div class="tour-tabs">${T.map((t, i) => `<button data-t="${i}">${t.title}</button>`).join("")}</div>
      <div class="tour-body">
        <div class="tour-code"><div class="tour-file mono tiny"></div><div class="tour-lines"></div></div>
        <div class="tour-side card">
          <div class="tiny tour-step"></div><h3 class="tour-title"></h3><p class="tour-text"></p>
          <ol class="tour-list"></ol>
          <div class="btns" style="margin-top:auto"><button class="tour-prev">← back</button><button class="tour-next">next →</button></div>
        </div>
      </div>`;
    $$(".tour-tabs button", this.root).forEach(b => b.onclick = () => { st.tab = +b.dataset.t; st.mark = 0; this.paint(true); });
    $(".tour-prev", this.root).onclick = () => this.prev();
    $(".tour-next", this.root).onclick = () => this.next();
    if (T.length === 1) $(".tour-tabs", this.root).style.display = "none";
    this.paint(true);
  },
  paint(rebuild) {
    const {T, st, root} = this, t = T[st.tab], mk = t.marks[st.mark];
    $$(".tour-tabs button", root).forEach((b, i) => b.classList.toggle("on", i === st.tab));
    if (rebuild || this._tab !== st.tab) {
      this._tab = st.tab;
      $(".tour-file", root).textContent = `${t.file} · ${t.func} · line ${t.line}`;
      const hl = highlight(t.code.join("\n"));
      $(".tour-lines", root).innerHTML = hl.map((h, i) => `<div class="tl" data-i="${i}"><span class="ln">${t.line + i}</span><span class="lc">${h || " "}</span></div>`).join("");
      $(".tour-list", root).innerHTML = t.marks.map((m, i) => `<li data-m="${i}">${m.title}</li>`).join("");
      $$(".tour-list li", root).forEach(li => li.onclick = () => { st.mark = +li.dataset.m; this.paint(); });
      $$(".tl", root).forEach(row => row.onclick = () => {
        const i = +row.dataset.i, m = t.marks.findIndex(m => i >= m.from && i <= m.to);
        if (m >= 0) { st.mark = m; this.paint(); }
      });
    }
    $$(".tl", root).forEach(row => { const i = +row.dataset.i; row.classList.toggle("on", i >= mk.from && i <= mk.to);
      row.classList.toggle("dim", !(i >= mk.from && i <= mk.to)); row.classList.toggle("marked", t.marks.some(m => i >= m.from && i <= m.to)); });
    $$(".tour-list li", root).forEach((li, i) => li.classList.toggle("current", i === st.mark));
    $(".tour-step", root).textContent = `${st.mark + 1} / ${t.marks.length}` + (T.length > 1 ? ` · file ${st.tab + 1} of ${T.length}` : "");
    $(".tour-title", root).textContent = mk.title;
    $(".tour-text", root).textContent = mk.text;
    const box = $(".tour-lines", root), row = $(`.tl[data-i="${mk.from}"]`, root);
    if (row) box.scrollTo({top: Math.max(0, row.offsetTop - box.offsetTop - 60), behavior: rebuild ? "auto" : "smooth"});
  },
  next() {
    const {T, st} = this;
    if (st.mark < T[st.tab].marks.length - 1) st.mark++;
    else if (st.tab < T.length - 1) { st.tab++; st.mark = 0; }
    else return false;
    this.paint(); return true;
  },
  prev() {
    const {T, st} = this;
    if (st.mark > 0) st.mark--;
    else if (st.tab > 0) { st.tab--; st.mark = T[st.tab].marks.length - 1; }
    else return false;
    this.paint(); return true;
  },
  stop() {}
};

// ═════════════════════════════ pipeline map: every experiment of the project
A.pipeline = {
  start(el) {
    const root = $(".pipe-root", el);
    const cols = ["Plant + data", "Reproduce the 3 papers", "Build ours + baselines", "Test", "Report"];
    root.innerHTML = `<div class="pipe-grid">${cols.map((c, j) => `<div class="pipe-col"><div class="tiny pipe-head">${c}</div>${
      PIPELINE.filter(p => p.col === j).map(p => `<button class="pipe-node" data-id="${p.id}">${p.title}<br><span class="mono">${p.file.split("/").pop()}</span></button>`).join("")}</div>`).join("")}</div>
      <canvas id="cv-pipe" width="1160" height="300"></canvas>
      <div class="card pipe-panel"></div>`;
    this.root = root; this.i = this.i ?? -1;
    $$(".pipe-node", root).forEach(b => b.onclick = () => { this.i = PIPELINE.findIndex(p => p.id === b.dataset.id); this.paint(); });
    this.paint();
  },
  paint() {
    const root = this.root, p = PIPELINE[this.i];
    $$(".pipe-node", root).forEach(b => { const k = PIPELINE.findIndex(q => q.id === b.dataset.id);
      b.classList.toggle("on", k === this.i); b.classList.toggle("seen", k < this.i); });
    $(".pipe-panel", root).innerHTML = p ? `<div class="cols" style="gap:20px"><div class="col"><h3>${p.title}</h3><p class="small" style="margin:0">${p.what}</p></div>
      <div class="col" style="flex:1.4"><div class="kv" style="grid-template-columns:70px 1fr;font-size:14px"><div>file</div><div class="mono">${p.file}</div><div>run</div><div class="mono">${esc(p.cmd)}</div><div>writes</div><div class="mono">${p.out}</div></div></div></div>`
      : `<p class="small" style="margin:0">Click a box, or press →, to walk through the project in the order it was built. Every box is one script in <code>experiments/</code>.</p>`;
    // arrows between columns
    const ctx = canvas("cv-pipe"); ctx.clearRect(0, 0, ctx.w, ctx.h);
    const box = $(".pipe-grid", root).getBoundingClientRect(), s = box.width / 1160;
    const rect = id => { const r = $(`.pipe-node[data-id="${id}"]`, root).getBoundingClientRect(); return {x0: (r.left - box.left) / s, x1: (r.right - box.left) / s, y: (r.top + r.bottom) / 2 / s - box.top / s}; };
    const links = [["data", "e20"], ["data", "e06"], ["data", "e23"], ["e20", "e32"], ["e20", "e39"], ["e32", "e28"], ["e20", "e38"], ["e28", "e33"], ["e38", "e33"], ["e39", "e33"], ["e33", "e40"], ["e33", "docs"], ["e40", "docs"], ["e06", "docs"], ["e23", "docs"]];
    links.forEach(([a, b]) => { const A_ = rect(a), B_ = rect(b);
      const on = p && (p.id === a || p.id === b);
      const x1 = A_.x1 + 2, x2 = B_.x0 - 4, y1 = A_.y, y2 = B_.y;
      if (x2 <= x1) { line(ctx, [[A_.x0 + 30, y1 + 16], [B_.x0 + 30, y2 - 16]], on ? C.orange : "#cbd5e1", on ? 2.5 : 1.5); return; }
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2);
      ctx.strokeStyle = on ? C.orange : "#cbd5e1"; ctx.lineWidth = on ? 2.5 : 1.5; ctx.stroke();
      arrow(ctx, x2 - 1, y2, x2, y2, on ? C.orange : "#cbd5e1", 1, 7); });
  },
  next() { if (this.i < PIPELINE.length - 1) { this.i++; this.paint(); return true; } return false; },
  prev() { if (this.i >= 0) { this.i--; this.paint(); return this.i >= -1; } return false; },
  stop() {}
};


// ═════════════════════════════ glossary: every word of the workshop, in plain words
const GLOSSARY = [
  ["data", "Numbers we have recorded.", "A diary of the plant: every 15 minutes, how much water came in and how dirty it was.", "notebook 1"],
  ["dataset", "A table of numbers a model learns from.", "A spreadsheet: one row per moment, one column per measurement.", "1 345 rows × 15 columns per weather file"],
  ["cleaning", "Finding and fixing wrong or missing values.", "Washing the vegetables before cooking.", "slides 10–13, notebook 1 part B"],
  ["input (x)", "What the model gets to see.", "The front of a flash card.", "the last 24 hours of flow, ammonium and COD"],
  ["target (y)", "What the model must guess.", "The back of a flash card.", "the flow over the next 2 hours"],
  ["model", "A function with adjustable numbers inside.", "A recipe whose amounts you can change.", "linear model, MLP, CNN, GRU, actor, critic"],
  ["parameter / weight", "One adjustable number inside a model.", "One knob on the mixing desk.", "23 176 in our MLP; 67 073 in paper 2's actor"],
  ["training", "Turning the knobs until the guesses get good.", "Taste, adjust, taste again.", "slides 27–30, notebook 2"],
  ["loss", "One number for how wrong the model is.", "Points lost in a game; training wants as few as possible.", "MSE, MAE, pinball, cross-entropy"],
  ["gradient", "Which way to turn every knob to lower the loss.", "Feeling which way is downhill with your feet, in the fog.", "computed by loss.backward()"],
  ["learning rate", "How big a step we take each time.", "Baby steps or giant leaps: too big and you overshoot.", "slide 28: try 0.02, 0.15, 0.5"],
  ["batch / epoch", "A handful of examples / one pass through all of them.", "A few flash cards at a time / the whole pack once.", "batch 128, up to 150 epochs"],
  ["neural network", "Layers of simple units, one after the other.", "An assembly line: each station does a small job.", "slide 22: watch numbers flow layer by layer"],
  ["ReLU", "Keep positive numbers, turn negatives into 0.", "A one-way valve.", "after every hidden layer"],
  ["train / validation / test", "Learn on / check while learning on / final exam, used once.", "Homework / practice test / the real exam.", "dry + rain week 2 / storm 1 / storm 2"],
  ["overfitting", "Memorising examples instead of learning the pattern.", "Learning the answers of last year's exam by heart.", "notebook 2 section 7"],
  ["baseline", "A simple rule with no learning.", "\"Tomorrow will be like today.\"", "persistence, yesterday, linear model, PID"],
  ["reinforcement learning", "Learning from rewards instead of right answers.", "Training a dog with treats.", "paper 2's DDPG, our SAC-IL, notebook 3"],
  ["agent / environment", "The one who decides / the world it acts in.", "The player / the game.", "the controller / the plant"],
  ["reward", "A score after each action.", "The treat. The agent does whatever earns treats.", "−|oxygen − 2| − air cost"],
  ["policy", "The agent's rule: in this situation, do this.", "A cook's habits.", "the actor network"],
  ["PID", "A classic controller: too low → more, too high → less.", "A thermostat.", "the expert of all three papers"],
];
A.glossary = {
  start(el) {
    const grid = $(".gl-grid", el), card = $(".gl-card", el);
    grid.innerHTML = GLOSSARY.map((g, i) => `<button data-i="${i}">${g[0]}</button>`).join("");
    this.pick = i => {
      this.i = i;
      $$("button", grid).forEach((b, j) => b.classList.toggle("on", j === i));
      const g = GLOSSARY[i];
      card.innerHTML = `<div class="tiny">${i + 1} of ${GLOSSARY.length}</div><h3 style="font-size:30px;color:${C.blue};margin:4px 0 10px">${g[0]}</h3>
        <p class="big" style="margin:0 0 14px">${g[1]}</p>
        <div class="note" style="font-size:18px">🧒 <b>Like:</b> ${g[2]}</div>
        <p class="small" style="margin-top:14px"><b>In this project:</b> ${g[3]}</p>`;
    };
    $$("button", grid).forEach(b => b.onclick = () => this.pick(+b.dataset.i));
    this.pick(this.i ?? 0);
  },
  next() { if (this.i < GLOSSARY.length - 1) { this.pick(this.i + 1); return true; } return false; },
  prev() { if (this.i > 0) { this.pick(this.i - 1); return true; } return false; },
  stop() {}
};


// ═════════════════════════════ code map: the same steps, the same colour, in every file
const CM_COLORS = ["#7c3aed", "#2563eb", "#ea580c", "#dc2626", "#15803d"];
const CM_TEXT = {
  lesson1: ["The knobs: the numbers training may change. In the script two (w, b); in notebook 2, 23 176; in the forecaster, 171 211. The optimiser is handed exactly these.",
            "Guess: run the model on a batch of inputs. The inputs never change.",
            "How wrong: one number, the loss. MSE in the script and the notebook, the quantile (pinball) loss in the forecaster, MSE against the Bellman target in the RL critic.",
            "Which way: the gradient. By hand in the script; loss.backward() everywhere else.",
            "Turn the knobs a little: w = w − lr · gradient; opt.step() does it for every knob at once."],
  lesson2: ["Download: from a pinned commit, only if the file is not already there.",
            "Read: the first row is names, not numbers. Read it as the header, name the columns, make them numbers.",
            "Check: time step, missing values, duplicates, impossible values. In the project these checks are tests (tests/test_official.py).",
            "Fix: repair broken readings (notebook), and the physics fix of the project: rain dilutes S_I (plant.py)."],
};
const CMSTATE = {};
A.codemap = {
  start(el) {
    const key = el.dataset.key, M = CODEMAPS[key];
    const st = CMSTATE[key] || (CMSTATE[key] = {s: -1});
    Object.assign(this, {M, st, key, root: $(".cm-root", el)});
    this.root.innerHTML = `<div class="cm-steps">${M.steps.map((t, i) => `<button data-s="${i}" style="--c:${CM_COLORS[i]}">${t}</button>`).join("")}<button data-s="-1" class="cm-all">all at once</button></div>
      <div class="cm-grid">${M.panels.map((p, j) => `<div class="cm-panel"><div class="cm-head"><b>${p.title}</b><span class="mono">${p.where}</span></div>
        <div class="cm-code">${highlight(p.code.join("\n")).map((h, i) => `<div class="cl" data-i="${i}"><span class="ln">${i + 1}</span><span>${h || " "}</span></div>`).join("")}</div></div>`).join("")}</div>
      <div class="cm-caption"></div>`;
    $$(".cm-steps button", this.root).forEach(b => b.onclick = () => { st.s = +b.dataset.s; this.paint(); });
    this.paint(true);
  },
  paint(instant) {
    const {M, st, root, key} = this;
    $$(".cm-steps button", root).forEach(b => b.classList.toggle("on", +b.dataset.s === st.s));
    $$(".cm-panel", root).forEach((panel, j) => {
      const mk = M.panels[j].marks;
      const owner = i => { for (const [s, [a, b]] of Object.entries(mk)) if (i >= a && i <= b) return +s; return -1; };
      $$(".cl", panel).forEach(row => {
        const o = owner(+row.dataset.i);
        const show = o >= 0 && (st.s === -1 || o === st.s);
        row.style.background = show ? CM_COLORS[o] + (st.s === -1 ? "22" : "33") : "";
        row.style.boxShadow = show ? `inset 4px 0 0 ${CM_COLORS[o]}` : "";
        row.style.opacity = st.s === -1 || o === st.s ? 1 : 0.35;
      });
      panel.classList.toggle("absent", st.s >= 0 && !(st.s in mk));
      const box = $(".cm-code", panel), target = st.s >= 0 && mk[st.s] ? mk[st.s][0] : Math.min(...Object.values(mk).map(v => v[0]));
      const row = $(`.cl[data-i="${target}"]`, panel);
      if (row) box.scrollTo({top: Math.max(0, row.offsetTop - box.offsetTop - 40), behavior: instant ? "auto" : "smooth"});
    });
    const cap = $(".cm-caption", root);
    cap.innerHTML = st.s === -1 ? `<b>Same colour = same step.</b> Press → to go through the steps one by one, or click a step. Every panel is real code from the repository.`
      : `<b style="color:${CM_COLORS[st.s]}">${M.steps[st.s]}.</b> ${CM_TEXT[key][st.s]}` + ($$(".cm-panel.absent", root).length ? ` <span class="tiny">(Dimmed panels don't have this step.)</span>` : "");
  },
  next() { if (this.st.s < this.M.steps.length - 1) { this.st.s++; this.paint(); return true; } return false; },
  prev() { if (this.st.s >= 0) { this.st.s--; this.paint(); return true; } return false; },
  stop() {}
};

// ───────────────────────────── go
fitStage();
setTimeout(() => { $("#help").style.transition = "opacity 1s"; $("#help").style.opacity = 0; }, 8000);
const h = parseInt((location.hash || "#1").slice(1), 10);
show(isNaN(h) ? 0 : h - 1);

