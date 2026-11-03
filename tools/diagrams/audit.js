// Audit .drawio diagrams with mxGraph: label overflow, label-vs-box overlaps, box-vs-box overlaps, edge-through-box.
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
const MX = path.join(__dirname, "node_modules/mxgraph/javascript/src");
(async () => {
  const browser = await chromium.launch();
  for (const f0 of process.argv.slice(2)) {
    const file = path.resolve(f0), xml = fs.readFileSync(file, "utf8");
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const tmp = file + ".a.html";
    fs.writeFileSync(tmp, `<!doctype html><html><head><meta charset="utf-8"><script>mxBasePath="file://${MX}";mxLoadResources=false;mxLoadStylesheets=false;</script><script src="file://${path.join(__dirname,"node_modules/mxgraph/javascript/mxClient.js")}"></script></head><body style="margin:0"><div id="g" style="position:relative"></div></body></html>`);
    await page.goto("file://" + tmp);
    const issues = await page.evaluate((xml) => {
      const doc = mxUtils.parseXml(xml), model = doc.getElementsByTagName("mxGraphModel")[0];
      const c = document.getElementById("g"); c.style.width = model.getAttribute("pageWidth") + "px"; c.style.height = model.getAttribute("pageHeight") + "px";
      const g = new mxGraph(c); g.setHtmlLabels(true); new mxCodec(model.ownerDocument).decode(model, g.getModel());
      const out = [], cells = Object.values(g.getModel().cells).filter(x => x.id !== "0" && x.id !== "1");
      const txt = s => (s || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 40);
      const inter = (a, b, m = 0) => a.x < b.x + b.width - m && b.x < a.x + a.width - m && a.y < b.y + b.height - m && b.y < a.y + a.height - m;
      const labelBox = st => { if (!st || !st.text || !st.text.node) return null;
        const el = st.text.node.querySelector("div div div") || st.text.node.querySelector("div div") || st.text.node;
        const r = el.getBoundingClientRect(); return r.width ? { x: r.left, y: r.top, width: r.width, height: r.height } : null; };
      const verts = cells.filter(x => x.vertex), edges = cells.filter(x => x.edge);
      const isText = v => /^text;|strokeColor=none/.test(v.style || "");
      const solid = verts.filter(v => !isText(v) && !/dashed=1;verticalAlign=top/.test(v.style) && !/arcSize=50/.test(v.style));
      for (const v of verts) {               // label inside its own box (2 px tolerance)
        const st = g.view.getState(v), lb = labelBox(st);
        if (!lb || isText(v)) continue;
        const b = st;
        if (lb.x < b.x + 2 || lb.y < b.y + 1 || lb.x + lb.width > b.x + b.width - 2 || lb.y + lb.height > b.y + b.height - 1)
          out.push(`OVERFLOW  "${txt(v.value)}" text ${Math.round(lb.width)}x${Math.round(lb.height)} in box ${b.width}x${b.height}`);
      }
      for (const v of verts.filter(isText)) {  // free text over a box
        const lb = labelBox(g.view.getState(v)); if (!lb) continue;
        for (const w of solid) { const s = g.view.getState(w); if (inter(lb, s, 1)) out.push(`TEXT ON BOX  "${txt(v.value)}" over "${txt(w.value)}"`); }
      }
      for (let i = 0; i < solid.length; i++) for (let j = i + 1; j < solid.length; j++) {
        const a = g.view.getState(solid[i]), b = g.view.getState(solid[j]);
        if (inter(a, b, 1)) out.push(`BOX OVERLAP  "${txt(solid[i].value)}" / "${txt(solid[j].value)}"`);
      }
      for (const e of edges) {
        const st = g.view.getState(e); if (!st) continue;
        const lb = labelBox(st);
        if (lb) for (const w of solid) { const s = g.view.getState(w); if (inter(lb, s, 1)) out.push(`EDGE LABEL ON BOX  "${txt(e.value)}" over "${txt(w.value)}"`); }
        const pts = st.absolutePoints; const ends = [e.source, e.target];
        for (let k = 0; k + 1 < pts.length; k++) { const p = pts[k], q = pts[k + 1];
          const seg = { x: Math.min(p.x, q.x), y: Math.min(p.y, q.y), width: Math.abs(p.x - q.x) || 0.5, height: Math.abs(p.y - q.y) || 0.5 };
          for (const w of solid) { if (ends.includes(w)) continue; const s = g.view.getState(w);
            if (inter(seg, { x: s.x + 2, y: s.y + 2, width: s.width - 4, height: s.height - 4 })) out.push(`EDGE THROUGH BOX  edge "${txt(e.value) || e.id}" crosses "${txt(w.value)}"`); } }
        if (pts.length >= 2) { const a = pts[pts.length - 2], b = pts[pts.length - 1]; const L = Math.hypot(a.x - b.x, a.y - b.y);
          if (L < 14) out.push(`SHORT LAST SEGMENT  edge "${txt(e.value) || e.id}" ${L.toFixed(0)} px (arrowhead cramped)`); }
      }
      return [...new Set(out)];
    }, xml);
    fs.unlinkSync(tmp);
    console.log(`\n== ${path.basename(file)}: ${issues.length} issue(s)`); issues.forEach(i => console.log("  " + i));
    await page.close();
  }
  await browser.close();
})();
