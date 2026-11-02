// Render .drawio files with mxGraph (draw.io's engine) in Chromium; write PNG next to each file.
// usage: node render.js a.drawio b.drawio ...
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const MX = path.join(__dirname, "node_modules/mxgraph/javascript/src");

(async () => {
  const browser = await chromium.launch();
  for (const f0 of process.argv.slice(2)) {
    const file = path.resolve(f0);
    const xml = fs.readFileSync(file, "utf8");
    const page = await browser.newPage({ deviceScaleFactor: 3, viewport: { width: 1600, height: 1000 } });
    const html = `<!doctype html><html><head><meta charset="utf-8">
      <script>mxBasePath="file://${MX}";mxLoadResources=false;mxLoadStylesheets=false;mxForceIncludes=false;</script>
      <script src="file://${path.join(__dirname, "node_modules/mxgraph/javascript/mxClient.js")}"></script>
      <style>body{margin:0;background:#fff} #g{position:relative;overflow:hidden;background:#fff}</style></head>
      <body><div id="g"></div></body></html>`;
    const tmp = file + ".html";
    fs.writeFileSync(tmp, html);
    await page.goto("file://" + tmp);
    const size = await page.evaluate((xml) => {
      const doc = mxUtils.parseXml(xml);
      const model = doc.getElementsByTagName("mxGraphModel")[0];
      const W = +model.getAttribute("pageWidth"), H = +model.getAttribute("pageHeight");
      const c = document.getElementById("g");
      c.style.width = W + "px"; c.style.height = H + "px";
      const graph = new mxGraph(c);
      graph.setHtmlLabels(true);
      graph.setEnabled(false);
      new mxCodec(model.ownerDocument).decode(model, graph.getModel());
      return { W, H };
    }, xml);
    await page.waitForTimeout(300);
    const out = file.replace(/\.drawio$/, ".png");
    await page.locator("#g").screenshot({ path: out });
    fs.unlinkSync(tmp);
    console.log("wrote", out, size.W + "x" + size.H);
    await page.close();
  }
  await browser.close();
})();
