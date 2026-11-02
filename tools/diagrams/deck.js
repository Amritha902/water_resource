// Paper 1 deck: architecture diagrams drawn in draw.io (d*.drawio, rendered with mxGraph), results native.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require("/root/.claude/skills/synced/255d15fd-5aca-411d-bb76-f644d356f751_9cb1af74-6e1e-4034-ab80-67bbea4c3a3d/pptx/scripts/apply_theme.js");

const HERE = __dirname;
const SIZES = JSON.parse(fs.readFileSync(path.join(HERE, "sizes.json"), "utf8"));
const N = JSON.parse(fs.readFileSync(path.join(HERE, "numbers.json"), "utf8"));
const FIG = "/home/user/water_resource/docs/figures/final";

const THEME = {
  name: "Water Teal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A33", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EEF4F6",
    accent1: "1C6E8C", accent2: "1B7F5A", accent3: "C8611E", accent4: "B03A2E",
    accent5: "7A7A74", accent6: "A07A2C", hlink: "1C6E8C", folHlink: "5B6B73",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Paper 1: reproduction, our controller and held-out tests";
pres.author = "Amritha";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.8, fontSize: 40, bold: true,
      color: C.background1, valign: "bottom", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.2, w: 11.7, h: 1.4, fontSize: 18,
      color: "CADCE3", valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.33, h: 0.85, fontSize: 30, bold: true,
      color: C.text2, valign: "middle", align: "left", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.55, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "8A979D" },
});

const notes = {};
function content(title, sect) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sect });
  s.addText(title, { placeholder: "title" });
  return s;
}
function fit(file, x, y, w, h) {
  const [pw, ph] = SIZES[file];
  const r = pw / ph;
  let W = w, H = w / r;
  if (H > h) { H = h; W = h * r; }
  return { path: path.join(HERE, file), x: x + (w - W) / 2, y, w: W, h: H };
}
function caption(s, text, y, opts = {}) {
  s.addText(text, { x: 0.5, y, w: 12.33, h: 0.5, fontSize: 13, color: C.text1, isTextBox: true, margin: 0,
    valign: "top", ...opts });
}
function stat(s, x, y, w, big, small, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.55, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 },
    rectRadius: 0.08, objectName: "stat card" });
  s.addText(big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.8, fontSize: 36, bold: true, color, isTextBox: true,
    margin: 0, fontFace: THEME.headFontFace });
  s.addText(small, { x: x + 0.2, y: y + 0.9, w: w - 0.4, h: 0.6, fontSize: 13, color: C.text1, isTextBox: true,
    margin: 0, valign: "top" });
}

// ---------------------------------------------------------------- 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Introduction" });
s.addText("Paper 1 reproduced, and what actually controls a wastewater plant better", { placeholder: "title" });
s.addText("MAACC (Wang et al., IEEE T-II 2026) on the official BSM1 benchmark  ·  Amritha", { placeholder: "body" });
s.addNotes("Paper 1 claims a learning controller that cuts the oxygen tracking error by 92 to 95 per cent. We rebuilt it exactly on the official benchmark, built our own controller, and tested everything on data nothing was tuned on.");

// ---------------------------------------------------------------- 2. plant
s = content("The plant and the control task", "Introduction");
s.addImage(fit("notitle/d0_plant_crop.png", 0.5, 1.35, 12.33, 4.6));
caption(s, "Two loops, as in paper 1: aeration KLa,5 keeps oxygen in tank 5 at 2 mg/L; internal recycle Q_a keeps nitrate in tank 2 at 1 mg/L. Score: IAE, the mean absolute tracking error (paper 1, eq. 36).", 6.25);
s.addNotes("BSM1 is the IWA benchmark plant. Paper 1 controls two inputs, drawn in green. All our runs use the official plant code and the official influent files.");

// ---------------------------------------------------------------- 3. paper architecture
pres.addSection({ title: "Paper 1" });
s = content("Paper 1's architecture: two adaptive-critic agents on top of a PID", "Paper 1");
s.addImage(fit("notitle/d1_paper_crop.png", 0.5, 1.3, 12.33, 5.75));
s.addNotes("Each loop has an agent with an action network and a critic. The output is the PID's input plus a learned part, u = a + b. The critic learns from one-step temporal-difference errors, each step used once.");

// ---------------------------------------------------------------- 4. reproduction
s = content("Rebuilt exactly as printed, MAACC never leaves its PID", "Paper 1");
stat(s, 0.5, 1.5, 3.9, "0 %", "change in oxygen error against its PID (paper claims −92 to −95 %)", HEX.accent4);
stat(s, 4.72, 1.5, 3.9, "80 / 80", "runs in which MAACC equals its PID (20 runs × 4 cases)", HEX.accent4);
stat(s, 8.93, 1.5, 3.9, "10⁻⁴", "size of its learning signal with the printed weights", HEX.accent4);
const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center" } });
const cell = (t, fill, bold) => ({ text: t, options: { align: "center", color: HEX.dk1, fill: { color: fill }, bold: !!bold } });
s.addTable([
  [{ text: "Oxygen error, 14 days (IAE)", options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } } },
    H("Paper's PID (Table II)"), H("Our PID"), H("MAACC, paper's claim"), H("MAACC as printed, our run")],
  ...[["Dry", "0.0114", "0.0114", "0.00053", "0.0114"], ["Rain", "0.0098", "0.0108", "0.00051", "0.0108"],
      ["Storm", "0.0117", "0.0118", "0.00095", "0.0118"]].map(([w, a, b, c, d]) => [
    { text: w, options: { bold: true, color: HEX.dk1 } }, cell(a, "F1F1EE"), cell(b, "F1F1EE"), cell(c, "F6D9D6"),
    cell(d, "F6D9D6", true)]),
], { x: 0.5, y: 3.45, w: 12.33, colW: [3.13, 2.3, 2.3, 2.3, 2.3], fontSize: 15, fontFace: "Calibri", rowH: 0.5,
  valign: "middle", border: { type: "solid", pt: 0.75, color: "FFFFFF" }, margin: [0.04, 0.12, 0.04, 0.12] });
s.addText("Our plant matches the paper's (PID columns). The PID gains, 45 s step, 14 days, input limits and noise are exactly as printed. "
  + "Differences we know of: the set points are printed as [1, 2] but the figures match [2, 1]; our nitrate PID error is about half the paper's.",
  { x: 0.5, y: 5.65, w: 12.33, h: 1.0, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("So the improvement the paper reports does not come from the method as printed. The plant matches, the settings match, the learner simply does not move.");

// ---------------------------------------------------------------- 5. ours
pres.addSection({ title: "Our work" });
s = content("Our controller: what is new", "Our work");
s.addImage(fit("notitle/d2_ours_crop.png", 0.5, 1.3, 9.0, 5.75));
const rows5 = [["Tuned PID", "P and I re-tuned per loop on one dry week. Alone: −85 % oxygen error (test week, dry)"],
  ["Daily ILC", "learns the daily load pattern, no neural network. With the tuned PID: −90 / −89 / −80 %"],
  ["Oxygen filter", "with gains ×0.5: never failed with the official sensors, but no better than the paper's PID there"]];
rows5.forEach(([t, d], i) => {
  const y = 1.55 + i * 1.7;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.85, y, w: 2.98, h: 1.45, fill: { color: "E2F3EA" },
    line: { color: HEX.accent2, width: 1.5 }, rectRadius: 0.08, objectName: "new item " + (i + 1) });
  s.addText(t, { x: 10.0, y: y + 0.1, w: 2.7, h: 0.4, fontSize: 16, bold: true, color: "0E3B2A", isTextBox: true, margin: 0 });
  s.addText(d, { x: 10.0, y: y + 0.5, w: 2.7, h: 0.9, fontSize: 13, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
});
s.addNotes("Green is new. The tuned PID and the daily iterative learning control do the work. The learned correction, SAC with imitation learning or the repaired MAACC, was built and tested; on top of the tuned PID with ILC it adds nothing.");

// ---------------------------------------------------------------- 6. ILC
s = content("The daily ILC: learning the load pattern, one update per night", "Our work");
s.addImage(fit("notitle/d3_ilc_crop.png", 0.5, 1.4, 12.33, 4.6));
caption(s, "The load on the plant repeats every day; most of the PID's remaining error repeats with it. The ILC removes that repeating part before it happens.", 6.25);
s.addNotes("This is iterative learning control. Each night it takes over part of the correction the PID had to make, and applies it a few minutes earlier the next day. Bumpless transfer removed a jump at midnight; anti-windup stopped it from hurting when the aeration is at its limit.");

// ---------------------------------------------------------------- 7. repaired MAACC
s = content("A fair baseline: the paper's learner with its training repaired", "Our work");
s.addImage(fit("notitle/d4_repaired_crop.png", 0.5, 1.35, 12.33, 4.3));
s.addText([
  { text: "Result: ", options: { bold: true } },
  { text: "it learns. Oxygen error −85 % in dry weather (−52 % rain, −67 % storm), but it moves the aeration 3.7 × as much as the PID." },
], { x: 0.5, y: 6.0, w: 12.33, h: 0.8, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("To compare fairly we kept the paper's structure and fixed only the training, with standard fixes. It does learn, so the paper's idea can work; its printed training is what fails.");

// ---------------------------------------------------------------- 8. method
s = content("How we tested: nothing was tuned on the test data", "Our work");
s.addImage(fit("notitle/d5_method_crop.png", 0.5, 1.35, 12.33, 4.85));
caption(s, "Both held-out protocols were pushed to git before their runs, so the order can be checked. The audit re-ran stored runs and recomputed every quoted number.", 6.4);
s.addNotes("Tuning and training used one dry week. The second week was the test. Then two held-out tests on new data and real sensor models, each fixed in writing before running.");

// ---------------------------------------------------------------- 9. results chart
pres.addSection({ title: "Results" });
s = content("Official test week: tuned PID + ILC is as good as any learner", "Results");
const ctrl = [["MAACC as printed", "MAACC"], ["Repaired MAACC", "rMAACC"], ["Ours: PID ×3 + ILC + SAC-IL", "Ours"],
  ["Robust (filter, ×0.5, ILC-AW)", "Robust"], ["Tuned PID + ILC", "PIDtuned+ILC"]];
const labels = ctrl.map((c) => c[0]);
const series = [["Dry", "tw_dry"], ["Rain", "tw_rain"], ["Storm", "tw_storm"]].map(([name, k]) => ({
  name, labels, values: ctrl.map(([, c]) => -N[k][c].pct),
}));
s.addChart(pres.charts.BAR, series, {
  x: 0.5, y: 1.35, w: 8.6, h: 5.6, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: ["1C6E8C", "A07A2C", "5B6B73"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: "1B2A33", dataLabelFormatCode: "0\"%\"",
  catAxisLabelFontSize: 13, catAxisLabelColor: "1B2A33", catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontSize: 11, valAxisLabelColor: "5B6B73", valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: "0\"%\"",
  valGridLine: { color: "E1E0D9", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: "1B2A33", legendFontFace: "+mn-lt",
  showTitle: true, title: "Reduction of oxygen error against the paper's PID (test week, higher is better)",
  titleFontSize: 13, titleColor: "1B2A33", titleFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
});
s.addText([
  { text: "What this shows", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  { text: "A PID tuned term by term plus the daily ILC: −90 / −89 / −80 %.", options: { bullet: true, breakLine: true } },
  { text: "SAC-IL on top of it adds 0 %.", options: { bullet: true, breakLine: true } },
  { text: "The repaired MAACC beats SAC-IL in the same stack.", options: { bullet: true, breakLine: true } },
  { text: "MAACC as printed: 0 %.", options: { bullet: true } },
], { x: 9.4, y: 1.6, w: 3.43, h: 4.6, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, paraSpaceAfter: 10, valign: "top" });
s.addNotes("Week 2 of the official files, never used for tuning or training. Learned controllers are the mean of five training runs.");

// ---------------------------------------------------------------- 10. held-out table
s = content("Held-out test 2: new influent and real sensors", "Results");
const cols = [["H4", "New influent"], ["H5", "Overload"], ["H6", "Official sensors"], ["H7", "New influent + sensors"]];
const trows = [["Paper's PID", "PID"], ["MAACC as printed", "MAACC"], ["Repaired MAACC", "rMAACC"],
  ["Ours: PID ×3 + ILC + SAC-IL", "Ours"], ["Tuned PID + ILC", "PIDtuned+ILC"],
  ["Tuned PID + ILC, anti-windup", "PIDtuned+ILC-AW"], ["Robust (filter, ×0.5, ILC-AW)", "Robust"]];
const hdr = [{ text: "Oxygen error vs paper's PID", options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } } },
  ...cols.map(([k, t]) => ({ text: `${k}  ${t}`, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center" } }))];
const body = trows.map(([lab, c]) => [
  { text: lab, options: { bold: c === "Robust", color: HEX.dk1 } },
  ...cols.map(([k]) => {
    const d = N[k][c];
    if (c === "PID") return { text: "reference", options: { align: "center", color: "5B6B73", fill: { color: "F1F1EE" } } };
    if (d.failed === d.n) return { text: "✗ all failed", options: { align: "center", bold: true, color: "FFFFFF", fill: { color: "3B3B38" } } };
    const p = d.pct;
    let t = (p > 0 ? "+" : p < 0 ? "−" : "") + Math.abs(p) + " %";
    if (d.failed) t += `  (✗ ${d.failed}/${d.n})`;
    const fill = p <= -20 ? "CFE6DA" : p < 0 ? "E8F3ED" : p < 5 ? "F1F1EE" : "F6D9D6";
    return { text: t, options: { align: "center", color: HEX.dk1, fill: { color: fill } } };
  }),
]);
s.addTable([hdr, ...body], { x: 0.5, y: 1.4, w: 12.33, colW: [3.93, 2.1, 2.1, 2.1, 2.1], fontSize: 14, fontFace: "Calibri",
  border: { type: "solid", pt: 0.75, color: "FFFFFF" }, rowH: 0.52, valign: "middle", margin: [0.04, 0.12, 0.04, 0.12] });
caption(s, "Green: better than the paper's PID; red: worse; dark: every run failed (oxygen left 0.3–4 mg/L). Each % compares only runs both controllers finished. Protocol pushed before the 432 runs.", 5.85, { fontSize: 12, color: "5B6B73" });
s.addNotes("New BSM2 influent windows, raw and after a simple primary clarifier, and the official sensor models with new noise seeds. Our SAC-IL controller fails every run with the official sensor noise. The robust tuned PID never fails, but with sensor noise it is no better than the paper's PID.");

// ---------------------------------------------------------------- 11. fixes
s = content("Two fixes, tested on new data: one works, one half-works", "Results");
stat(s, 0.5, 1.5, 3.0, "−39 %", "ILC anti-windup under overload, better in 5 of 6 new windows", HEX.accent2);
stat(s, 0.5, 3.25, 3.0, "0 %", "its cost in normal operation", HEX.accent2);
stat(s, 0.5, 5.0, 3.0, "+3 / +24 %", "robust PID vs paper's PID with real sensor noise", HEX.accent4);
s.addImage({ path: path.join(FIG, "f6_sensor_noise.png"), x: 3.85, y: 1.45, w: 8.98, h: 8.98 * 1240 / 2400 });
s.addText("True oxygen with the official sensors (noise 0.25 g/m³): the paper's PID and the robust controller stay in range; our SAC-IL controller fails after 11 h.",
  { x: 3.85, y: 6.2, w: 8.98, h: 0.6, fontSize: 12, color: "5B6B73", isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Anti-windup fixed the ILC under overload at no cost. The robust version never failed, kept −83 % on new normal influent, and moves the aeration no more than the paper's PID, but under real sensor noise nothing we built beats the paper's PID. That is the open problem.");

// ---------------------------------------------------------------- 12. what's new table
pres.addSection({ title: "Conclusion" });
s = content("Paper 1 against our work", "Conclusion");
const cmp = [
  ["", "Paper 1", "Our work"],
  ["Controller", "two adaptive-critic agents on a PID", "tuned PID + daily ILC (bumpless, anti-windup); learned options tested"],
  ["Learning", "online, one step at a time; does not learn as printed", "ILC learns daily; repaired MAACC and SAC-IL trained on one dry week"],
  ["Testing", "learns and is scored on the same 14-day run", "train week → test week → two pre-registered held-out tests"],
  ["Robustness", "noise added to the two inputs only", "new influent, overload, official sensor delay and noise"],
  ["Verification", "—", "audit: stored runs re-run exactly, 76 of 76 checks pass"],
  ["Oxygen result", "claims −92 to −95 %", "−80 to −90 % on the test week (tuned PID + ILC); not robust to real sensor noise"],
];
s.addTable(cmp.map((r, i) => r.map((t, j) => ({ text: t, options: {
  bold: i === 0 || j === 0, color: i === 0 ? "FFFFFF" : HEX.dk1,
  fill: { color: i === 0 ? HEX.dk2 : j === 2 ? "E2F3EA" : j === 1 ? "F1F1EE" : "FFFFFF" } } }))),
  { x: 0.5, y: 1.4, w: 12.33, colW: [2.1, 4.4, 5.83], fontSize: 14, fontFace: "Calibri", rowH: 0.66, valign: "middle",
    border: { type: "solid", pt: 0.75, color: "FFFFFF" }, margin: [0.04, 0.12, 0.04, 0.12] });
s.addNotes("The contribution is a reproducibility and robustness study plus a simple controller that works on nominal data, with every test fixed before it was run.");

// ---------------------------------------------------------------- 13. claims and limits
s = content("What we can claim, and what we cannot", "Conclusion");
const col = (x, title, color, items) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: 5.95, h: 4.9, fill: { color: color === HEX.accent2 ? "F4FBF7" : "FBEFEE" },
    line: { color, width: 1.5 }, rectRadius: 0.06, objectName: title });
  s.addText(title, { x: x + 0.3, y: 1.6, w: 5.4, h: 0.5, fontSize: 20, bold: true, color, isTextBox: true, margin: 0 });
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x: x + 0.3, y: 2.25, w: 5.4, h: 3.95, fontSize: 18, color: C.text1, isTextBox: true, margin: 0, paraSpaceAfter: 14, valign: "top" });
};
col(0.5, "We can claim", HEX.accent2, [
  "MAACC as printed does not learn; it fails every held-out condition",
  "With repaired training it learns (−85 % dry), but is not robust to sensor noise",
  "Tuned PID + daily ILC: −80 to −90 % on the official test week",
  "Anti-windup makes the ILC safe under overload (−39 %)",
  "Every number re-run and audited",
]);
col(6.88, "We cannot claim", HEX.accent4, [
  "The paper's −92 to −95 %",
  "That SAC-IL beats good tuning",
  "Any gain over the paper's PID under real sensor noise",
  "A real-plant result (simulation, one benchmark)",
  "Statistical significance (5 runs, 6 windows)",
]);
s.addNotes("Be clear about both columns. Every percentage is against our rebuilt PID, which matches the paper's PID on oxygen.");

// ---------------------------------------------------------------- 14. next
s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Conclusion" });
s.addText("Next: make the controller robust to real sensor noise", { placeholder: "title" });
s.addText("1. Model-based filtering (Kalman filter or observer) instead of a simple filter\n2. Train the repaired MAACC with the official sensor models in the loop\n3. Report nominal and robust results side by side, as standard",
  { placeholder: "body" });
s.addNotes("Under the official sensor noise, nothing we built beats the paper's PID. That is the open problem and the next step.");

(async () => {
  const out = path.join(HERE, "Paper1_Architecture_Deck.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
