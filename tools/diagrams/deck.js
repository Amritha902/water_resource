// Paper 1 deck, version 3: mirrors the journal paper. Black-and-white figures only.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require("/root/.claude/skills/synced/255d15fd-5aca-411d-bb76-f644d356f751_9cb1af74-6e1e-4034-ab80-67bbea4c3a3d/pptx/scripts/apply_theme.js");

const FIG = "/home/user/water_resource/docs/paper/figures";
const SIZES = JSON.parse(fs.readFileSync(path.join(__dirname, "sizes_v3.json"), "utf8"));
const THEME = {
  name: "Journal Mono", headFontFace: "Times New Roman", bodyFontFace: "Times New Roman",
  colors: { dk1: "000000", lt1: "FFFFFF", dk2: "1A1A1A", lt2: "F2F2F2", accent1: "404040", accent2: "7F7F7F",
    accent3: "A6A6A6", accent4: "262626", accent5: "595959", accent6: "BFBFBF", hlink: "000000", folHlink: "404040" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Daily ILC with a tuned PID prior for dissolved oxygen control in BSM1";
pres.author = "Amritha, Soumya Ranjan Mahapatro";
const INK = "000000", GREY = "595959", RULE = "BFBFBF", HEAD = "F2F2F2";
const FONT = "Times New Roman";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.33, h: 0.75, fontSize: 28, bold: true,
      color: INK, fontFace: FONT, valign: "middle", margin: 0 }, text: "" } },
    { line: { x: 0.5, y: 1.1, w: 12.33, h: 0, line: { color: RULE, width: 0.75 } } },
  ],
  slideNumber: { x: 12.55, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: GREY, fontFace: FONT },
});

const content = (t) => { const s = pres.addSlide({ masterName: "CONTENT" }); s.addText(t, { placeholder: "title" }); return s; };
const CMP = "/tmp/claude-0/-home-user-gan-driver/a50d7365-fe82-52fd-bebb-f039d7ffe97e/scratchpad/cmp/w/fig";
function fit(name, x, y, w, h) {
  const [pw, ph] = SIZES[name]; let W = w, H = w * ph / pw;
  if (H > h) { H = h; W = h * pw / ph; }
  const file = name.startsWith("cmp/") ? path.join(CMP, name.slice(4) + ".png") : path.join(FIG, name + ".png");
  return { path: file, x: x + (w - W) / 2, y, w: W, h: H };
}
const text = (s, t, o) => s.addText(t, { fontFace: FONT, color: INK, isTextBox: true, margin: 0, valign: "top", fontSize: 16, ...o });
const bullets = (s, items, o) => text(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
  { paraSpaceAfter: 8, ...o });
const caption = (s, t, y) => text(s, t, { x: 0.5, y, w: 12.33, h: 0.5, fontSize: 13, color: GREY });
function table(s, rows, o) {
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: String(c), options: {
    bold: i === 0, fill: { color: i === 0 ? HEAD : "FFFFFF" }, color: INK, align: j === 0 ? "left" : "center",
    border: [{ type: i === 0 ? "solid" : "none", pt: 0.75, color: INK }, { type: "none" },
      { type: (i === 0 || i === rows.length - 1 || (o.rules || []).includes(i)) ? "solid" : "none", pt: 0.75, color: INK }, { type: "none" }],
  } }))), { fontFace: FONT, fontSize: 14, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08], ...o });
}

// 1. title
let s = pres.addSlide({ background: { color: "FFFFFF" } });
text(s, "Daily Iterative Learning Control With a Tuned PID Prior for Dissolved Oxygen Control in BSM1: Comparison With Multiagent Adaptive Critic Control",
  { x: 0.8, y: 1.8, w: 11.7, h: 2.2, fontSize: 32, bold: true, valign: "bottom" });
s.addShape(pres.shapes.LINE, { x: 0.8, y: 4.2, w: 11.7, h: 0, line: { color: INK, width: 1 } });
text(s, "Amritha and Dr. Soumya Ranjan Mahapatro\nVellore Institute of Technology, Chennai", { x: 0.8, y: 4.4, w: 11.7, h: 1.0, fontSize: 18, color: GREY });

// 2. base paper and plant
s = content("Base paper and control problem");
text(s, [
  { text: "Base paper: ", options: { bold: true } },
  { text: "D. Wang, X. Li, J. Ren and J. Qiao, “Multiagent Adaptive Critic Control With Expert Knowledge for Wastewater Treatment Plants,” IEEE Trans. Ind. Informat., 2026. Reported oxygen error reductions of 92 to 95 % relative to its PID prior." },
], { x: 0.5, y: 1.3, w: 12.33, h: 1.0, fontSize: 16 });
s.addImage(fit("plant", 0.5, 2.45, 12.33, 3.4));
caption(s, "BSM1: two anoxic and three aerated tanks, ten-layer settler. Manipulated: KLa,5 (oxygen in tank 5 to 2 mg/L) and Qa (nitrate in tank 2 to 1 mg/L). Sampling period 45 s.", 6.1);
s.addNotes("The base paper controls the two standard BSM1 loops with one adaptive critic agent per loop on top of an incremental PID.");

// 3. architectures
s = content("Base architecture and proposed architecture");
const a = fit("base", 0.5, 1.45, 5.95, 4.3), b = fit("proposed", 6.75, 1.45, 6.08, 4.3);
s.addImage(a); s.addImage(b);
text(s, "(a) MAACC, one agent per loop [base paper]", { x: 0.5, y: 5.95, w: 5.95, h: 0.4, fontSize: 14, align: "center" });
text(s, "(b) Proposed: tuned PID, daily ILC with anti-windup, measurement filter", { x: 6.75, y: 5.95, w: 6.08, h: 0.4, fontSize: 14, align: "center" });
s.addNotes("In MAACC the action network output is integrated and added to the PID output. The proposed controller replaces the networks with a daily feedforward learned from the PID output.");

// 4. ILC algorithm
s = content("Daily iterative learning control");
text(s, "Profile update at the end of each day d (1920 slots of 45 s):", { x: 0.5, y: 1.35, w: 12.33, h: 0.4 });
text(s, "f(d+1) = S_w [ f(d) + β · L_ℓ ( u_PID(d) − mean u_PID(d) ) ]", { x: 0.5, y: 1.85, w: 12.33, h: 0.5, fontSize: 20, align: "center", italic: true });
text(s, [
  "1:  for each control step k with slot j",
  "2:    u_PID ← PID step, limited to [U_min − f(j), U_max − f(j)]",
  "3:    u ← sat(u_PID + f(j))",
  "4:    record g(j) ← u_PID;  o(j) ← 1 if u is not saturated",
  "5:    if j = 1920 and at least half of the slots are unsaturated",
  "6:        δ(j) ← o(j)(g(j) − mean g);   f⁺ ← S_w[f + β L_ℓ δ]",
  "7:        PID state ← PID state − (f⁺(1) − f(1));   f ← f⁺",
].join("\n"), { x: 0.5, y: 2.6, w: 8.6, h: 3.4, fontSize: 12.5, fontFace: "Courier New" });
bullets(s, ["L_ℓ: shift earlier by ℓ = 4 slots (3 min)", "S_w: circular moving average, w = 30 slots", "β = 1 (tuned PID)",
  "Line 2 and the saturation test: anti-windup", "Line 7: bumpless update at midnight"], { x: 9.3, y: 2.6, w: 3.53, h: 3.4, fontSize: 14 });
s.addNotes("The ILC transfers the recurring part of the PID correction to a feedforward. Settings were selected on dry days 0 to 7.");

// 5. data
s = content("Data and evaluation protocol");
s.addImage(fit("influent", 0.5, 1.4, 6.4, 3.6));
bullets(s, [
  "BSM1 influent files Inf_dry_2006, Inf_rain_2006, Inf_strm_2006 (14 days, 15 min), the same benchmark files as the base paper",
  "Source: github.com/davidfernand/BSM1_in_Python, commit 5dadb20; dry file identical to the one in bsm2-python 0.0.16",
  "Tuning and training: dry weather, days 0 to 7 only",
  "Test period: days 7 to 14 of dry, rain and storm weather",
  "Additional conditions: BSM2 influent windows, actuator saturation, BSM1 sensor models",
], { x: 7.2, y: 1.45, w: 5.63, h: 5.2, fontSize: 15 });
caption(s, "Influent flow rate of the three weather files.", 5.15);

// 6. training
s = content("Training of the learned corrections");
s.addImage(fit("training", 0.5, 1.4, 7.4, 4.3));
bullets(s, [
  "Five independent training runs per method, five episodes of seven days",
  "rMAACC: oxygen IAE 0.0115 to 0.0013–0.0023 (retained policies)",
  "SAC-IL: stays near its imitation start (about 0.0020); one run diverged in episodes 3 and 4",
  "A complete retraining from scratch reproduced all ten training curves to five decimal places",
], { x: 8.2, y: 1.5, w: 4.63, h: 5.0, fontSize: 15 });
caption(s, "Oxygen IAE on the training period after each episode. Dashed line: PID prior.", 5.9);

// 7. test period table
s = content("Test period: change in oxygen IAE relative to the PID prior");
table(s, [
  ["Controller", "Dry", "Rain", "Storm", "Actuator noise", "KLa,5 movement"],
  ["PID prior (IAE)", "0.0113", "0.0100", "0.0120", "0.0249", "501"],
  ["MAACC [base paper]", "0 %", "0 %", "0 %", "0 %", "501"],
  ["rMAACC", "−85 %", "−52 %", "−67 %", "−13 %", "1835"],
  ["PID×3 + ILC + SAC-IL", "−87 %", "−82 %", "−74 %", "−26 %", "982"],
  ["Tuned PID", "−85 %", "−85 %", "−77 %", "−22 %", "852"],
  ["Tuned PID + ILC (proposed)", "−90 %", "−89 %", "−80 %", "−23 %", "852"],
  ["Tuned PID + ILC + SAC-IL", "−90 %", "−88 %", "−80 %", "−22 %", "874"],
  ["Noise-robust variant (proposed)", "−83 %", "−80 %", "−73 %", "+10 %", "522"],
], { x: 0.5, y: 1.4, w: 12.33, colW: [3.9, 1.5, 1.5, 1.5, 2.0, 1.93], rowH: 0.48, rules: [3] });
caption(s, "Days 7 to 14, excluded from tuning and training. Learned corrections: mean of five training runs. Aeration energy (3690 kWh/d) and effluent ammonium are equal for all controllers.", 6.0);

// 8. trajectories
s = content("Trajectories of S_O,5 and S_NO,2 over 14 days");
const tr = fit("trajectories", 0.5, 1.25, 8.0, 5.75);
s.addImage(tr);
bullets(s, [
  "Format of Figs. 4 to 6 of the base paper",
  "Grey: PID prior (MAACC coincides with it). Black: proposed",
  "PID prior: oxygen falls by up to 0.10 mg/L at each morning load increase",
  "Proposed: within 0.011 mg/L of the set point in dry and rain weather",
  "Storm on day 9 and nitrate dips in rain and storm: the same for both controllers (Qa at its upper bound)",
], { x: 8.8, y: 1.5, w: 4.03, h: 5.4, fontSize: 15 });

// 9. inputs and actuator noise
s = content("Control inputs and actuator noise");
s.addImage(fit("inputs", 0.5, 1.35, 5.9, 4.9));
s.addImage(fit("noise", 6.93, 1.35, 5.9, 4.9));
text(s, "(a) KLa,5 and Qa of the proposed controller, three weathers", { x: 0.5, y: 6.35, w: 5.9, h: 0.4, fontSize: 13, align: "center", color: GREY });
text(s, "(b) Dry weather with actuator noise (3 % of each upper bound); proposed: −23 % oxygen IAE", { x: 6.93, y: 6.35, w: 5.9, h: 0.5, fontSize: 13, align: "center", color: GREY });

// 10. mechanism
s = content("Mechanism: a small correction at the load increase");
s.addImage(fit("inside", 0.5, 1.35, 6.0, 5.6));
bullets(s, [
  "Two days of the dry test period",
  "(a) PID prior (dashed) against proposed (solid)",
  "(b) The aeration differs by at most 6 d⁻¹, applied at the morning load increase",
  "Mean oxygen error over the two days: 0.0112 (PID prior) and 0.0010 mg/L (proposed)",
], { x: 7.0, y: 1.6, w: 5.83, h: 5.0, fontSize: 16 });

// 11. additional conditions
s = content("Additional conditions: change in oxygen IAE relative to the PID prior");
table(s, [
  ["Controller", "H4 new influent", "H5 saturation", "H6 sensors", "H7 influent + sensors"],
  ["PID prior (IAE)", "0.0105", "0.0815", "0.261", "0.178"],
  ["rMAACC", "−73 %", "−8 %", "+1 % (5/45)", "+6 % (1/30)"],
  ["Tuned PID + ILC", "−89 %", "+48 %", "+46 %", "+50 %"],
  ["Tuned PID + ILC with anti-windup", "−89 %", "−10 %", "–", "+84 % (5/6)"],
  ["Noise-robust variant", "−83 %", "−10 %", "+3 %", "+24 %"],
], { x: 0.5, y: 1.4, w: 12.33, colW: [4.13, 2.0, 2.0, 2.0, 2.2], rowH: 0.5 });
bullets(s, [
  "H4: BSM2 influent windows after a simplified primary clarifier; H5: without it (actuator saturation)",
  "H6: BSM1 sensor models (noise 0.25 mg/L) on the BSM1 files; H7: H4 with the sensor models",
  "In brackets: runs terminated (oxygen outside 0.3 to 4 mg/L); –: no completed run",
  "Under measurement noise, none of the evaluated controllers improved on the PID prior",
], { x: 0.5, y: 4.75, w: 12.33, h: 2.0, fontSize: 14 });


// ---------------------------------------------------------------- comparison with the base paper (local copy only)
if (process.env.COMPARE === "1") {
  s = content("Base paper and this work: 14-day results");
  table(s, [
    ["Weather", "Controller", "S_O,5 IAE", "S_O,5 DEVmax", "S_NO,2 IAE", "S_NO,2 DEVmax"],
    ["Dry", "MAACC, published", "5.27e-4", "0.0092", "3.70e-3", "0.0299"],
    ["", "PID, published", "1.14e-2", "0.0871", "1.45e-2", "0.0882"],
    ["", "PID / MAACC, this work", "1.14e-2", "0.102", "6.35e-3", "0.0575"],
    ["", "Proposed", "1.13e-3", "0.0133", "1.71e-3", "0.0262"],
    ["Rain", "MAACC, published", "5.09e-4", "0.0098", "6.00e-3", "0.167"],
    ["", "PID, published", "9.80e-3", "0.0822", "2.04e-2", "0.190"],
    ["", "PID / MAACC, this work", "1.08e-2", "0.102", "1.07e-2", "0.483"],
    ["", "Proposed", "1.15e-3", "0.0133", "4.95e-3", "0.479"],
    ["Storm", "MAACC, published", "9.47e-4", "0.0712", "4.80e-3", "0.123"],
    ["", "PID, published", "1.17e-2", "0.0916", "1.96e-2", "0.158"],
    ["", "PID / MAACC, this work", "1.18e-2", "0.141", "1.04e-2", "0.514"],
    ["", "Proposed", "1.77e-3", "0.140", "5.65e-3", "0.508"],
  ], { x: 0.5, y: 1.3, w: 12.33, colW: [1.3, 3.4, 1.9, 1.9, 1.9, 1.93], rowH: 0.36, fontSize: 12, rules: [4, 8] });
  caption(s, "Published rows: Table II of the base paper. With the published settings, MAACC stays at its PID in all 80 runs here, so it shares the PID row. Oxygen IAE against each paper's own PID: base paper claims −95 / −95 / −92 %; proposed −90 / −89 / −85 % (14 days).", 6.25);

  s = content("Architecture: base paper Fig. 2 and this work");
  s.addImage(fit("cmp/b_fig2", 0.5, 1.3, 12.33, 2.9));
  s.addImage(fit("cmp/o_base", 0.5, 4.35, 5.9, 2.5));
  s.addImage(fit("cmp/o_proposed", 6.93, 4.35, 5.9, 2.5));
  text(s, "Top: base paper. Bottom: MAACC as implemented here (left) and the proposed controller (right).", { x: 0.5, y: 6.9, w: 12.33, h: 0.35, fontSize: 12, color: GREY });

  const pair = (title, b, o, capL, capR) => {
    const sl = content(title);
    sl.addImage(fit("cmp/" + b, 0.5, 1.35, 5.95, 5.1));
    sl.addImage(fit("cmp/" + o, 6.88, 1.35, 5.95, 5.1));
    text(sl, capL, { x: 0.5, y: 6.55, w: 5.95, h: 0.45, fontSize: 13, color: GREY, align: "center" });
    text(sl, capR, { x: 6.88, y: 6.55, w: 5.95, h: 0.45, fontSize: 13, color: GREY, align: "center" });
  };
  pair("Influent flow rate", "b_fig3", "o_influent", "Base paper, Fig. 3", "This work (same BSM1 files)");
  pair("Dry weather: S_O,5 and S_NO,2", "b_fig4", "o_traj_dry", "Base paper, Fig. 4 (MAACC and PID)", "This work: PID (grey), proposed (black)");
  pair("Rain weather: S_O,5 and S_NO,2", "b_fig5", "o_traj_rain", "Base paper, Fig. 5", "This work");
  pair("Storm weather: S_O,5 and S_NO,2", "b_fig6", "o_traj_storm", "Base paper, Fig. 6", "This work");
  pair("Control inputs in the three weathers", "b_fig7", "o_inputs", "Base paper, Fig. 7 (MAACC)", "This work (proposed)");
  pair("Dry weather with input noise", "b_fig8", "o_noise", "Base paper, Fig. 8", "This work: PID (grey), proposed (black)");
  pair("Control inputs with and without input noise", "b_fig9", "o_inputs_noise", "Base paper, Fig. 9", "This work (proposed)");
}

// 12. conclusion
s = content("Conclusions and limitations");
bullets(s, [
  "Tuned PID with a daily ILC: oxygen IAE reduced by 90, 89 and 80 % relative to the PID prior of MAACC, without a neural network and with the same aeration energy",
  "MAACC with the published parameters remains equal to its PID prior; with replay and twin critics it reaches 85, 52 and 67 %",
  "SAC-IL adds no further improvement to the proposed controller",
  "Anti-windup reduces the error under actuator saturation by 39 %",
  "Limitation: under the BSM1 sensor noise, no evaluated controller improves on the PID prior",
  "Limitation: one benchmark, five training runs, six windows or nine noise runs per condition; no significance tests",
], { x: 0.5, y: 1.4, w: 12.33, h: 5.4, fontSize: 18, paraSpaceAfter: 12 });

(async () => {
  const out = path.join(__dirname, process.env.COMPARE === "1" ? "Paper1_Deck_comparison.pptx" : "Paper1_Deck.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
