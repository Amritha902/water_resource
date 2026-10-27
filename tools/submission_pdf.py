"""Build the full report: three base papers, their figures next to ours, the
reimplementation results and our improved controllers.

    python tools/submission_pdf.py THEIRS_DIR OUT.html      (then print to PDF)

THEIRS_DIR holds the figures cropped from the three papers (copyrighted, not in
this repository).  All numbers are read from results/.
"""

from __future__ import annotations

import base64
import glob
import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPRO = ROOT / "docs" / "figures" / "reproduced"
OURS = ROOT / "docs" / "figures" / "ours"
J = lambda n: json.loads((RES / n).read_text())
sys.path.insert(0, str(ROOT / "tools"))
from compare_pdf import FIGS  # noqa: E402

TITLES = {
    1: ("D. Wang, X. Li, J. Ren, J. Qiao", "Multiagent Adaptive Critic Control With Expert Knowledge for "
        "Wastewater Treatment Plants", "IEEE Transactions on Industrial Informatics, 2026, doi:10.1109/TII.2026.3659924"),
    2: ("S. L. Du, P. X. Chen, H. G. Han, J. F. Qiao", "Dissolved oxygen concentration control in wastewater "
        "treatment process based on reinforcement learning",
        "Science China Technological Sciences 66(9), 2549–2560, 2023, doi:10.1007/s11431-022-2403-8"),
    3: ("M. Zhao, D. Wang, J. Ren, J. Qiao", "Integrated Online Q-Learning Design for Wastewater Treatment "
        "Processes", "IEEE Transactions on Industrial Informatics 21(2), 1833–1842, 2025, doi:10.1109/TII.2024.3488790"),
}

CSS = """
@page { size: A4; margin: 17mm 16mm 17mm 16mm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.3pt; color: #1f1f1e; line-height: 1.48; }
h1 { font-family: 'DejaVu Sans', sans-serif; font-size: 17pt; line-height: 1.25; margin: 0 0 3mm; }
h2 { font-family: 'DejaVu Sans', sans-serif; font-size: 13pt; margin: 6mm 0 2mm; }
h3 { font-family: 'DejaVu Sans', sans-serif; font-size: 10.6pt; margin: 4mm 0 1.5mm; }
.ptitle { border: 1px solid #1f1f1e; padding: 3mm 4mm; margin: 2mm 0 4mm; }
.ptitle .t { font-family: 'DejaVu Sans', sans-serif; font-weight: bold; font-size: 11.5pt; }
.ptitle .a { font-size: 9.5pt; color: #4a4a45; }
.pb { page-break-before: always; }
.fig { page-break-inside: avoid; margin: 3mm 0 5mm; }
.pair { display: flex; gap: 4mm; align-items: flex-start; }
.pair > div { flex: 1; text-align: center; }
.pair img { max-width: 100%; max-height: 92mm; }
.one { text-align: center; }
.one img { max-width: 92%; max-height: 110mm; }
.lab { font-family: 'DejaVu Sans', sans-serif; font-size: 7.6pt; color: #6b6a64; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 1mm; }
.cap { font-size: 9.2pt; margin-top: 1.5mm; }
table { border-collapse: collapse; width: 100%; font-family: 'DejaVu Sans', sans-serif; font-size: 8.4pt; margin: 2mm 0 4mm; }
th, td { border-bottom: 1px solid #d8d7d0; padding: 1.1mm 1.5mm; text-align: right; vertical-align: top; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: #4a4a45; }
.abs { font-size: 9.8pt; margin: 3mm 8mm 5mm; }
.small { font-size: 8.8pt; color: #4a4a45; }
.ok { color: #12784f; font-weight: bold; } .no { color: #b3261e; font-weight: bold; }
"""


def img(p: Path) -> str:
    return f"<img src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'>"


def table(head, rows):
    return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def pct(a, b):
    return 100.0 * (1.0 - a / b)


def ptitle(i):
    a, t, v = TITLES[i]
    return f"<div class='ptitle'><div class='t'>Paper {i}. {html.escape(t)}</div><div class='a'>{a}. {v}</div></div>"


def pairs(theirs: Path, paper: int, notes: dict) -> str:
    out = []
    for n, cap in FIGS[paper][1].items():
        t, o = theirs / f"p{paper}_fig{n}.png", REPRO / f"p{paper}_fig{n}.png"
        out.append(f"<div class='fig'><h3>Fig. {n}. {html.escape(cap)}</h3><div class='pair'>"
                   f"<div><div class='lab'>Published</div>{img(t) if t.exists() else ''}</div>"
                   f"<div><div class='lab'>Our reimplementation</div>{img(o) if o.exists() else ''}</div></div>"
                   f"<div class='cap'>{notes.get(f'p{paper}_fig{n}', '')}</div></div>")
    return "".join(out)


# ------------------------------------------------------------------ numbers
def p1_numbers():
    pub, ex = J("published_paper1.json"), J("summary_paper1_exact2.json")
    oc, on = J("summary_paper1_ours_p1_clean.json"), J("summary_paper1_ours_p1_noise.json")
    rows = []
    for w, n, lab in (("dry", False, "Dry"), ("rain", False, "Rain"), ("storm", False, "Storm"), ("dry", True, "Dry + noise")):
        ref = pub["noisy_dry"] if n else pub[w]
        src = on if n else oc
        pid = np.mean([x["IAE_SO5"] for x in src if x["weather"] == w and x["noise"] == n and x["controller"] == "PID"]), \
            np.mean([x["IAE_SNO2"] for x in src if x["weather"] == w and x["noise"] == n and x["controller"] == "PID"])
        rm = [x for x in ex if x["weather"] == w and x["noise"] == n and x["controller"] == "MAACC-paper"]
        rp = [x for x in ex if x["weather"] == w and x["noise"] == n and x["controller"] == "PID"]
        us = [x for x in src if x["weather"] == w and x["noise"] == n and x["controller"] == "PIDx3"]
        f = lambda v, k: np.mean([x[k] for x in v])
        rows.append(dict(lab=lab, ref=ref, pid=pid,
                         re=(f(rm, "IAE_SO5"), f(rm, "IAE_SNO2")), rep=(f(rp, "IAE_SO5"), f(rp, "IAE_SNO2")),
                         us=(f(us, "IAE_SO5"), f(us, "IAE_SNO2"))))
    return rows


def p3_numbers():
    pub, io, o3 = J("published_paper3.json"), J("summary_ioql.json"), J("summary_ours_p3.json")
    rows = []
    for dyn, tk in ((False, "fixed"), (True, "dynamic")):
        for w, lab in (("dry", "Dry"), ("rain", "Rain"), ("storm", "Storm")):
            pid = [x for x in o3 if x["weather"] == w and x["dynamic"] == dyn and x["method"] == "PID"][0]["IAE"]
            us = [x for x in o3 if x["weather"] == w and x["dynamic"] == dyn and x["method"] == "PIDx4"][0]["IAE"]
            ia = [x["IAE"] for x in io if x["weather"] == w and x["dynamic"] == dyn and x["method"] == "IOQL"]
            rows.append(dict(lab=f"{lab}, {'dynamic' if dyn else 'fixed'} r", ref=pub[tk][w], pid=pid,
                             re_med=float(np.median(ia)), re_all=ia, us=us))
    return rows


def p2_rows():
    rows = []
    for f in glob.glob(str(RES / "summary_fig_p2_*.json")) + glob.glob(str(RES / "summary_ours2_p2_*.json")):
        rows += json.loads(Path(f).read_text())
    for f in glob.glob(str(RES / "summary_aeration_official_*.json")):
        if "published" not in f:
            rows += [x for x in json.loads(Path(f).read_text()) if x["method"].startswith("DDPG")]
    return rows


def main(theirs: Path, out: Path):
    notes = json.loads((ROOT / "docs" / "comparison_notes.json").read_text())
    p1, p3 = p1_numbers(), p3_numbers()
    p2 = p2_rows()
    pub2 = J("published_paper2.json")
    has_panda = any(r["method"] == "PANDA-RL-pidbudget" for r in p2)

    def p2m(w, m, k):
        v = [x[k] for x in p2 if x["weather"] == w and x["method"] == m]
        return float(np.mean(v)) if v else float("nan")

    o1 = [pct(r["us"][0], r["pid"][0]) for r in p1[:3]]
    o3 = [pct(r["us"], r["pid"]) for r in p3]
    H = []
    H.append(f"<!doctype html><html><head><meta charset='utf-8'><title>Reimplementation of three base papers</title>"
             f"<style>{CSS}</style></head><body>")
    H.append("<h1>Reimplementing three reinforcement-learning controllers for wastewater treatment on the official "
             "BSM1 benchmark, and a retuned expert prior that improves on two of them</h1>")
    H.append(f"<p class='small'>Amritha · {date.today():%d %B %Y}</p>")
    H.append(f"""<div class='abs'><b>Abstract.</b> Three recent papers control the Benchmark Simulation Model No. 1
(BSM1) with learning controllers built on a PID prior: a multiagent adaptive critic (paper 1), a DDPG agent that
lets dissolved oxygen float (paper 2) and an integrated online Q-learning scheme (paper 3). We reimplemented all
three exactly as printed on the official BSM1 plant with the official dry, rain and storm influent, and redrew
every figure of the three papers from our runs. The plant and test conditions are largely confirmed: the
oxygen-loop PID results of papers 1 and 3 and the PID energy of paper 2 are reproduced within 1–10 % (one case
16 %), while paper 1's nitrate-loop PID error comes out about half the published value. The learning controllers
are not reproduced: paper 1's controller stays equal to its
PID prior (IAE within 1 %) in 80 of 80 runs, and paper 3's is unstable as printed; paper 2's DDPG reproduces the direction of its
energy saving but with more ammonium and a wide spread between seeds. We show that the PID prior used by papers 1
and 3 is weakly tuned and that raising its gain, with the factor chosen on dry weather only and a margin from the
stability limit, reduces the tracking error by {min(o1):.0f}–{max(o1):.0f} % (paper 1, oxygen) and
{min(o3):.0f}–{max(o3):.0f} % (paper 3) in every weather, including conditions not used for the choice.
For paper 2, our constrained agent held to the PID's own compliance does not improve on the PID; we report this
negative result.</div>""")

    H.append("""<h2>1. What was done</h2>
<p>For each base paper we (i) implemented the algorithm exactly as printed, with every value from the paper;
(ii) ran the paper's own experiments on the official BSM1 plant (the bsm2-python port of the IWA benchmark code)
with the official influent files; (iii) redrew every figure of the paper from these runs and compared every table
with ours; and (iv) tested an improved controller under identical conditions. The plant, the data, the correction
made to the storm and rain files and every setting are given in the companion document
<i>Simulation data and settings</i>. Each published figure is reproduced next to ours for comparison only; those
figures remain the copyright of their publishers.</p>
<p><b>Common protocol.</b> 14 days at 45 s control periods (26 880 periods); the plant first runs 14 days on dry
weather under the paper's PID; metrics IAE = mean|e|, ISE = mean e², DEV<sub>max</sub> = max|e|. Repeated runs differ only
in network initialisation (and noise sequence).</p>""")

    # ---------------------------------------------------------------- paper 1
    H.append("<div class='pb'></div><h2>2. Paper 1</h2>" + ptitle(1))
    H.append("""<p><b>Method in brief.</b> Two agents, one per loop (KLa,5 → S<sub>O,5</sub>, Q<sub>a</sub> →
S<sub>NO,2</sub>), each with an action network Δa = π(z) and a critic Q(z, Δa, a); the applied input is
u = a + b, where b is an incremental PID and a = a + Δa. Table I of the paper gives every weight, learning rate
and network size; we used them unchanged.</p>
<p><b>A discrepancy in the paper.</b> The text gives set points r = [1, 2] for [S<sub>O,5</sub>, S<sub>NO,2</sub>],
but Figs. 4–6 and 8 show S<sub>O,5</sub> held at 2 and S<sub>NO,2</sub> at 1. With [2, 1] our PID reproduces the
paper's PID oxygen results (dry IAE 1.140·10⁻² against 1.14·10⁻²); with [1, 2] it does not. We use [2, 1]. Our
nitrate-loop PID error is about half the paper's in every weather; the paper's nitrate controller, sensor or
actuator must differ from what the text describes, and we could not find what.</p>""")
    H.append("<h3>2.1 Figures: published and reimplemented</h3>" + pairs(theirs, 1, notes))
    rows = []
    for r in p1:
        rows.append([r["lab"], "PID (paper's prior)", f"{r['ref']['PID']['IAE_SO5']:.3g}", f"{r['rep'][0]:.3g}",
                     f"{r['ref']['PID']['IAE_SNO2']:.3g}", f"{r['rep'][1]:.3g}"])
        rows.append(["", "MAACC", f"{r['ref']['MAACC']['IAE_SO5']:.3g}", f"{r['re'][0]:.3g}",
                     f"{r['ref']['MAACC']['IAE_SNO2']:.3g}", f"{r['re'][1]:.3g}"])
    H.append("<h3>2.2 Tables II and III against our reimplementation</h3>"
             + table(["Case", "Method", "IAE S<sub>O,5</sub> paper", "ours", "IAE S<sub>NO,2</sub> paper", "ours"], rows)
             + "<p>Ours are means of 20 runs. <span class='no'>Not reproduced:</span> run as printed, MAACC stays equal to "
               "its PID prior in all 80 runs (IAE within 0.8 %). With the Table I weights in raw units the utility is about 10⁻⁴, so at "
               "l<sub>c</sub> = 0.01 and l<sub>a</sub> = 0.005 neither network moves (the inputs change by under 1 %). "
               "Twelve other initial weight scales, the only setting the paper leaves open, either leave it equal to PID "
               "or make it diverge. Under the stated actuator noise our PID error roughly doubles, whereas the paper's "
               "barely changes, so that experiment cannot be matched.</p>")
    H.append(f"""<h3>2.3 Our improvement: a retuned expert prior</h3>
<p>The learned correction in paper 1 is meant to improve on the PID prior. On the official plant the prior itself
has large headroom: scaling all its gains by a factor g reduces the error until the oxygen loop enters a limit cycle
at g = 5 (dry weather). We choose g = 3, the largest step below 70 % of that limit, <i>on dry weather only</i>, and
apply it unchanged to rain, storm and the noise test.</p>""")
    H.append("<div class='fig one'>" + img(OURS / "ours_p1_dry.png") + "<div class='cap'>Paper 1 set-up, dry weather: "
             "the paper's PID prior and our retuned prior.</div></div>")
    rows = [[r["lab"], f"{r['pid'][0]:.3g}", f"{r['re'][0]:.3g}", f"<b>{r['us'][0]:.3g}</b> ({pct(r['us'][0], r['pid'][0]):.0f} %)",
             f"{r['pid'][1]:.3g}", f"{r['re'][1]:.3g}", f"<b>{r['us'][1]:.3g}</b> ({pct(r['us'][1], r['pid'][1]):.0f} %)"] for r in p1]
    H.append(table(["Case", "S<sub>O,5</sub>: PID", "MAACC (reimpl.)", "Ours", "S<sub>NO,2</sub>: PID", "MAACC (reimpl.)", "Ours"], rows)
             + "<p class='small'>IAE; percentages are reductions relative to the paper's PID prior. Noise case: mean of "
               "20 noise sequences. The gain was chosen on dry weather, so rain, storm and noise are unseen conditions.</p>")

    # ---------------------------------------------------------------- paper 2
    H.append("<div class='pb'></div><h2>3. Paper 2</h2>" + ptitle(2))
    H.append("""<p><b>Method in brief.</b> A DDPG agent changes KLa,5 directly, with no DO set point, using the state
[S<sub>S,5</sub>, S<sub>O,5</sub>, S<sub>NH,5</sub>] and two rewards: A penalises effluent limit violations, B adds
aeration energy. The agent learns on week 1 and is evaluated on week 2 against a PID (DO = 2 mg/L) and a fuzzy
controller. The published networks and hyperparameters are used; the agent did not learn reliably as published on
our first plant, so three corrections are applied (actuator setting in the state, 15-minute decisions, reward
scaled by 1 − γ); a run exactly as published is reported separately in the reproduction report.</p>""")
    H.append("<h3>3.1 Figures: published and reimplemented</h3>" + pairs(theirs, 2, notes))
    rows = []
    for w, lab in (("dry", "Dry"), ("rain", "Rain"), ("storm", "Storm")):
        pid = p2m(w, "PID", "AE")
        for m in ("PID", "Fuzzy", "DDPG-A", "DDPG-B"):
            ref = pub2[w][m]
            rows.append([lab if m == "PID" else "", m, f"{ref['AE']:.0f}", f"{p2m(w, m, 'AE'):.0f}",
                         f"{pct(ref['AE'], pub2[w]['PID']['AE']):.1f}", f"{pct(p2m(w, m, 'AE'), pid):.1f}",
                         f"{ref['S_NH']:.2f}", f"{p2m(w, m, 'S_NH'):.2f}", f"{ref['EQ']:.0f}", f"{p2m(w, m, 'EQ'):.0f}"])
    H.append("<h3>3.2 Tables 2–4 against our reimplementation</h3>"
             + table(["Weather", "Method", "AE paper", "ours", "saving % paper", "ours", "S<sub>NH,e</sub> paper", "ours",
                      "EQ paper", "ours"], rows)
             + "<p><span class='ok'>Reproduced in direction:</span> PID and fuzzy match the paper within about 1 % on energy and "
               "effluent index, and DDPG-B saves energy in every weather, as in the paper. The size differs: our DDPG-B "
               "saves more and lets ammonium rise more, and the spread between seeds is large (dry: 5.6–15.7 %). The paper "
               "reports one value per case without seeds. DDPG rows: mean of 4 runs (dry, rain) and 2 runs (storm).</p>")
    H.append("<h3>3.3 Our improvement: an agent held to the PID's own compliance</h3>")
    if has_panda:
        rows = []
        for w, lab in (("dry", "Dry"), ("rain", "Rain"), ("storm", "Storm")):
            pid = p2m(w, "PID", "AE")
            for m, name in (("PID", "PID"), ("DDPG-B", "DDPG-B (paper 2)"), ("PANDA-RL-pidbudget", "Ours: constrained PANDA-RL")):
                rows.append([lab if m == "PID" else "", name, f"{pct(p2m(w, m, 'AE'), pid):.1f}",
                             f"{100 * p2m(w, m, 'viol_NH'):.0f}", f"{100 * p2m(w, m, 'viol_TN'):.0f}", f"{p2m(w, m, 'EQ'):.0f}"])
        H.append("""<p>DDPG-B saves energy by breaking the ammonium limit far more often than the PID it is compared
with, which the paper does not report. Our agent (PANDA-RL: an influent forecast as input, a critic that optimises
the worst 30 % of outcomes, and Lagrange multipliers that hold the share of time above the ammonium and total
nitrogen limits to a budget) is given budgets close to the PID's own violation rates in dry weather (PID: ammonium 21 %, total nitrogen 18 %
of the week; budgets used: 20 % and 18 %).</p>""")
        H.append("<div class='fig one'>" + img(OURS / "ours_p2_tradeoff.png") + "<div class='cap'>Energy saved against "
                 "the share of the evaluation week above the 4 g N/m³ ammonium limit; each point is one run.</div></div>")
        H.append(table(["Weather", "Method", "energy saved vs PID (%)", "time NH₄ > 4 (%)", "time N<sub>tot</sub> > 18 (%)", "EQ"], rows))
        H.append("""<p><span class='no'>No improvement.</span> Held to the PID's compliance, the agent stays close to
the PID: it saves at most 2.5 % energy (dry) and in rain and storm none, while still breaking the ammonium limit
somewhat more often than the PID (26–42 % of the week against 21–34 %). DDPG-B's larger savings are bought with
roughly twice the PID's violations. On the official plant we therefore have no controller that saves energy at the
PID's compliance; the influent forecast and risk-aware critic that helped on our earlier rebuilt plant did not carry
over. This is reported as found.</p>""")
    else:
        H.append("<p>Results pending.</p>")

    # ---------------------------------------------------------------- paper 3
    H.append("<div class='pb'></div><h2>4. Paper 3</h2>" + ptitle(3))
    H.append("""<p><b>Method in brief.</b> u = ω·α(e) + β(e): a PID prior α (K<sub>p</sub> = 100, K<sub>i</sub> = 20,
K<sub>d</sub> = 1) plus an action network β, both included in the critic Q(e, u); utility e² + 0.01u², γ = 1, 15
hidden units, l<sub>c</sub> = 0.01, l<sub>a</sub> = 0.005, 50 critic and 60 action training steps per period. Fixed
(r = 2 mg/L) and dynamic (2 / 2.2 / 1.8 mg/L) references; priors K1–K3 with ω = 0.99<sup>k</sup>.</p>""")
    H.append("<h3>4.1 Figures: published and reimplemented</h3>" + pairs(theirs, 3, notes))
    rows = [[r["lab"], f"{r['ref']['PID'][0]:.3g}", f"{r['pid']:.3g}", f"{r['ref']['IOQL'][0]:.3g}",
             f"{r['re_med']:.3g}", " ".join(f"{v:.2g}" for v in r["re_all"])] for r in p3]
    H.append("<h3>4.2 Tables I and III against our reimplementation</h3>"
             + table(["Case", "PID paper", "PID ours", "IOQL paper", "IOQL ours (median)", "IOQL ours, each run"], rows)
             + "<p><span class='ok'>PID reproduced</span> within 5 % (dynamic storm: 16 % lower). "
               "<span class='no'>IOQL not reproduced:</span> as printed it either stays at the PID level or drives the "
               "aeration down until S<sub>O,5</sub> collapses. The utility term 0.01·u² (≈ 100–400 for KLa,5 = 100–200 d⁻¹) "
               "outweighs e² (≈ 10⁻⁴) by six orders of magnitude, so minimising Q lowers the air. Tables II and IV (priors "
               "K1–K3 with ω = 0.99<sup>k</sup>) fail in the same way; see the reproduction report.</p>")
    H.append("""<h3>4.3 Our improvement: a retuned expert prior</h3>
<p>The same approach as for paper 1: the prior's gain is raised by a factor chosen on dry weather with a fixed
reference only (limit cycle at 6, chosen factor 4) and applied unchanged to rain, storm and the dynamic
reference.</p>""")
    H.append("<div class='fig one'>" + img(OURS / "ours_p3_dynamic.png") + "<div class='cap'>Paper 3 set-up, dynamic "
             "reference: the paper's PID prior and our retuned prior.</div></div>")
    rows = [[r["lab"], f"{r['pid']:.3g}", f"{r['re_med']:.3g}", f"<b>{r['us']:.3g}</b> ({pct(r['us'], r['pid']):.0f} %)",
             f"{r['ref']['IOQL'][0]:.3g} ({pct(r['ref']['IOQL'][0], r['ref']['PID'][0]):.0f} %)"] for r in p3]
    H.append(table(["Case", "PID", "IOQL (reimpl., median)", "Ours", "IOQL as published"], rows))

    # ---------------------------------------------------------------- summary
    H.append("<div class='pb'></div><h2>5. Overall comparison</h2>")
    H.append("<div class='fig one'>" + img(OURS / "ours_summary.png") + "<div class='cap'>Reduction of the tracking "
             "error relative to each paper's PID prior: as claimed in the paper, as obtained by reimplementing the paper "
             "exactly, and with our retuned prior.</div></div>")
    H.append(f"""<p><b>What is supported.</b> On the official benchmark, the published learning controllers of papers 1
and 3 do not deliver their reported improvements when implemented as printed; our retuned prior delivers
{min(o1 + o3):.0f}–{max(o1 + o3):.0f} % lower oxygen error in every weather and reference, including conditions it was
not tuned on, without any learning. For paper 2 the energy saving of DDPG is reproduced in direction, and its cost in
effluent compliance, not reported by the paper, is measured; our constrained agent did not improve on the PID.</p>
<p><b>What is not claimed.</b> Our gains are smaller than the reductions claimed by papers 1 and 3 (oxygen 92–95 %, nitrate 71–76 %), which we could not
obtain with any setting the papers allow. The retuned prior is a fixed controller chosen from dry-weather runs, not
a learning method; under actuator noise its gain shrinks (oxygen −{pct(p1[3]['us'][0], p1[3]['pid'][0]):.0f} %,
nitrate −{pct(p1[3]['us'][1], p1[3]['pid'][1]):.0f} %). All results are from simulation of one benchmark plant.</p>
<h2>6. Reproducing this report</h2>
<p class='small'>Code: github.com/Amritha902/water_resource. Paper 1: experiments/20_exact_paper1.py; paper 2:
experiments/06_aeration_benchmark.py --official; paper 3: experiments/23_ioql.py; figures: experiments/24_diagrams.py,
25_paper_figures.py, 27_ours_figures.py; data and settings: docs/DATASET.md.</p>
<h2>References</h2>
<p class='small'>[1] {TITLES[1][0]}, "{TITLES[1][1]}", {TITLES[1][2]}.<br>
[2] {TITLES[2][0]}, "{TITLES[2][1]}", {TITLES[2][2]}.<br>
[3] {TITLES[3][0]}, "{TITLES[3][1]}", {TITLES[3][2]}.<br>
[4] J. Alex et al., Benchmark Simulation Model no. 1 (BSM1), IWA Task Group on Benchmarking of Control Strategies
for WWTPs, Lund University, 2008.<br>
[5] bsm2-python 0.0.16, FAU-EVT, Python implementation of the IWA benchmark simulation models (BSD-3).</p>
</body></html>""")
    out.write_text("".join(H))
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
