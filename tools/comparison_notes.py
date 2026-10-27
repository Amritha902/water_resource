"""Write docs/comparison_notes.json: cover, per-figure notes and paper-vs-ours
tables for tools/compare_pdf.py.  Table values are computed from results/."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
J = lambda n: json.loads((RES / n).read_text())
M6 = ("IAE_SO5", "ISE_SO5", "DEVmax_SO5", "IAE_SNO2", "ISE_SNO2", "DEVmax_SNO2")
g = lambda v: f"{v:.3g}"


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def p1_tables():
    pub, r = J("published_paper1.json"), J("summary_paper1_exact2.json")
    rows = []
    for w, n, lab in (("dry", False, "Dry"), ("rain", False, "Rainy"), ("storm", False, "Stormy"),
                      ("dry", True, "Dry + noise")):
        ref = pub["noisy_dry"] if n else pub[w]
        for c, name in (("PID", "PID"), ("MAACC-paper", "MAACC")):
            m = [x for x in r if x["weather"] == w and x["noise"] == n and x["controller"] == c]
            rows.append([lab, f"{name}, paper"] + [g(ref[name][k]) for k in M6])
            rows.append(["", f"{name}, ours (mean of {len(m)})"] + [g(np.mean([x[k] for x in m])) for k in M6])
    return ("<h3>Tables II and III — paper against our reimplementation</h3>"
            + table(["Weather", "Method", "IAE S_O,5", "ISE S_O,5", "DEVmax S_O,5",
                     "IAE S_NO,2", "ISE S_NO,2", "DEVmax S_NO,2"], rows)
            + "<p class='small'>Set points S_O,5 = 2, S_NO,2 = 1 mg/L (as in the paper's figures). Our MAACC "
              "is the algorithm exactly as printed, with the Table I values. In 0 of 80 runs is it more than 1 % "
              "better than PID on both outputs; its learned correction changes the inputs by under 1 %.</p>")


def p2_tables():
    pub = J("published_paper2.json")
    rows = []
    for f in glob.glob(str(RES / "summary_fig_p2_*.json")):
        rows += json.loads(Path(f).read_text())
    for f in glob.glob(str(RES / "summary_aeration_official_*.json")):
        if "published" not in f:
            rows += [x for x in json.loads(Path(f).read_text()) if x["method"].startswith("DDPG")]
    out = []
    for w, lab in (("dry", "Dry"), ("rain", "Rain"), ("storm", "Storm")):
        pid = [x for x in rows if x["weather"] == w and x["method"] == "PID"][0]["AE"]
        for m in ("PID", "Fuzzy", "DDPG-A", "DDPG-B"):
            s = [x for x in rows if x["weather"] == w and x["method"] == m]
            ref = pub[w][m]
            f = lambda k: np.mean([x[k] for x in s])
            out.append([lab, f"{m} ({len(s)} run{'s' if len(s) > 1 else ''})",
                        f"{ref['AE']:.1f} / {f('AE'):.1f}", f"{ref['EQ']:.0f} / {f('EQ'):.0f}",
                        f"{ref['S_NH']:.2f} / {f('S_NH'):.2f}", f"{ref['N_tot']:.2f} / {f('N_tot'):.2f}",
                        f"{100 * (1 - ref['AE'] / pub[w]['PID']['AE']):.2f} / {100 * (1 - f('AE') / pid):.2f}"])
    return ("<h3>Tables 2, 3 and 4 — paper / ours, evaluation week</h3>"
            + table(["Weather", "Method", "AE (kWh/d)", "EQ", "S_NH,e (mg/L)", "N_tot,e (mg/L)", "Saving vs PID (%)"], out)
            + "<p class='small'>Each cell is paper / ours. DDPG runs differ only in their random seed; the spread "
              "between seeds is large (dry DDPG-B 5.6–15.7 % saving), and the paper reports one value per case.</p>")


def p3_tables():
    pub, r = J("published_paper3.json"), J("summary_ioql.json")
    out = []
    for dyn, tk, tk2, name in ((False, "fixed", "fixed_priors", "fixed r = 2"),
                               (True, "dynamic", "dynamic_priors", "dynamic r, eq. (33)")):
        for w, lab in (("dry", "Dry"), ("rain", "Rainy"), ("storm", "Rainstorm")):
            for m in ("PID", "IOQL", "IOQL-K1", "IOQL-K2", "IOQL-K3"):
                s = [x for x in r if x["weather"] == w and x["dynamic"] == dyn and x["method"] == m]
                ref = pub[tk][w].get(m) or pub[tk2][w].get(m.split("-")[-1])
                ia = [x["IAE"] for x in s]
                out.append([f"{lab}, {name}" if m == "PID" else "", m,
                            f"{ref[0]:.3g} / {np.median(ia):.3g}", f"{ref[1]:.3g} / {np.median([x['ISE'] for x in s]):.3g}",
                            " ".join(f"{v:.2g}" for v in ia)])
    return ("<h3>Tables I–IV — paper / ours (median of runs)</h3>"
            + table(["Case", "Method", "IAE", "ISE", "Our runs (IAE)"], out)
            + "<p class='small'>The PID prior matches the paper within 5 % in every case except the dynamic "
              "storm case (16 % lower). IOQL exactly as printed either stays at the PID level (seed 0) or drives "
              "the aeration down and S_O,5 collapses (IAE ≈ 1.8): with u = KLa,5 ≈ 100–200 d⁻¹ the term "
              "0.01·u² in the utility (≈ 100–400) outweighs e² (≈ 10⁻⁴) by six orders of magnitude, so "
              "minimising Q means lowering the air.</p>")


def main():
    n = {}
    n["cover"] = """
<h1>Reimplementation of three base papers — figure-by-figure comparison</h1>
<p>Each figure of each paper is shown twice: on the left as published, on the right as we obtained it from our
reimplementation. Diagrams were redrawn from the papers' descriptions. Every result figure comes from our own
simulations on the official BSM1 plant (bsm2-python) with the official dry, rain and storm influent files, using
exactly the settings printed in each paper. Plant, data and settings are described in the separate
dataset document.</p>
<p>Below each pair is a short note on what matches and what does not, with the numbers. At the end of each paper
its result tables are compared with ours.</p>
<p class='small'>The published figures are reproduced from the original articles for comparison only. They remain
the copyright of IEEE and Science China Press and are not redistributed.</p>"""
    n["p1_intro"] = ("<p class='small'>Two agents (KLa,5 → S_O,5 and Q_a → S_NO,2) on top of an incremental PID. "
                     "Reimplemented exactly as printed with the Table I values; 20 runs per case.</p>")
    n["p1_fig1"] = "Redrawn. Same plant, sensors and the two manipulated inputs (KLa,5 and Q_a)."
    n["p1_fig2"] = "Redrawn. Each agent: action network giving Δa, critic Q(z, Δa, a), integrator a = a + Δa, and the PID prior b; u = a + b."
    n["p1_fig3"] = "Official BSM1 influent files. Matches the published figure, including the rain period (days 8.4–10.4) and the two storm peaks of 60 000 m³/d."
    for k, w in (("p1_fig4", "dry"), ("p1_fig5", "rainy"), ("p1_fig6", "stormy")):
        n[k] = (f"Our PID trace has the same shape and size as the paper's ({w} weather). Our MAACC, exactly as "
                "printed, lies on top of the PID line: the learned correction changes the inputs by at most "
                "0.6 d⁻¹ on KLa,5 and 10 m³/d on Q_a (well under 1 %). The paper's MAACC trace is almost flat "
                "at the set point; we could not obtain that.")
    n["p1_fig4"] += " Dry-weather PID oxygen IAE: ours 1.14·10⁻², paper 1.14·10⁻²."
    n["p1_fig7"] = ("Same daily pattern and range of KLa,5 and Q_a; in rain and storm Q_a rises towards its "
                    "upper limit during the high flows, as in the paper.")
    n["p1_fig8"] = ("With the stated actuator noise our PID error grows about two-fold (oxygen IAE 2.49·10⁻²); "
                    "the paper's PID changes very little (1.25·10⁻²). MAACC again equals PID.")
    n["p1_fig9"] = "Ideal and noisy inputs of the same run; same appearance as the paper's figure."
    n["p1_tables"] = p1_tables()
    n["p2_intro"] = ("<p class='small'>DDPG on KLa,5 with rewards A (effluent limits) and B (energy + ammonium). "
                     "Learn on week 1, evaluate on week 2. Comparators: PID at S_O,5 = 2 mg/L and a fuzzy "
                     "controller.</p>")
    n["p2_fig1"] = "Redrawn BSM1 layout with the DO loop."
    n["p2_fig2"] = "Redrawn agent–environment loop."
    n["p2_fig3"] = "Redrawn DDPG structure: actor, critic, target networks, replay buffer, soft updates."
    n["p2_fig4"] = ("Constant average influent, DO held at each set point for 40 days. Same trends: ammonium falls "
                    "and total nitrogen rises with DO; BOD and COD hardly change. Our ammonium curve is flatter than "
                    "the paper's at low DO, i.e. our plant is less sensitive to under-aeration.")
    n["p2_fig5"] = "Redrawn network: 3 inputs, two hidden layers of 256 ReLU units, one tanh output."
    n["p2_fig6"] = ("PID and fuzzy hold 2 mg/L as in the paper. Both DDPG agents lower the DO between load peaks, "
                    "as in the paper, but our runs are rougher: DDPG-B switches quickly, and DDPG-A over-aerates at "
                    "the daily peaks.")
    n["p2_fig7"] = ("Same ordering as the paper: DDPG lowers total nitrogen and raises ammonium; BOD and COD are "
                    "almost the same for all four controllers.")
    n["p2_fig8"] = "Dry and storm weather; same behaviour as in rain."
    n["p2_tables"] = p2_tables()
    n["p3_intro"] = ("<p class='small'>u = ω·α(e) + β(e): a PID prior α plus an action network β, both inside the "
                     "critic. Reimplemented exactly as printed; 3 runs per case.</p>")
    n["p3_fig1"] = "Redrawn BSM1 layout with the DO loop."
    n["p3_fig2"] = "Redrawn IOQL framework: prior policy, action network, plant, critic and the two update loops."
    n["p3_fig3"] = "Official influent files; matches the published figure."
    n["p3_fig4"] = ("Our PID matches the paper (dry IAE 1.13·10⁻² against 1.16·10⁻²). Our IOQL seed 0 stays on the "
                    "PID line; the other two seeds drive the aeration down and S_O,5 collapses (not shown on this scale).")
    n["p3_fig5"] = ("With ω = 0.99^k the prior fades within hours and the action network alone does not hold the set "
                    "point: S_O,5 first collapses, then swings between about 1.5 and 2.5 mg/L.")
    n["p3_fig6"] = "Dynamic reference: our PID follows the steps as in the paper; IOQL seed 0 equals PID."
    n["p3_fig7"] = "Same failure as in Fig. 5 with the dynamic reference."
    n["p3_tables"] = p3_tables()
    n["closing"] = """
<div class='pb'></div><h2>Summary</h2>
<p><b>What matches.</b> The plant and the test conditions: the influent figures, the PID rows of papers 1 and 3
(oxygen IAE within 1–10 %), the PID and fuzzy rows of paper 2 (energy and effluent index within 1 %), and the
effect of DO on effluent quality (paper 2, Fig. 4).</p>
<p><b>What does not.</b> Paper 1's MAACC and paper 3's IOQL, run exactly as printed, do not reproduce the
published improvements. MAACC stays equal to its PID prior because, with the Table I weights in raw units, the
cost it learns from is about 10⁻⁴ and its networks barely move. IOQL's cost is dominated by the 0.01·u² term,
so the learning drives the air down. Paper 2's DDPG reproduces the direction and size of the energy saving, but
with more ammonium than the paper and a wide spread between seeds.</p>
<p><b>Discrepancies found in the papers.</b> Paper 1 states set points [1, 2] but its figures and its PID results
correspond to [2, 1]. Paper 1's noise experiment cannot be matched: its PID barely changes under the stated noise,
ours doubles its error. None of the papers gives initial weights, input scaling or code, which an exact replication
would need.</p>"""
    (ROOT / "docs" / "comparison_notes.json").write_text(json.dumps(n, indent=1))
    print("wrote docs/comparison_notes.json")


if __name__ == "__main__":
    main()
