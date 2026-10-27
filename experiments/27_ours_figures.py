"""Figures for our improved controllers (docs/figures/ours/).

A  paper 1 set-up: PID prior vs our retuned prior (x3), S_O,5 and S_NO,2, dry.
B  paper 3 set-up: PID vs our retuned prior (x4), dynamic reference, 3 weathers.
C  paper 2 set-up: energy saved vs share of the week above the NH4 limit.
D  improvement over PID: paper's claim, our reimplementation, our method.
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "ours"
PID, OURS, REIMP, PAPER = "#eb6834", "#2a78d6", "#1baf7a", "#8a8a84"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
                     "grid.color": "#e4e3de", "axes.spines.top": False, "axes.spines.right": False})
DT4 = 4 * 45 / 86400
J = lambda n: json.loads((RES / n).read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_a():
    t = J("traces_paper1_ours_p1_clean.json")
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    for ax, i, lab, sp, lim in zip(axes, (0, 1), ("S_O,5 (mg/L)", "S_NO,2 (mg/L)"), (2, 1),
                                   ((1.8, 2.1), (0.8, 1.3))):
        for k, c, name in (("dry|clean|PID", PID, "PID prior (paper)"),
                           ("dry|clean|PIDx3", OURS, "Ours: retuned prior")):
            s = np.array(t[k]["s"])
            ax.plot(np.arange(len(s)) * DT4, s[:, i], color=c, lw=0.8, label=name)
        ax.axhline(sp, color="k", lw=0.8, ls=":")
        ax.set_ylabel(lab)
        ax.set_ylim(*lim)
        ax.set_xlim(0, 14)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("Time (day)")
    save(fig, "ours_p1_dry.png")


def fig_b():
    t = J("traces_ours_p3.json")
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True)
    for ax, w, tag in zip(axes, ("dry", "rain", "storm"), "abc"):
        for k, c, name in ((f"{w}|dynamic|PID", PID, "PID prior (paper)"),
                           (f"{w}|dynamic|PIDx4", OURS, "Ours: retuned prior")):
            x = np.array(t[k]["x"])
            ax.plot(np.arange(len(x)) * DT4, x, color=c, lw=0.8, label=name)
        r = np.array(t[f"{w}|dynamic|PID"]["r"])
        ax.plot(np.arange(len(r)) * DT4, r, color="k", lw=0.8, ls=":", label="r_k")
        ax.set_ylabel("S_O,5 (mg/L)")
        ax.set_title(f"({tag}) {w}", loc="left", fontsize=9)
        ax.set_xlim(0, 14)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[2].set_xlabel("Time (day)")
    save(fig, "ours_p3_dynamic.png")


def p2_rows():
    rows = []
    for f in glob.glob(str(RES / "summary_fig_p2_*.json")) + glob.glob(str(RES / "summary_ours2_p2_*.json")):
        rows += json.loads(Path(f).read_text())
    for f in glob.glob(str(RES / "summary_aeration_official_*.json")):
        if "published" not in f:
            rows += [x for x in json.loads(Path(f).read_text()) if x["method"].startswith("DDPG")]
    return rows


def fig_c():
    rows = p2_rows()
    if not any(r["method"] == "PANDA-RL-pidbudget" for r in rows):
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, w in zip(axes, ("dry", "rain", "storm")):
        pid = [x for x in rows if x["weather"] == w and x["method"] == "PID"][0]
        for m, c, mk, name in (("PID", PID, "s", "PID"), ("DDPG-B", REIMP, "o", "DDPG-B (paper 2)"),
                               ("PANDA-RL-pidbudget", OURS, "D", "Ours: constrained PANDA-RL")):
            v = [x for x in rows if x["weather"] == w and x["method"] == m]
            ax.scatter([100 * x["viol_NH"] for x in v], [100 * (1 - x["AE"] / pid["AE"]) for x in v],
                       color=c, marker=mk, s=40, label=name, zorder=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(w, loc="left", fontsize=9)
        ax.set_xlabel("time above 4 g N/m³ (%)")
    axes[0].set_ylabel("aeration energy saved vs PID (%)")
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, "ours_p2_tradeoff.png")


def fig_d():
    p1 = J("published_paper1.json")
    ex = J("summary_paper1_exact2.json")
    o1 = J("summary_paper1_ours_p1_clean.json")
    p3 = J("published_paper3.json")
    io = J("summary_ioql.json")
    o3 = J("summary_ours_p3.json")
    cases, paper, reimp, ours = [], [], [], []
    for w in ("dry", "rain", "storm"):
        for key, lab in (("IAE_SO5", "O₂"), ("IAE_SNO2", "NO₃")):
            pid = [x for x in o1 if x["weather"] == w and x["controller"] == "PID"][0][key]
            cases.append(f"P1 {w}\n{lab}")
            paper.append(100 * (1 - p1[w]["MAACC"][key] / p1[w]["PID"][key]))
            m = np.mean([x[key] for x in ex if x["weather"] == w and not x["noise"] and x["controller"] == "MAACC-paper"])
            pe = [x for x in ex if x["weather"] == w and not x["noise"] and x["controller"] == "PID"][0][key]
            reimp.append(100 * (1 - m / pe))
            ours.append(100 * (1 - [x for x in o1 if x["weather"] == w and x["controller"] == "PIDx3"][0][key] / pid))
    for w in ("dry", "rain", "storm"):
        pid = [x for x in o3 if x["weather"] == w and not x["dynamic"] and x["method"] == "PID"][0]["IAE"]
        cases.append(f"P3 {w}\nO₂")
        paper.append(100 * (1 - p3["fixed"][w]["IOQL"][0] / p3["fixed"][w]["PID"][0]))
        reimp.append(100 * (1 - np.median([x["IAE"] for x in io if x["weather"] == w and not x["dynamic"]
                                           and x["method"] == "IOQL"]) / pid))
        ours.append(100 * (1 - [x for x in o3 if x["weather"] == w and not x["dynamic"] and x["method"] == "PIDx4"][0]["IAE"] / pid))
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for off, vals, c, name in ((-0.27, paper, PAPER, "Paper's claim"), (0, reimp, REIMP, "Paper reimplemented as printed"),
                               (0.27, ours, OURS, "Ours")):
        ax.bar(x + off, np.clip(vals, -100, 100), 0.25, color=c, label=name, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, cases, fontsize=8)
    ax.set_ylabel("IAE reduction vs PID prior (%)")
    ax.set_ylim(-100, 100)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    ax.set_title("bars at −100 % are clipped (paper 3 as printed: median IAE 140–170 times the PID's)", loc="right", fontsize=7, color="#6b6a64")
    save(fig, "ours_summary.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_a(); fig_b(); fig_c(); fig_d()
