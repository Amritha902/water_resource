"""Figures for the README: each base paper's published result next to ours.

The paper values are typed in from the papers' tables (``results/
published_paper1.json``; paper 2 in ``18_write_reproduction.PAPER2``); ours
are read from ``results/``.  The paper figures themselves are not copied.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES, FIG = ROOT / "results", ROOT / "docs" / "figures"

spec = importlib.util.spec_from_file_location("rep", ROOT / "experiments" / "18_write_reproduction.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

PAPER, OURS, EXACT = "#eb6834", "#2a78d6", "#1baf7a"   # categorical slots 2, 1, 3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def bars(ax, labels, series, title, ylabel, fmt="{:+.0f}%"):
    """Grouped bars; ``series`` = [(name, values, colour), ...]."""
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for j, (name, vals, col) in enumerate(series):
        off = (j - (n - 1) / 2) * (w + 0.01)
        b = ax.bar(x + off, vals, w, color=col, label=name, zorder=3)
        for rect, v in zip(b, vals):
            y = rect.get_height()
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, y),
                        xytext=(0, 3 if y >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


def paper1_bars():
    pub = json.loads((RES / "published_paper1.json").read_text())
    runs = rep.load_tags("paper1_full")
    exact = rep.load("summary_paper1_exact.json") or []
    pick = lambda c, w, n: [r for r in runs if r["condition"] == c and r["weather"] == w and r["controller"] == n]
    ex = lambda w, n, c: [r for r in exact if r["weather"] == w and r["noise"] == n and r["controller"] == c]
    cases = [("clean", "dry", pub["dry"], "Dry", False), ("clean", "rain", pub["rain"], "Rain", False),
             ("paper_noise", "dry", pub["noisy_dry"], "Dry +\nnoise", True)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, key, name in zip(axes, ("IAE_SO5", "IAE_SNO2"), ("oxygen S_O,5", "nitrate S_NO,2")):
        paper = [rep.pct(r["MAACC"][key], r["PID"][key]) for _, _, r, _, _ in cases]
        printed = [rep.pct(rep.mean(ex(w, n, "MAACC-paper"), key), rep.mean(ex(w, n, "PID"), key))
                   for _, w, _, _, n in cases]
        ours = [rep.pct(rep.mean(pick(c, w, "MAACC-tuned"), key), rep.mean(pick(c, w, "PID"), key))
                for c, w, _, _, _ in cases]
        bars(ax, [l for _, _, _, l, _ in cases],
             [("Paper (published)", paper, PAPER),
              ("Paper as printed, official BSM1 (ours)", printed, EXACT),
              ("Our modified MAACC, rebuilt plant", ours, OURS)],
             f"Paper 1 · {name}", "error reduction vs PID (IAE)")
    axes[0].set_ylim(-240, 125)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.text(0.01, 0.01, "Mean of 20 runs per case. Up = MAACC better than PID. "
             "As printed, MAACC equals PID (0%) or does worse - the published gains are not reproduced.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "readme_paper1_vs_ours.png", dpi=150)


def paper2_bars():
    aer = rep.load("summary_aeration.json") or []
    for extra in ("summary_aeration_rs.json", "summary_aeration_storm_base.json"):
        seen = {(r["weather"], r["method"], r["seed"]) for r in aer}
        aer += [r for r in (rep.load(extra) or []) if (r["weather"], r["method"], r["seed"]) not in seen]
    off = [r for r in rep.load_tags("aeration_official") if "published" not in r.get("method", "")]
    pick = lambda rows, w, m: [r for r in rows if r["weather"] == w and r["method"] == m]
    ws = ("dry", "rain")
    pub = json.loads((RES / "published_paper2.json").read_text())
    paper = [rep.pct(pub[w]["DDPG-B"]["AE"], pub[w]["PID"]["AE"]) for w in ws]
    official = [rep.pct(rep.mean(pick(off, w, "DDPG-B"), "AE"), rep.mean(pick(off, w, "PID"), "AE")) for w in ws]
    rebuilt = [rep.pct(rep.mean(pick(aer, w, "DDPG-B"), "AE"), rep.mean(pick(aer, w, "PID"), "AE")) for w in ws]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars(ax, ["Dry", "Rain"], [("Paper", paper, PAPER),
                               ("Ours, official BSM1 (mean of 3 seeds)", official, EXACT),
                               ("Ours, rebuilt plant (mean of 2 seeds)", rebuilt, OURS)],
         "Paper 2 · DDPG-B aeration energy saved vs PID", "saving (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 14)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "readme_paper2_vs_ours.png", dpi=150)


def paper1_tracking():
    from wwtp.baselines.pid_controller import PIDController
    from wwtp.bsm1 import influent
    from wwtp.envs import run_closed_loop
    from wwtp.maacc.agent import MAACCController, tuned_agent_configs
    s = influent.canonical_scenario("dry")
    pid = run_closed_loop(PIDController(), s, seed=100)
    m = run_closed_loop(MAACCController(configs=tuned_agent_configs(), seed=0), s, seed=100)
    t = pid.time_days
    win = (t >= 8.1) & (t <= 10.1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    for ax, i, name, sp in zip(axes, (0, 1), ("S_O,5 [mg/L]", "S_NO,2 [mg N/L]"), (1.0, 2.0)):
        ax.plot(t[win], pid.state[win, i], color=PAPER, lw=1.4, label="PID (expert prior)")
        ax.plot(t[win], m.state[win, i], color=OURS, lw=1.4, label="MAACC (ours)")
        ax.axhline(sp, color=MUTED, lw=0.8, ls="--")
        ax.set_ylabel(name)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_title("Paper 1 · our modified MAACC vs PID, dry weather, 2 days (rebuilt plant, seed 0)",
                      loc="left", fontsize=10.5, color=INK)
    axes[0].legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    axes[1].set_xlabel("time [days]")
    fig.tight_layout()
    fig.savefig(FIG / "readme_paper1_tracking.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    paper1_bars()
    paper2_bars()
    paper1_tracking()
    print("wrote docs/figures/readme_*.png")
