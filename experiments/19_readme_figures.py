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

PAPER, OURS = "#eb6834", "#2a78d6"          # categorical slots 2 and 1
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def bars(ax, labels, paper, ours, title, ylabel):
    x = np.arange(len(labels))
    w = 0.36
    for off, vals, col, name in ((-w / 2 - 0.01, paper, PAPER, "Paper"),
                                 (w / 2 + 0.01, ours, OURS, "Ours")):
        b = ax.bar(x + off, vals, w, color=col, label=name, zorder=3)
        for rect, v in zip(b, vals):
            y = rect.get_height()
            ax.annotate(f"{v:+.0f}%" if title.startswith("Paper 1") or "PID" in ylabel else f"{v:.1f}%",
                        (rect.get_x() + rect.get_width() / 2, y),
                        xytext=(0, 3 if y >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


def paper1_bars():
    pub = json.loads((RES / "published_paper1.json").read_text())
    runs = rep.load_tags("paper1_full")
    pick = lambda c, w, n: [r for r in runs if r["condition"] == c and r["weather"] == w and r["controller"] == n]
    cases = [("clean", "dry", pub["dry"], "Dry"), ("clean", "rain", pub["rain"], "Rain"),
             ("clean", "storm", pub["storm"], "Storm"), ("paper_noise", "dry", pub["noisy_dry"], "Dry +\nnoise")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, name in zip(axes, ("IAE_SO5", "IAE_SNO2"), ("oxygen S_O,5", "nitrate S_NO,2")):
        paper = [rep.pct(r["MAACC"][key], r["PID"][key]) for _, _, r, _ in cases]
        ours = [rep.pct(rep.mean(pick(c, w, "MAACC-tuned"), key), rep.mean(pick(c, w, "PID"), key))
                for c, w, _, _ in cases]
        bars(ax, [l for *_, l in cases], paper, ours,
             f"Paper 1 · {name}", "error reduction vs PID (IAE)")
    axes[0].set_ylim(-230, 120)
    axes[0].legend(frameon=False, loc="lower left")
    fig.text(0.01, 0.01, "Mean of 20 runs per case. Up = MAACC better than PID. "
             "Our noise case is worse than PID (not reproduced).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "readme_paper1_vs_ours.png", dpi=150)


def paper2_bars():
    aer = rep.load("summary_aeration.json") or []
    for extra in ("summary_aeration_rs.json", "summary_aeration_storm_base.json"):
        seen = {(r["weather"], r["method"], r["seed"]) for r in aer}
        aer += [r for r in (rep.load(extra) or []) if (r["weather"], r["method"], r["seed"]) not in seen]
    pick = lambda w, m: [r for r in aer if r["weather"] == w and r["method"] == m]
    ws = ("dry", "rain", "storm")
    paper = [rep.PAPER2["DDPG-B_saving"][w] for w in ws]
    ours = [rep.pct(rep.mean(pick(w, "DDPG-B"), "AE"), rep.mean(pick(w, "PID"), "AE")) for w in ws]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars(ax, ["Dry", "Rain", "Storm"], paper, ours, "Paper 2 · aeration energy saved vs PID", "saving (%)")
    ax.set_ylim(0, 9)
    ax.legend(frameon=False, loc="upper right")
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
    axes[0].set_title("Paper 1 · set-point tracking, dry weather, 2 days (our run, seed 0)",
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
