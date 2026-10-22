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
    exact = rep.load("summary_paper1_exact.json") or []
    ex = lambda w, n, c: [r for r in exact if r["weather"] == w and r["noise"] == n and r["controller"] == c]
    cases = [("dry", pub["dry"], "Dry", False), ("rain", pub["rain"], "Rain", False),
             ("dry", pub["noisy_dry"], "Dry +\nnoise", True)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, key, name in zip(axes, ("IAE_SO5", "IAE_SNO2"), ("oxygen S_O,5", "nitrate S_NO,2")):
        paper = [rep.pct(r["MAACC"][key], r["PID"][key]) for _, r, _, _ in cases]
        ours = [rep.pct(rep.mean(ex(w, n, "MAACC-paper"), key), rep.mean(ex(w, n, "PID"), key))
                for w, _, _, n in cases]
        bars(ax, [l for _, _, l, _ in cases],
             [("Paper (published)", paper, PAPER), ("Our reimplementation", ours, EXACT)],
             f"Paper 1 · {name}", "error reduction vs PID (IAE)")
    axes[0].set_ylim(-130, 125)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.text(0.01, 0.01, "Official BSM1 plant and influent, algorithm and Table I values as printed, mean of 20 runs. "
             "Up = MAACC better than PID. Reimplemented, MAACC equals PID.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "readme_paper1_vs_ours.png", dpi=150)


def paper2_bars():
    aer = rep.load("summary_aeration.json") or []
    for extra in ("summary_aeration_rs.json", "summary_aeration_storm_base.json"):
        seen = {(r["weather"], r["method"], r["seed"]) for r in aer}
        aer += [r for r in (rep.load(extra) or []) if (r["weather"], r["method"], r["seed"]) not in seen]
    off = rep.load_tags("aeration_official", exclude="published")
    pick = lambda rows, w, m: [r for r in rows if r["weather"] == w and r["method"] == m]
    ws = ("dry", "rain")
    pub = json.loads((RES / "published_paper2.json").read_text())
    paper = [rep.pct(pub[w]["DDPG-B"]["AE"], pub[w]["PID"]["AE"]) for w in ws]
    official = [rep.pct(rep.mean(pick(off, w, "DDPG-B"), "AE"), rep.mean(pick(off, w, "PID"), "AE")) for w in ws]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars(ax, ["Dry", "Rain"], [("Paper (published)", paper, PAPER),
                               ("Our reimplementation (mean of 3 seeds)", official, EXACT)],
         "Paper 2 · DDPG-B aeration energy saved vs PID", "saving (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 14)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "readme_paper2_vs_ours.png", dpi=150)


def paper1_tracking():
    d = np.load(RES / "video_data.npz")
    t = np.arange(len(d["off_pid"])) * 45 / 86400
    win = (t >= 8.0) & (t <= 10.0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    for ax, i, name, sp in zip(axes[:2], (0, 1), ("S_O,5 [mg/L]", "S_NO,2 [mg N/L]"), (1.0, 2.0)):
        ax.plot(t[win], d["off_pid"][win, i], color=PAPER, lw=1.6, label="PID (expert prior)")
        ax.plot(t[win], d["off_exact"][win, i], color=EXACT, lw=1.6, ls="--", label="MAACC, reimplemented as printed")
        ax.axhline(sp, color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel(name)
        ax.grid(axis="y", color=GRID, lw=0.8)
    a = d["off_exact_a"][win]
    axes[2].plot(t[win], a[:, 0], color=EXACT, lw=1.4)
    axes[2].set_ylabel("learned correction\na on KLa_5 [1/d]")
    axes[2].grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_title("Paper 1 reimplemented · official BSM1, dry weather, days 8–10 (seed 0)",
                      loc="left", fontsize=10.5, color=INK)
    axes[0].legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=2, loc="upper right", fontsize=9)
    axes[2].set_xlabel("time [days]")
    fig.tight_layout()
    fig.savefig(FIG / "readme_paper1_tracking.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    paper1_bars()
    paper2_bars()
    paper1_tracking()
    print("wrote docs/figures/readme_*.png")
