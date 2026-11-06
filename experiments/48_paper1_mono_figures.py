"""Black-and-white figures for the journal paper, from result files only.

inside.png    two days of the dry test period: S_O,5 under the PID prior and the proposed
              controller, and the difference in KLa,5 (results/traces_paper1_study.json).
training.png  training-period oxygen IAE per episode, five runs per method, read from
              curves_maacc_rep_paper_seed*.json and curves_sacil_v3_seed*.json in the
              directory given by --curves (a complete retraining).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.7, "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black", "text.color": "black",
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def inside():
    T = json.loads((ROOT / "results" / "traces_paper1_study.json").read_text())
    a, b = np.array(T["PID|dry|clean"]["s"])[:, 0], np.array(T["PIDtuned+ILC|dry|clean"]["s"])[:, 0]
    ua, ub = np.array(T["PID|dry|clean"]["u"])[:, 0], np.array(T["PIDtuned+ILC|dry|clean"]["u"])[:, 0]
    t = np.arange(len(a)) * 14.0 / len(a)
    m = (t >= 9.0) & (t < 11.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax1.plot(t[m], a[m], color="black", lw=0.9, ls=(0, (3, 1.5)), label="PID prior")
    ax1.plot(t[m], b[m], color="black", lw=1.0, label="Proposed")
    ax1.set_ylabel(r"$S_{O,5}$ (mg/L)")
    ax1.set_ylim(1.9, 2.06)
    ax1.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax1.text(0.01, 0.97, "(a)", transform=ax1.transAxes, va="top")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.plot(t[m], ub[m] - ua[m], color="black", lw=0.8)
    ax2.set_ylabel(r"$\Delta K_La_{,5}$ (d$^{-1}$)")
    ax2.set_xlabel("Time (d)")
    ax2.set_xlim(9, 11)
    ax2.text(0.01, 0.97, "(b)", transform=ax2.transAxes, va="top")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "inside.png", dpi=400)
    plt.close(fig)


def training(curves: Path):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    pid = None
    for ax, (tag, title) in zip(axs, (("maacc_rep_paper", "(a) rMAACC"), ("sacil_v3", "(b) SAC-IL"))):
        styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
        for s in range(5):
            d = json.loads((curves / f"curves_{tag}_seed{s}.json").read_text())
            log = d["log"] if isinstance(d, dict) else d
            ep = [x["episode"] for x in log]
            v = [x["IAE_SO5"] for x in log]
            if tag == "maacc_rep_paper":
                pid = v[0]
            ax.plot(ep, v, color="black", lw=0.9, ls=styles[s], marker="o", ms=2.2)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Episode")
        ax.set_xticks(range(0, 6))
        ax.set_yscale("log")
    for ax in axs:
        ax.axhline(pid, color="#7f7f7f", lw=0.8, ls=(0, (4, 2)))
    axs[0].set_ylabel("Oxygen IAE, training")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "training.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    inside()
    training(a.curves)
