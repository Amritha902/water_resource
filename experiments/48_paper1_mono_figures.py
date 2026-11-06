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


def _tr(T, key):
    s, u = np.array(T[key]["s"]), np.array(T[key]["u"])
    return np.arange(len(s)) * 14.0 / len(s), s, u


def trajectories():
    """Counterpart of Figs. 4-6 of the base paper: S_O,5 and S_NO,2 over 14 days."""
    T = json.loads((ROOT / "results" / "traces_paper1_study.json").read_text())
    fig, axs = plt.subplots(3, 2, figsize=(7.16, 5.6), sharex=True)
    for r, (w, name) in enumerate((("dry", "Dry"), ("rain", "Rain"), ("storm", "Storm"))):
        t, sp, _ = _tr(T, f"PID|{w}|clean")
        _, so, _ = _tr(T, f"PIDtuned+ILC|{w}|clean")
        for c, (ref, lo, hi, lab) in enumerate(((2.0, 1.8, 2.1, r"$S_{O,5}$ (mg/L)"), (1.0, 0.4, 1.3, r"$S_{NO,2}$ (mg/L)"))):
            ax = axs[r, c]
            ax.plot(t, sp[:, c], color="#9a9a9a", lw=0.6, label="PID prior")
            ax.plot(t, so[:, c], color="black", lw=0.6, label="Proposed")
            ax.axhline(ref, color="black", lw=0.6, ls=(0, (4, 3)), label="Set point")
            ax.set_ylim(lo, hi)
            ax.set_xlim(0, 14)
            ax.set_ylabel(lab)
            ax.text(0.01, 0.95, f"({'abcdef'[2 * r + c]}) {name}", transform=ax.transAxes, va="top", fontsize=8.5)
    for ax in axs[-1]:
        ax.set_xlabel("Time (d)")
        ax.set_xticks(range(0, 15, 2))
    axs[0, 1].legend(loc="upper right", fontsize=7.5, ncol=3)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "trajectories.png", dpi=400)
    plt.close(fig)


def inputs():
    """Counterpart of Fig. 7: KLa,5 and Q_a of the proposed controller in the three weathers."""
    T = json.loads((ROOT / "results" / "traces_paper1_study.json").read_text())
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
    for w, col, lw in (("storm", "#b0b0b0", 0.7), ("rain", "#6e6e6e", 0.7), ("dry", "black", 0.7)):
        t, _, u = _tr(T, f"PIDtuned+ILC|{w}|clean")
        axs[0].plot(t, u[:, 0], color=col, lw=lw, label=w.capitalize())
        axs[1].plot(t, u[:, 1] / 1e4, color=col, lw=lw)
    axs[0].set_ylabel(r"$K_La_{,5}$ (d$^{-1}$)")
    axs[1].set_ylabel(r"$Q_a$ ($10^4$ m$^3$/d)")
    axs[0].set_ylim(0, 260)
    axs[1].set_ylim(0, 10)
    axs[1].set_xlabel("Time (d)")
    axs[1].set_xlim(0, 14)
    axs[1].set_xticks(range(0, 15, 2))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", fontsize=7.5, ncol=3, bbox_to_anchor=(0.55, 1.0))
    for i, ax in enumerate(axs):
        ax.text(0.01, 0.97, f"({'ab'[i]})", transform=ax.transAxes, va="top")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "inputs.png", dpi=400)
    plt.close(fig)


def noise():
    """Counterpart of Fig. 8: dry weather with actuator noise (std 3 % of the upper bounds)."""
    T = json.loads((ROOT / "results" / "traces_paper1_study.json").read_text())
    t, sp, _ = _tr(T, "PID|dry|noise")
    _, so, _ = _tr(T, "PIDtuned+ILC|dry|noise")
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
    for c, (ref, lo, hi, lab) in enumerate(((2.0, 1.8, 2.2, r"$S_{O,5}$ (mg/L)"), (1.0, 0.6, 1.4, r"$S_{NO,2}$ (mg/L)"))):
        ax = axs[c]
        ax.plot(t, sp[:, c], color="#9a9a9a", lw=0.5, label="PID prior")
        ax.plot(t, so[:, c], color="black", lw=0.5, label="Proposed")
        ax.axhline(ref, color="black", lw=0.6, ls=(0, (4, 3)), label="Set point")
        ax.set_ylim(lo, hi)
        ax.set_ylabel(lab)
        ax.text(0.01, 0.97, f"({'ab'[c]})", transform=ax.transAxes, va="top")
    axs[0].legend(loc="upper right", fontsize=7, ncol=3)
    axs[1].set_xlabel("Time (d)")
    axs[1].set_xlim(0, 14)
    axs[1].set_xticks(range(0, 15, 2))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "noise.png", dpi=400)
    plt.close(fig)


def influent():
    """Counterpart of Fig. 3: influent flow rate of the three BSM1 files used here."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from wwtp.official.plant import load_influent
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for w, col in (("storm", "#b0b0b0"), ("rain", "#6e6e6e"), ("dry", "black")):
        a = load_influent(w)
        ax.plot(a[:, 0], a[:, 15] / 1e4, color=col, lw=0.7, label=w.capitalize())
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=7.5, ncol=3)
    ax.set_ylim(0, 7.5)
    ax.set_xlim(0, 14)
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlabel("Time (d)")
    ax.set_ylabel(r"$Q_{in}$ ($10^4$ m$^3$/d)")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "influent.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    inside()
    training(a.curves)
    trajectories()
    inputs()
    noise()
    influent()
