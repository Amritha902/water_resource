"""Diagrams for the paper 1 study document (docs/figures/study/).

    python experiments/34_paper1_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "figures" / "study"
INK, SOFT = "#1b2a33", "#6b7b83"
C = {"plant": "#d9eef2", "pid": "#fde3d4", "ilc": "#d6f2e4", "sac": "#dbe8fa", "gate": "#fbf1cc", "data": "#eeeeea"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, text="", style="-|>", color=INK, rad=0.0, fs=7.5, off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=1.1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", fontsize=fs, color=SOFT)


def controller_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    box(ax, 8.6, 2.0, 2.1, 1.4, "BSM1 plant\n(official)\n5 tanks + settler", C["plant"], 9, True)
    box(ax, 0.2, 4.1, 1.9, 0.9, "set points\nS_O,5 = 2, S_NO,2 = 1", C["data"], 7.5)
    box(ax, 2.6, 3.9, 2.0, 1.2, "1  PID ×3\n(paper's PID,\ngains ×3)", C["pid"], 8.5, True)
    box(ax, 2.6, 2.1, 2.0, 1.3, "2  Daily ILC\nlearns the PID's daily\ncorrection, applies\nit earlier", C["ilc"], 8, True)
    box(ax, 2.6, 0.3, 2.0, 1.3, "3  SAC-IL actor\nimitation + soft\nactor-critic; extra\naeration correction", C["sac"], 8, True)
    box(ax, 5.2, 0.3, 1.9, 1.3, "4  Safety gate\nclip inputs; off if out\nof range or noise\nneeds more", C["gate"], 7.5, True)
    ax.add_patch(plt.Circle((7.6, 2.7), 0.28, fc="white", ec=INK, lw=1.1))
    ax.text(7.6, 2.7, "Σ", ha="center", va="center", fontsize=12)
    arrow(ax, (2.1, 4.55), (2.6, 4.5), "")
    arrow(ax, (4.6, 4.5), (7.45, 2.95), "Δu (both loops)", rad=-0.1, off=(0.3, 0.25))
    arrow(ax, (4.6, 2.75), (7.32, 2.7), "f(time of day)", off=(0, 0.12))
    arrow(ax, (4.6, 0.95), (5.2, 0.95), "")
    arrow(ax, (7.1, 0.95), (7.55, 2.42), "", rad=0.15)
    ax.text(7.75, 1.45, "KLa,5\ncorrection", fontsize=7.5, color=SOFT)
    arrow(ax, (7.88, 2.7), (8.6, 2.7), "KLa,5, Q_a", off=(0, 0.15))
    ax.plot([9.65, 9.65, 4.85], [2.0, 0.08, 0.08], color=SOFT, lw=1.1)
    arrow(ax, (4.85, 0.08), (3.6, 0.3), "", color=SOFT, rad=0.0)
    ax.text(7.3, 0.14, "measurements S_O,5, S_NO,2 (+ influent flow for SAC-IL)", fontsize=7.3, color=SOFT, ha="center")
    arrow(ax, (9.65, 3.4), (3.6, 5.1), "", rad=0.25, color=SOFT)
    arrow(ax, (3.6, 3.9), (3.6, 3.4), "", style="-|>", color="#1baf7a")
    ax.text(3.75, 3.62, "PID input,\nby time of day", fontsize=6.8, color="#1baf7a")
    ax.text(0.2, 2.9, "Learning:\n2  online, every day\n    of operation\n3  offline, dry week 1:\n    imitate the expert,\n    then SAC", fontsize=7.8,
            color=INK, va="center")
    ax.set_title("Our controller for paper 1's task", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "our_controller.png", dpi=170)
    plt.close(fig)


def data_split():
    from wwtp.official.plant import load_influent
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.8), sharex=True)
    for ax, w in zip(axes, ("dry", "rain", "storm")):
        d = load_influent(w)
        ax.plot(d[:, 0], d[:, 15] / 1000, color=INK, lw=0.8)
        ax.axvspan(0, 7, color="#d6f2e4" if w == "dry" else "#f1f0ec", zorder=0)
        ax.axvspan(7, 14, color="#dbe8fa", zorder=0)
        ax.set_ylabel(f"{w}\nQ (1000 m³/d)", fontsize=8)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 62)
        ax.spines[["top", "right"]].set_visible(False)
        if w == "dry":
            ax.text(3.5, 52, "TUNING + TRAINING (dry week 1)", ha="center", fontsize=8, fontweight="bold", color="#127a50")
        else:
            ax.text(3.5, 52, "not used for tuning or training", ha="center", fontsize=7.5, color=SOFT)
        ax.text(13.0, 52, "TEST (week 2)", ha="center", fontsize=8, fontweight="bold", color="#1f5fb0")
    axes[-1].set_xlabel("day of the official influent file")
    fig.suptitle("Same data as paper 1 (official BSM1 influent), split into tuning/training and test",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "data_split.png", dpi=170)
    plt.close(fig)


def training_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    steps = [("A  Tune on dry week 1", "PID gain sweep ×1..×6\n→ ×3 (70 % of the\nunstable ×5)", C["pid"]),
             ("B  Tune the ILC", "60 settings on dry\nweek 1 → best;\nmap kept", C["ilc"]),
             ("C  Expert data", "PID ×3 + ILC run on\ndry week 1, every\nstep recorded", C["data"]),
             ("D  Imitation", "actor copies the\nexpert (behaviour\ncloning)", C["sac"]),
             ("E  SAC", "5 episodes on dry\nweek 1; imitation\nterm kept ≥ floor", C["sac"]),
             ("F  Test", "dry, rain, storm,\nnoise; week 2 =\nunseen", C["gate"])]
    for i, (t, d, c) in enumerate(steps):
        x = 0.15 + i * 1.8
        box(ax, x, 0.35, 1.6, 1.55, "", c)
        ax.text(x + 0.8, 1.65, t, ha="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + 0.8, 0.95, d, ha="center", va="center", fontsize=7.3, color=INK)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.6, 1.12), (x + 1.8, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=170)
    plt.close(fig)


def maacc_vs_sacil():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax in axes:
        ax.set_xlim(0, 5.4)
        ax.set_ylim(0, 3.2)
        ax.axis("off")
    a = axes[0]
    a.set_title("Paper 1: MAACC (adaptive critic)", loc="left", fontsize=10)
    box(a, 0.1, 2.0, 1.5, 0.8, "PID prior b", C["pid"])
    box(a, 0.1, 0.6, 1.5, 0.9, "action net\nΔa (on-line)", C["sac"])
    box(a, 2.1, 0.6, 1.5, 0.9, "critic Q\n(one step, HDP)", C["sac"])
    box(a, 3.9, 1.3, 1.3, 0.9, "u = a + b", C["data"])
    arrow(a, (1.6, 2.4), (3.9, 1.95))
    arrow(a, (0.85, 1.5), (3.9, 1.6), rad=-0.35)
    arrow(a, (2.1, 1.05), (1.6, 1.05), "")
    a.text(0.1, 0.1, "learns on-line from each step only; gradient through the critic;\nsensitive to input scaling (we found: does not move)", fontsize=7.3, color=SOFT)
    b = axes[1]
    b.set_title("Ours: SAC with imitation learning", loc="left", fontsize=10)
    box(b, 0.1, 2.0, 1.5, 0.8, "expert data\n(PID ×3 + ILC)", C["data"])
    box(b, 0.1, 0.6, 1.5, 0.9, "actor π\n(Gaussian, tanh)", C["sac"])
    box(b, 2.1, 0.6, 1.5, 0.9, "twin critics\nQ1, Q2 + entropy", C["sac"])
    box(b, 2.1, 2.0, 1.5, 0.8, "replay buffer\n(expert + own)", C["data"])
    box(b, 3.9, 1.3, 1.3, 0.9, "u = expert\n+ correction", C["data"])
    arrow(b, (0.85, 2.0), (0.85, 1.5), "imitate", off=(0.35, 0))
    arrow(b, (1.6, 2.4), (2.1, 2.4))
    arrow(b, (2.85, 2.0), (2.85, 1.5))
    arrow(b, (2.1, 1.05), (1.6, 1.05))
    arrow(b, (0.85, 0.6), (4.55, 1.3), rad=0.45)
    b.text(0.1, 0.1, "re-uses all past data (off-policy); starts from the expert, not from\nrandom actions; entropy keeps exploration small and smooth", fontsize=7.3, color=SOFT)
    fig.tight_layout()
    fig.savefig(OUT / "maacc_vs_sacil.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    controller_diagram()
    data_split()
    training_pipeline()
    maacc_vs_sacil()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
