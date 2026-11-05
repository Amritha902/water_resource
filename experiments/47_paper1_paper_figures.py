"""Two figures for the paper, drawn from stored results only (no number typed by hand).

f0_headline.png  base vs ours: % change of oxygen IAE against the paper's PID, paired over the
                 runs both finished (results/paper1_final_figure_table.json, written by script 45).
f8_inside.png    what the controller does: true oxygen and aeration over two days of the dry test
                 week, paper's PID against tuned PID + ILC (results/traces_paper1_study.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "final"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, GREEN = "#2a78d6", "#118a62"          # validated pair (all six checks pass on SURF)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 9.5, "axes.titlelocation": "left",
})


def paired(S, cond, c, ref="PID"):
    a, b = S[cond][c]["units"], S[cond][ref]["units"]
    common = sorted(set(a) & set(b))
    if not common:
        return None
    return 100 * (np.mean([a[u] for u in common]) / np.mean([b[u] for u in common]) - 1)


def headline():
    S = json.loads((RES / "paper1_final_figure_table.json").read_text())
    conds = [("tw_dry", "Dry"), ("tw_rain", "Rain"), ("tw_storm", "Storm"), ("H4", "New\ninfluent (H4)"),
             ("H6", "Sensors\n(H6)"), ("H7", "Influent +\nsensors (H7)")]
    series = [("MAACC", "Base: MAACC as printed", INK2), ("PIDtuned+ILC", "Ours: tuned PID + ILC", BLUE),
              ("Robust", "Ours: robust variant", GREEN)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w, gap = 0.26, 0.02
    for i, (c, lab, col) in enumerate(series):
        for j, (cond, _) in enumerate(conds):
            x = j + (i - 1) * (w + gap)
            v = paired(S, cond, c)
            if v is None:                       # every run failed: marked, never drawn as a value
                ax.text(x, 3, "✗\nall\nfailed", ha="center", va="bottom", fontsize=7, color=INK2, linespacing=0.95)
                continue
            ax.bar(x, v, w, color=col, zorder=3)
            if c != "MAACC" or abs(v) > 0.5:
                ax.text(x, v + (2 if v >= 0 else -2), f"{v:+.0f}".replace("-", "−"), ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7.5, color=INK)
            else:
                ax.text(x, 2, "0", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK, lw=1.0, zorder=4)
    ax.axvline(3.5, color=AXIS, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.5, 78, "ideal sensors", ha="center", fontsize=8.5, color=INK2)
    ax.text(4.5, 78, "official sensor noise", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(conds)), [l for _, l in conds])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-105, 92)
    ax.set_yticks([-100, -75, -50, -25, 0, 25, 50], ["−100", "−75", "−50", "−25", "0 = paper's PID", "+25", "+50"])
    ax.set_ylabel("oxygen IAE vs paper's PID (%)\n← better        worse →")
    ax.grid(axis="x", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in series]
    ax.legend(h, [l for _, l, _ in series], loc="lower right", fontsize=8, ncol=1, bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    fig.savefig(OUT / "f0_headline.png", dpi=300)
    plt.close(fig)


def inside():
    T = json.loads((RES / "traces_paper1_study.json").read_text())
    a, b = np.array(T["PID|dry|clean"]["s"]), np.array(T["PIDtuned+ILC|dry|clean"]["s"])
    ua, ub = np.array(T["PID|dry|clean"]["u"]), np.array(T["PIDtuned+ILC|dry|clean"]["u"])
    n = len(a)
    t = np.arange(n) * 14.0 / n                     # days
    m = (t >= 9.0) & (t < 11.0)                     # two weekdays of the test week
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True,
                                        gridspec_kw=dict(height_ratios=[1.1, 1, 0.8]))
    ax1.axhline(2.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax1.plot(t[m], a[m, 0], color=INK2, lw=1.4, zorder=3)
    ax1.plot(t[m], b[m, 0], color=BLUE, lw=1.6, zorder=4)
    ax1.set_ylabel("true $S_{O,5}$ (mg/L)")
    ax1.set_title("A. Oxygen in tank 5 (set point 2 mg/L)")
    lo, hi = a[m, 0].min(), a[m, 0].max()
    pad = 0.15 * (hi - lo)
    ax1.set_ylim(lo - pad, hi + 2 * pad)
    ax2.plot(t[m], ub[m, 0], color=BLUE, lw=2.2)
    ax2.plot(t[m], ua[m, 0], color=INK2, lw=1.1, ls=(0, (4, 2)))
    ax2.set_ylabel("$K_La_{,5}$ (1/d)")
    ax2.set_title("B. Aeration: almost the same for both (PID dashed)")
    dd = ub[m, 0] - ua[m, 0]
    ax3.axhline(0, color=INK, lw=0.8)
    ax3.plot(t[m], dd, color=BLUE, lw=1.4)
    ax3.set_ylabel("difference (1/d)")
    ax3.set_title(f"C. Tuned PID + ILC minus paper's PID: at most {np.abs(dd).max():.0f} 1/d, "
                  "timed with the daily load")
    ax3.set_xlabel("time (days, dry test week)")
    ax3.set_xlim(9.0, 11.0)
    ea = np.mean(np.abs(a[m, 0] - 2)); eb = np.mean(np.abs(b[m, 0] - 2))
    ax1.legend([plt.Line2D([], [], color=INK2, lw=1.6), plt.Line2D([], [], color=BLUE, lw=1.6)],
               [f"paper's PID (mean |e| {ea:.4f})", f"tuned PID + ILC (mean |e| {eb:.4f})"],
               loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "f8_inside.png", dpi=300)
    plt.close(fig)
    print(f"days 9-11 mean |e|: PID {ea:.5f}, tuned PID + ILC {eb:.5f}")


if __name__ == "__main__":
    headline()
    inside()
