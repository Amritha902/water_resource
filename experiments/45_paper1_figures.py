"""Final figures of the paper 1 work, drawn from the result files (docs/figures/final/).

Reads results/summary_paper1_study.json (33), summary_paper1_heldout.json (40),
summary_paper1_heldout2.json (43), their traces, curves_*.json (28, 39) and
paper1_tuned_pid.json (38).  Every number is computed here from the runs.

Colour = family, fixed in every figure: grey = paper 1's controllers, blue =
our controllers without learning, orange = learned controllers.  A run whose
oxygen left 0.3-4 mg/L is a failure: it is never drawn as a value, always
marked with a cross and counted.

    python experiments/45_paper1_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "final"
J = lambda n: json.loads((RES / n).read_text())

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRIT = "#d03b3b"
FAM = {"paper": INK2, "ours": BLUE, "learned": ORANGE}
FAM_LABEL = {"paper": "paper 1's controllers", "ours": "ours, no learning", "learned": "learned controllers"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 9, "axes.titlelocation": "left",
})

# controllers, top to bottom, with family and label
ROWS = [
    ("PID", "paper", "Paper's PID (reference)"),
    ("MAACC", "paper", "MAACC as printed"),
    ("rMAACC", "learned", "Repaired MAACC (on paper's PID)"),
    ("PIDx3+ILC+rMAACC", "learned", "PID ×3 + ILC + repaired MAACC"),
    ("Ours", "learned", "Ours: PID ×3 + ILC + SAC-IL"),
    ("OursT", "learned", "Tuned PID + ILC + SAC-IL"),
    ("PIDx3", "ours", "PID ×3"),
    ("PIDx3+ILC", "ours", "PID ×3 + ILC"),
    ("PIDtuned", "ours", "Tuned PID"),
    ("PIDtuned+ILC", "ours", "Tuned PID + ILC"),
    ("PIDtuned+ILC-AW", "ours", "Tuned PID + ILC, anti-windup"),
    ("Robust", "ours", "Robust (filter, gains ×0.5, ILC-AW)"),
]
FAMOF = {c: f for c, f, _ in ROWS}
LAB = {c: l for c, _, l in ROWS}


def runs_by_condition():
    """condition -> list of runs (dicts with controller, unit, value, fail)."""
    C = {}
    for r in J("summary_paper1_study.json")["runs"]:
        if r["noise"]:
            C.setdefault("act", []).append(dict(c=r["controller"], u=r["noise_seed"], v=r["IAE_SO5"], fail=r["fail"]))
        else:
            C.setdefault(f"tw_{r['weather']}", []).append(dict(c=r["controller"], u=r["weather"], v=r["IAE_SO5_test"],
                                                                fail=r["fail"]))
    for f in ("summary_paper1_heldout.json", "summary_paper1_heldout2.json"):
        for r in J(f)["runs"]:
            C.setdefault(r["condition"], []).append(dict(c=r["controller"], u=f"{r['weather']}|{r['sensors']}",
                                                         v=r["IAE_SO5_test"], fail=r["fail"]))
    return C


COLS = [("tw_dry", "Dry"), ("tw_rain", "Rain"), ("tw_storm", "Storm"), ("act", "Actuator\nnoise"),
        ("H1", "H1\nBSM2 raw"), ("H2", "H2\nsensors"), ("H3", "H3\nsensor\ndelay"),
        ("H4", "H4\nBSM2\nclarified"), ("H5", "H5\nBSM2 raw"), ("H6", "H6\nsensors"), ("H7", "H7\nBSM2 +\nsensors")]
GROUPS = [(0, 4, "Official data, test week (ideal sensors)"), (4, 7, "Held-out test 1"), (7, 11, "Held-out test 2")]


def stats(C):
    """condition -> controller -> mean over finished runs, failures, runs, values per run."""
    S = {}
    for cond, rows in C.items():
        S[cond] = {}
        for c in {r["c"] for r in rows}:
            rr = [r for r in rows if r["c"] == c]
            ok = [r["v"] for r in rr if not r["fail"]]
            units = {}
            for r in rr:
                if not r["fail"]:
                    units.setdefault(r["u"], []).append(r["v"])
            S[cond][c] = dict(mean=float(np.mean(ok)) if ok else np.nan, failed=sum(r["fail"] for r in rr),
                              n=len(rr), values=ok, units={u: float(np.mean(v)) for u, v in units.items()})
    return S


def paired_ratio(S, cond, c, ref="PID"):
    """Mean error of c over the units (window, weather or noise run) where both c and the
    reference finished, divided by the reference's mean over the same units."""
    a, b = S[cond][c]["units"], S[cond][ref]["units"]
    common = sorted(set(a) & set(b))
    if not common:
        return np.nan, 0
    return float(np.mean([a[u] for u in common]) / np.mean([b[u] for u in common])), len(common)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 1: the whole picture
def fig_matrix(S):
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#8f2423"])
    nr, nc = len(ROWS), len(COLS)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=12)
    for i, (c, fam, lab) in enumerate(ROWS):
        for j, (cond, _) in enumerate(COLS):
            d = S.get(cond, {}).get(c)
            x0, y0 = j - 0.47, i - 0.45
            if d is None:
                ax.add_patch(plt.Rectangle((x0, y0), 0.94, 0.9, fc=SURF, ec=GRID, lw=0.6))
                ax.text(j, i, "not run", ha="center", va="center", fontsize=6.5, color=MUTED)
                continue
            if d["failed"] == d["n"]:
                ax.add_patch(plt.Rectangle((x0, y0), 0.94, 0.9, fc="#3b3b38", ec=SURF, lw=1.5))
                ax.text(j, i, "✗ all\nfailed", ha="center", va="center", fontsize=7, color="#ffffff",
                        fontweight="bold", linespacing=0.95)
                continue
            ratio, _ = paired_ratio(S, cond, c)
            z = np.clip(np.log10(ratio), -1.1, 1.1) / 1.1
            fc = cmap(0.5 + 0.5 * z)
            ax.add_patch(plt.Rectangle((x0, y0), 0.94, 0.9, fc=fc, ec=SURF, lw=1.5))
            dark = abs(z) > 0.55
            txt = "ref." if c == "PID" else f"{100 * (ratio - 1):+.0f}%".replace("-", "−")
            if round(100 * (ratio - 1)) == 0 and c != "PID":
                txt = "0%"
            if d["failed"]:
                txt += f"\n✗ {d['failed']}/{d['n']}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.4 if not d["failed"] else 6.6,
                    color="#ffffff" if dark else INK, fontweight="bold" if not d["failed"] else "normal",
                    linespacing=0.95)
    ax.set_yticks(range(nr), [l for _, _, l in ROWS], fontsize=7.8)
    for t, (_, fam, _) in zip(ax.get_yticklabels(), ROWS):
        t.set_color(INK)
    for i, (_, fam, _) in enumerate(ROWS):
        ax.add_patch(plt.Rectangle((-0.60, i - 0.3), 0.07, 0.6, fc=FAM[fam], ec="none", clip_on=False))
    ax.set_xticks(range(nc), [l for _, l in COLS], fontsize=7.3)
    ax.xaxis.tick_top()
    tr = ax.get_xaxis_transform()
    for a, b, name in GROUPS:
        ax.plot([a - 0.45, b - 0.55], [1.165, 1.165], color=INK2, lw=0.8, clip_on=False, transform=tr)
        ax.text((a + b - 1) / 2, 1.18, name, ha="center", va="bottom", fontsize=8, color=INK2, transform=tr)
    for x in (3.5, 6.5):
        ax.axvline(x, color=AXIS, lw=0.8)
    for y in (1.5, 5.5):
        ax.axhline(y, color=AXIS, lw=0.8)
    # colour scale
    cax = fig.add_axes([0.80, 0.045, 0.16, 0.022])
    g = np.linspace(-1, 1, 200)[None, :]
    cax.imshow(g, aspect="auto", cmap=cmap, extent=(-1.1, 1.1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-1, -np.log10(2), 0, np.log10(2), 1], ["−90%", "−50%", "0", "+100%", "+900%"], fontsize=6.5)
    cax.grid(False)
    cax.set_title("oxygen error vs paper's PID", fontsize=7, color=INK2, loc="center")
    handles = [Line2D([], [], marker="s", ls="", ms=7, color=FAM[f], label=FAM_LABEL[f]) for f in FAM]
    handles.append(Line2D([], [], marker="s", ls="", ms=7, color="#3b3b38", label="✗ every run failed"))
    fig.legend(handles=handles, loc="lower left", ncol=4, fontsize=7.3, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Paper 1: oxygen tracking error (IAE of S_O,5) of every controller in every test, against the paper's PID",
                 x=0.01, y=0.99, ha="left", va="top", fontsize=10, color=INK)
    fig.text(0.01, 0.945, "Blue = better than the paper's PID, red = worse. Learned controllers: mean of 5 training runs. "
             "Failures (oxygen left 0.3–4 mg/L) are counted (✗ failed/runs) and left out; each % compares only the\n"
             "windows / runs in which both this controller and the paper's PID finished.", fontsize=7.2, color=INK2,
             va="top")
    fig.subplots_adjust(left=0.25, right=0.995, top=0.74, bottom=0.11)
    save(fig, "f1_overview_matrix.png")


# ---------------------------------------------------------------- dot plots
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

PCT_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10]


def pct_axis(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    t = [x for x in PCT_TICKS if lo <= x <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t))
    ax.set_xticklabels(["0" if x == 1 else f"{100 * (x - 1):+.0f}%".replace("-", "−") for x in t])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def iae_axis(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    cand = [a * 10.0 ** e for e in range(-4, 1) for a in (1, 2, 5)]
    t = [x for x in cand if lo <= x <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t))
    ax.set_xticklabels([f"{x:g}" for x in t])
    ax.xaxis.set_minor_locator(NullLocator())


def dotplot(ax, S, cond, rows, ratio=False, show_labels=True):
    """One row per controller: a dot per run, a diamond for the mean; failures are counted, never drawn.
    ratio=True: each run divided by the paper's PID on the same window / run (the paired view)."""
    ref_units = S[cond]["PID"]["units"]
    pts, means = {}, {}
    for c, _, _ in rows:
        d = S[cond].get(c)
        if d is None:
            continue
        if ratio:
            rr = [x for x in RUNS[cond] if x["c"] == c and not x["fail"] and x["u"] in ref_units]
            pts[c] = [x["v"] / ref_units[x["u"]] for x in rr]
            means[c] = paired_ratio(S, cond, c)[0]
        else:
            pts[c] = d["values"]
            means[c] = d["mean"]
    allv = [v for vs in pts.values() for v in vs] + ([1.0] if ratio else [S[cond]["PID"]["mean"]])
    lo, hi = min(allv) / 1.6, max(allv) * 2.6
    (pct_axis if ratio else iae_axis)(ax, lo, hi)
    ax.axvline(1.0 if ratio else S[cond]["PID"]["mean"], color=INK2, lw=1.0, zorder=1)
    for i, (c, fam, _) in enumerate(rows):
        d = S[cond].get(c)
        if d is None:
            continue
        v = pts.get(c, [])
        if v:
            jit = ((np.arange(len(v)) % 5) - 2) * 0.07
            ax.scatter(v, i + jit, s=11, color=FAM[fam], alpha=0.5, lw=0, zorder=2)
            if means[c] == means[c]:
                ax.scatter([means[c]], [i], s=46, marker="D", color=FAM[fam], edgecolor=SURF, lw=1.2, zorder=3)
        if d["failed"]:
            ax.text(hi / 1.05, i, f"✗ {'all' if d['failed'] == d['n'] else d['failed']}/{d['n']} failed",
                    ha="right", va="center", fontsize=6.4, color=CRIT, fontweight="bold")
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for _, _, l in rows] if show_labels else [], fontsize=7.4)
    if not show_labels:
        ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


def fig_testweek(S):
    rows = ROWS
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
    for k, (ax, (cond, name)) in enumerate(zip(axes, COLS[:3])):
        dotplot(ax, S, cond, rows, show_labels=k == 0)
        ax.set_xlabel("oxygen IAE, days 7–14 (log scale)", fontsize=7.6)
        ax.set_title(f"{name} weather, test week", fontsize=9)
    handles = [Line2D([], [], marker="D", ls="", ms=6, color=FAM[f], label=FAM_LABEL[f]) for f in FAM]
    handles += [Line2D([], [], marker="o", ls="", ms=4, color=MUTED, alpha=0.6, label="one training run"),
                Line2D([], [], color=INK2, lw=1, label="paper's PID")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.4, bbox_to_anchor=(0.6, 0.0))
    fig.suptitle("Official BSM1 data, test week (never used for tuning or training): oxygen tracking error, lower is better",
                 x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.17, wspace=0.08)
    save(fig, "f2_test_week.png")


T1 = ["PID", "MAACC", "rMAACC", "PIDx3+ILC+rMAACC", "Ours", "OursT", "PIDx3", "PIDx3+ILC", "PIDtuned", "PIDtuned+ILC"]
T2 = ["PID", "MAACC", "rMAACC", "Ours", "PIDtuned", "PIDtuned+ILC", "PIDtuned+ILC-AW", "Robust"]


def fig_heldout(S):
    conds = [("H1", "H1  new influent (BSM2), raw"), ("H2", "H2  official sensors"), ("H3", "H3  sensor delay only"),
             ("H4", "H4  new influent, clarified"), ("H5", "H5  new influent, raw"), ("H6", "H6  official sensors"),
             ("H7", "H7  new influent + sensors")]
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[len(T1), len(T2)], hspace=0.42, wspace=0.07)
    for k, (cond, name) in enumerate(conds):
        top = k < 3
        rows = [r for r in ROWS if r[0] in (T1 if top else T2)]
        ax = fig.add_subplot(gs[0 if top else 1, k if top else k - 3])
        dotplot(ax, S, cond, rows, ratio=True, show_labels=k in (0, 3))
        ax.set_title(name, fontsize=8.6)
        ax.set_xlabel("oxygen IAE vs paper's PID, same window / run", fontsize=7.2)
    a = fig.add_subplot(gs[0, 3])
    a.axis("off")
    a.text(0.03, 1.0, "Held-out test 1 (top row): protocol pushed\nbefore the runs; 6 BSM2 windows, or\n"
           "3 weathers × 3 noise seeds.\n\nHeld-out test 2 (bottom row): new windows\nand new noise seeds, after two fixes\n"
           "(ILC anti-windup; robust tuning);\nprotocol pushed before the runs.\n\n"
           "Each dot: one run (window, noise seed,\ntraining run) divided by the paper's PID\non the same window / noise.\n"
           "Diamond: mean over the windows / runs\nboth finished, divided by the PID's.\n"
           "✗: runs in which oxygen left 0.3–4 mg/L;\nnever drawn as a value.",
           va="top", fontsize=7.3, color=INK2, transform=a.transAxes)
    handles = [Line2D([], [], marker="D", ls="", ms=6, color=FAM[f], label=FAM_LABEL[f]) for f in FAM]
    handles.append(Line2D([], [], color=INK2, lw=1, label="paper's PID (0)"))
    a.legend(handles=handles, loc="lower left", fontsize=7.3)
    fig.suptitle("Held-out tests: oxygen tracking error on data and conditions nothing was tuned or trained on "
                 "(left of the line = better than the paper's PID)", x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.995, top=0.93, bottom=0.06)
    save(fig, "f3_heldout.png")


# ---------------------------------------------------------------- trade-offs
def fig_tradeoffs(S):
    study = J("summary_paper1_study.json")["runs"]
    ho2 = J("summary_paper1_heldout2.json")["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    pts = {}
    for c, fam, lab in ROWS:
        rr = [r for r in study if r["controller"] == c and r["weather"] == "dry" and not r["noise"]]
        if not rr or c == "MAACC":
            continue
        pts[c] = (np.mean([r["KLa_move_per_day"] for r in rr]), np.mean([r["IAE_SO5_test"] for r in rr]))
        ax.scatter(*pts[c], s=48, color=FAM[fam], edgecolor=SURF, lw=1.2, zorder=3)
    # label positions (x, y) chosen by hand so that no two labels touch; a hairline leads to each point
    place = {"PID": (560, 0.0125, "Paper's PID"), "PIDx3": (1030, 0.0042, "PID ×3"),
             "Robust": (560, 0.0026, "Robust"), "PIDtuned": (640, 0.00145, "Tuned PID"),
             "PIDx3+ILC": (1080, 0.0024, "PID ×3 + ILC"), "Ours": (1080, 0.00175, "Ours: PID ×3 + ILC + SAC-IL"),
             "PIDtuned+ILC": (600, 0.00088, "Tuned PID + ILC (with or without anti-windup)"),
             "OursT": (940, 0.00078, "Tuned PID + ILC + SAC-IL"),
             "PIDx3+ILC+rMAACC": (1300, 0.00086, "PID ×3 + ILC + repaired MAACC"),
             "rMAACC": (1530, 0.0024, "Repaired MAACC")}
    for c, (x, y, text) in place.items():
        px, py = pts[c]
        ax.annotate(text, (px, py), xytext=(x, y), fontsize=6.9, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=4))
    ax.set_yscale("log")
    ax.set_ylim(0.0007, 0.016)
    ax.set_xlim(420, 1980)
    ax.yaxis.set_major_locator(FixedLocator([0.001, 0.002, 0.005, 0.01]))
    ax.set_yticklabels(["0.001", "0.002", "0.005", "0.01"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("aeration movement, Σ|ΔKLa,5| per day (1/d per day): actuator wear")
    ax.set_ylabel("oxygen IAE, test week, dry (log)")
    ax.set_title("A. Tracking against actuator wear (official data, ideal sensors)")
    ax.text(0.02, 0.03, "better: lower left", transform=ax.transAxes, fontsize=7.2, color=MUTED)
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=FAM[f], label=FAM_LABEL[f]) for f in FAM]
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    ax = axes[1]
    failed = []
    for c, fam, lab in ROWS:
        a = [r for r in ho2 if r["controller"] == c and r["condition"] == "H4" and not r["fail"]]
        b = [r for r in ho2 if r["controller"] == c and r["condition"] == "H6"]
        if not a or not b:
            continue
        x = np.mean([r["IAE_SO5_test"] for r in a])
        okb = [r["IAE_SO5_test"] for r in b if not r["fail"]]
        nf = sum(r["fail"] for r in b)
        short = lab.replace(" (on paper's PID)", "").replace(" (filter, gains ×0.5, ILC-AW)", "")
        if not okb:
            failed.append((x, short, FAM[fam]))
            continue
        y = np.mean(okb)
        ax.scatter(x, y, s=48, color=FAM[fam], edgecolor=SURF, lw=1.2, zorder=3)
        ax.annotate(short + (f"  (✗ {nf} of {len(b)} runs failed)" if nf else ""), (x, y),
                    xytext=(7, -11) if c == "PID" else (7, 3),
                    textcoords="offset points", fontsize=6.9, color=INK)
    ax.axhspan(0.5, 0.9, color="#efeeea", zorder=0)
    ax.text(0.99, 0.86, "✗ every run failed", fontsize=7.4, color=CRIT, fontweight="bold", ha="right", va="center",
            transform=ax.get_yaxis_transform())
    for k, (x, short, col) in enumerate(sorted(failed)):
        yl = 0.56 + 0.065 * k
        ax.scatter(x, 0.53, s=50, marker="X", color=col, zorder=3)
        ax.annotate(short, (x, 0.53), xytext=(x * 1.04, yl), fontsize=6.7, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0009, 0.03)
    ax.set_ylim(0.2, 0.9)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.002, 0.005, 0.01, 0.02]))
    ax.set_xticklabels(["0.001", "0.002", "0.005", "0.01", "0.02"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4]))
    ax.set_yticklabels(["0.2", "0.3", "0.4"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("oxygen IAE, new influent, ideal sensors (H4, log)")
    ax.set_ylabel("oxygen IAE, official sensor noise (H6, log)")
    ax.set_title("B. Nominal performance against robustness to sensor noise (held-out test 2)")
    fig.tight_layout()
    save(fig, "f4_tradeoffs.png")


# ---------------------------------------------------------------- anti-windup
def fig_antiwindup():
    ho2 = J("summary_paper1_heldout2.json")["runs"]
    tr = J("traces_paper1_heldout2.json")
    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.35, 1.35], hspace=0.12, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    wins = sorted({r["weather"] for r in ho2 if r["condition"] == "H5"})
    series = [("PID", "paper's PID", INK2), ("PIDtuned+ILC", "tuned PID + ILC", AQUA),
              ("PIDtuned+ILC-AW", "+ anti-windup", BLUE)]
    for w in wins:
        v = [next(r["IAE_SO5_test"] for r in ho2 if r["condition"] == "H5" and r["weather"] == w and r["controller"] == c)
             for c, _, _ in series]
        ax.plot(range(3), v, color=GRID, lw=1.2, zorder=1)
        for k, (c, _, col) in enumerate(series):
            ax.scatter(k, v[k], s=30, color=col, zorder=2, edgecolor=SURF, lw=0.8)
    ax.set_xticks(range(3), [l for _, l, _ in series], fontsize=7.3)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("oxygen IAE, days 7–14")
    ax.set_title("A. Overload (H5), 6 new windows", fontsize=8.8)
    means = [np.mean([r["IAE_SO5_test"] for r in ho2 if r["condition"] == "H5" and r["controller"] == c])
             for c, _, _ in series]
    ax.text(2.0, max(means) * 1.25, f"−{100 * (1 - means[2] / means[1]):.0f}% vs\nwithout", fontsize=7.5,
            color=INK, ha="center")
    # the window shown is the one whose improvement is closest to the mean improvement (not the best one)
    val = lambda c, w: next(r["IAE_SO5_test"] for r in ho2 if r["condition"] == "H5" and r["weather"] == w
                            and r["controller"] == c)
    gain = {w: val("PIDtuned+ILC-AW", w) / val("PIDtuned+ILC", w) for w in wins}
    w = min(wins, key=lambda x: abs(gain[x] - means[2] / means[1]))
    t0, t1 = 8.0, 10.0
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1:], sharex=ax1)
    for c, lab, col in series:
        d = tr[f"H5|{c}|{w}|None"]
        s, u = np.array(d["s"]), np.array(d["u"])
        t = np.arange(len(s)) * 4 * 45 / 86400
        m = (t >= t0) & (t <= t1)
        ax1.plot(t[m], s[m, 0], color=col, lw=1.3, label=lab)
        ax2.plot(t[m], u[m, 0], color=col, lw=1.3)
    ax1.axhline(2.0, color=INK2, lw=0.8)
    ax1.set_ylabel("S_O,5 (mg/L)")
    ax1.set_title(f"B. A typical window (BSM2 day {w.split(':')[1]} on, improvement closest to the mean), "
                  f"days {t0:.0f}–{t1:.0f}", fontsize=8.8)
    ax1.legend(loc="lower right", fontsize=7, ncol=3)
    ax2.axhline(240, color=AXIS, lw=0.8)
    ax2.text(t0 + 0.02, 232, "aeration limit 240", fontsize=6.8, color=MUTED, va="top")
    ax2.set_ylabel("KLa,5 (1/d)")
    ax2.set_xlabel("day")
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.suptitle("ILC anti-windup: the PID's limits move with the feedforward, and the ILC learns nothing at a limit",
                 x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.12)
    save(fig, "f5_antiwindup.png")


# ---------------------------------------------------------------- sensor noise
def fig_sensors():
    tr = J("traces_paper1_heldout2.json")
    ho2 = J("summary_paper1_heldout2.json")["runs"]
    series = [("PID", "paper's PID", INK2), ("Robust", "robust (ours)", BLUE),
              ("PIDtuned+ILC-AW", "tuned PID + ILC, anti-windup", AQUA), ("Ours", "ours: PID ×3 + ILC + SAC-IL", ORANGE)]
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 6.2), sharex=True, sharey=True)
    for ax, (c, lab, col) in zip(axes, series):
        d = tr[f"H6|{c}|dry|10"]
        s = np.array(d["s"])[:, 0]
        t = np.arange(len(s)) * 4 * 45 / 86400
        m = t <= 4.0
        ax.plot(t[m], s[m], color=col, lw=1.0)
        ax.axhline(2.0, color=INK2, lw=0.8)
        ax.axhspan(0.0, 0.3, color="#f3dcdc", zorder=0)
        ax.axhspan(4.0, 6.0, color="#f3dcdc", zorder=0)
        r = next(r for r in ho2 if r["condition"] == "H6" and r["weather"] == "dry" and r["sensors"] == 10
                 and r["controller"] == c and r["seed"] == 0)
        if r["fail"]:
            tf = r["steps"] * 45 / 86400
            ax.plot([tf], [s[-1]], marker="X", ms=9, color=CRIT, zorder=4)
            ax.text(tf - 0.03, 3.2, f"✗ failed after {tf * 24:.0f} h", color=CRIT, fontsize=7.5, fontweight="bold",
                    ha="right")
        ax.set_ylim(0, 5)
        ax.set_xlim(-0.05, 4.05)
        ax.text(0.005, 0.80, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK,
                bbox=dict(fc=SURF, ec="none", pad=1.0))
        ax.set_ylabel("S_O,5")
    axes[-1].set_xlabel("day (official dry influent, official sensors, noise seed 10)")
    fig.suptitle("Official BSM1 sensors (noise 0.25 g/m³): true oxygen in the first 4 days. "
                 "Shaded: outside 0.3–4 mg/L (a failure).", x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08, hspace=0.12)
    save(fig, "f6_sensor_noise.png")


# ---------------------------------------------------------------- learning curves
def fig_learning():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    sets = [("A. Repaired MAACC on the paper's PID", [f"curves_maacc_rep_paper_seed{s}.json" for s in range(5)], True,
             "0 = the paper's PID alone"),
            ("B. Repaired MAACC in PID ×3 + ILC", [f"curves_maacc_rep_stack_seed{s}.json" for s in range(5)], True,
             "0 = PID ×3 + ILC alone"),
            ("C. SAC-IL in PID ×3 + ILC (ours)", [f"curves_sacil_v3_seed{s}.json" for s in range(5)], False,
             "0 = after imitation")]
    for ax, (title, files, nested, zero) in zip(axes, sets):
        kept, allv = [], []
        for f in files:
            d = J(f)
            log = d["log"] if nested else d
            ep = [x["episode"] for x in log]
            y = [x["IAE_SO5"] for x in log]
            allv += y
            ax.plot(ep, y, color=ORANGE, lw=1.0, alpha=0.5, marker="o", ms=3)
            k = d["kept_episode"] if nested else J(f.replace("curves_", "summary_"))[0]["kept_episode"]
            kept.append((ep[k], y[k]))
        ax.scatter(*zip(*kept), s=50, marker="D", color=ORANGE, edgecolor=SURF, lw=1.2, zorder=3)
        ax.set_yscale("log")
        lo, hi = min(allv) / 1.3, max(allv) * 1.6
        ax.set_ylim(lo, hi)
        cand = [a * 10.0 ** e for e in range(-4, 1) for a in (1, 2, 5)]
        t = [x for x in cand if lo <= x <= hi]
        ax.yaxis.set_major_locator(FixedLocator(t))
        ax.set_yticklabels([f"{x:g}" for x in t])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(title, fontsize=8.8)
        ax.set_xlabel(f"episode of 7 dry days ({zero})", fontsize=7.6)
        if "paper's PID" in title:
            ax.axhline(0.0115, color=INK2, lw=0.8)
            ax.text(5, 0.0115 * 0.88, "paper's PID", fontsize=6.8, color=INK2, ha="right", va="top")
    axes[0].set_ylabel("oxygen IAE, dry week 1 (training data, log)")
    handles = [Line2D([], [], color=ORANGE, alpha=0.5, marker="o", ms=3, lw=1, label="one training run"),
               Line2D([], [], marker="D", ls="", ms=6, color=ORANGE, label="policy kept (best training-week score, oxygen + nitrate)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.4)
    fig.suptitle("Learning curves on the training data, 5 training runs each (the run that diverged in C kept its "
                 "episode-1 policy)", x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.25, wspace=0.25)
    save(fig, "f7_learning.png")


RUNS = {}


def main():
    RUNS.update(runs_by_condition())
    S = stats(RUNS)
    fig_matrix(S)
    fig_testweek(S)
    fig_heldout(S)
    fig_tradeoffs(S)
    fig_antiwindup()
    fig_sensors()
    fig_learning()
    # the numbers behind figure 1, for the table view
    tab = {cond: {c: {k: v for k, v in d.items() if k != "values"} for c, d in S[cond].items()} for cond, _ in COLS}
    (RES / "paper1_final_figure_table.json").write_text(json.dumps(tab, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
