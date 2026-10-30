"""Paper 1 task, our controller: retuned PID prior + daily ILC + safe SAC-IL.

Collects every stage of the comparison into results/summary_paper1_new.json,
records traces of the final controller, and draws docs/figures/new/.

Stages (oxygen and nitrate IAE over the full 14 days, from the common warm-up):
  paper's PID prior (= MAACC as printed) -> PID x3 -> + daily ILC
  -> + SAC-IL correction (imitation + soft actor-critic) -> + input clipping and safety gate

Needs results/summary_ilc_paper1.json (29), summary_sacil_ilc_seed*.json (28),
summary_sacil_guard.json (30) and the trained actors in artifacts/.

    python experiments/31_paper1_new_method.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "new"
J = lambda n: json.loads((RES / n).read_text())
W3 = ("dry", "rain", "storm")
DT4 = 4 * 45 / 86400
PID_C, X3_C, ILC_C, OURS_C, PAPER_C, MA_C = "#eb6834", "#8a8a84", "#1baf7a", "#2a78d6", "#5b5b57", "#c9a227"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.color": "#e4e3de",
                     "axes.spines.top": False, "axes.spines.right": False})


def load_guard():
    spec = importlib.util.spec_from_file_location("g30", ROOT / "experiments" / "30_sacil_guard.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def table():
    ilc = J("summary_ilc_paper1.json")
    guard = J("summary_sacil_guard.json")
    pub = J("published_paper1.json")
    ex = J("summary_paper1_exact2.json")
    rows = []
    for w in W3 + ("noise",):
        noise = w == "noise"
        ww = "dry" if noise else w
        def m(vals, k):
            return float(np.mean([v[k] for v in vals]))
        stage = {}
        for c in ("PID", "PIDx3", "PIDx3+ILC"):
            v = [x for x in ilc if x["weather"] == ww and x["noise"] == noise and x["controller"] == c]
            stage[c] = (m(v, "IAE_SO5"), m(v, "IAE_SNO2"), len(v))
        v = [x for x in ex if x["weather"] == ww and x["noise"] == noise and x["controller"] == "MAACC-paper"]
        stage["MAACC as printed"] = (m(v, "IAE_SO5"), m(v, "IAE_SNO2"), len(v))
        for mode, name in (("raw", "PIDx3+ILC+SAC-IL"), ("clip+gate", "Ours: PIDx3+ILC+safe SAC-IL")):
            v = [x for x in guard if x["weather"] == ww and x["noise"] == noise and x["mode"] == mode]
            stage[name] = (m(v, "IAE_SO5"), m(v, "IAE_SNO2"), len(v))
        key = "noisy_dry" if noise else w
        for name, (o, n, cnt) in stage.items():
            rows.append(dict(case=w, controller=name, IAE_SO5=o, IAE_SNO2=n, runs=cnt,
                             paper_PID_SO5=pub[key]["PID"]["IAE_SO5"], paper_PID_SNO2=pub[key]["PID"]["IAE_SNO2"],
                             paper_MAACC_SO5=pub[key]["MAACC"]["IAE_SO5"], paper_MAACC_SNO2=pub[key]["MAACC"]["IAE_SNO2"],
                             vs_our_PID_SO5=o / stage["PID"][0] - 1, vs_our_PID_SNO2=n / stage["PID"][1] - 1))
    (RES / "summary_paper1_new.json").write_text(json.dumps(rows, indent=1))
    return rows


def final_traces(g):
    """Final controller, seed 0, with traces (clip + gate)."""
    lo, hi = g.RANGE
    tr = {}
    for w in W3:
        pol, _ = g.policy(0, "clip+gate", lo, hi)
        r = g.evaluate(g.env(w), pol)
        tr[w] = r["_trace"]
    (RES / "traces_paper1_new.json").write_text(json.dumps(tr))
    return tr


def fig_tracking(tr_final):
    ilc = J("traces_ilc_paper1.json")
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for row, w in enumerate(W3):
        for col, (a, b) in enumerate(((0, 14), (9.0, 11.0))):
            ax = axes[row, col]
            for s, c, lw, name in ((np.array(ilc[f"{w}|clean|PID"]["s"]), PID_C, 1.0, "paper's PID prior (= MAACC as printed)"),
                                   (np.array(ilc[f"{w}|clean|PIDx3"]["s"]), X3_C, 0.9, "PID ×3"),
                                   (np.array(tr_final[w]["s"]), OURS_C, 0.9, "ours: PID ×3 + ILC + safe SAC-IL")):
                t = np.arange(len(s)) * DT4
                k = (t >= a) & (t <= b)
                ax.plot(t[k], s[k, 0], color=c, lw=lw, label=name)
            ax.axhline(2.0, color="k", lw=0.7, ls=":")
            ax.set_xlim(a, b)
            ax.set_ylim(1.84, 2.08)
            if col == 0:
                ax.set_ylabel(f"{w}\nS_O,5 (mg/L)")
            else:
                ax.set_title("zoom, days 9–11" if row == 0 else "", fontsize=8, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
    axes[2, 0].set_xlabel("time (day)")
    axes[2, 1].set_xlabel("time (day)")
    fig.tight_layout()
    fig.savefig(OUT / "new_p1_tracking.png", dpi=170)
    plt.close(fig)


def fig_bars(rows):
    names = ["MAACC as printed", "PIDx3", "PIDx3+ILC", "Ours: PIDx3+ILC+safe SAC-IL"]
    labels = ["MAACC as printed", "PID ×3", "PID ×3 + ILC", "Ours (+ safe SAC-IL)"]
    colors = [MA_C, X3_C, ILC_C, OURS_C]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title in zip(axes, ("SO5", "SNO2"), ("oxygen S_O,5", "nitrate S_NO,2")):
        x = np.arange(3)
        claim = [100 * (1 - [r for r in rows if r["case"] == w][0][f"paper_MAACC_{key}"] /
                        [r for r in rows if r["case"] == w][0][f"paper_PID_{key}"]) for w in W3]
        ax.bar(x - 0.36, claim, 0.15, color="white", edgecolor=PAPER_C, hatch="///", label="paper's claim (not reproducible)")
        for i, (n, lab, c) in enumerate(zip(names, labels, colors)):
            v = [-100 * [r for r in rows if r["case"] == w and r["controller"] == n][0][f"vs_our_PID_{key}"] for w in W3]
            bars = ax.bar(x - 0.18 + 0.15 * i, v, 0.15, color=c, label=lab)
            if n.startswith("Ours"):
                for b, val in zip(bars, v):
                    ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center", fontsize=7.5,
                            color=OURS_C, fontweight="bold")
        ax.set_xticks(x, W3)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylim(-5, 100)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("reduction of IAE vs the paper's PID prior (%)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "new_p1_reduction.png", dpi=170)
    plt.close(fig)


def fig_daily(tr_final):
    ilc = J("traces_ilc_paper1.json")
    per = int(round(1 / DT4))
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    for s, c, name in ((np.array(ilc["dry|clean|PID"]["s"]), PID_C, "paper's PID prior"),
                       (np.array(ilc["dry|clean|PIDx3"]["s"]), X3_C, "PID ×3"),
                       (np.array(ilc["dry|clean|PIDx3+ILC"]["s"]), ILC_C, "PID ×3 + ILC"),
                       (np.array(tr_final["dry"]["s"]), OURS_C, "ours (+ safe SAC-IL)")):
        e = np.abs(s[:, 0] - 2.0)
        d = [e[i * per:(i + 1) * per].mean() for i in range(len(e) // per)]
        ax.plot(np.arange(1, len(d) + 1), d, "o-", color=c, ms=3.5, lw=1.2, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("day of the run (dry weather)")
    ax.set_ylabel("daily IAE of S_O,5 (mg/L)")
    ax.axvspan(5.5, 7.5, color="#f1f0ec", zorder=0)
    ax.axvspan(12.5, 14.5, color="#f1f0ec", zorder=0)
    ax.text(6.5, ax.get_ylim()[1] * 0.75, "weekend", ha="center", fontsize=7, color="#6b6a64")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "new_p1_daily_learning.png", dpi=170)
    plt.close(fig)


def fig_training():
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for s, c in zip((0, 1, 2), ("#2a78d6", "#5d9be3", "#9cc2ee")):
        log = J(f"curves_sacil_ilc_seed{s}.json")
        ep = [x["episode"] for x in log]
        ax.plot(ep, [x["IAE_SO5"] for x in log], "o-", color=c, ms=4, label=f"run {s + 1}")
    ax.axhline(log[0]["IAE_SO5"], color=ILC_C, ls="--", lw=1, label="imitation start (= PID ×3 + ILC)")
    ax.set_yscale("log")
    ax.set_xlabel("training episode (7 days of dry weather; 0 = after imitation)")
    ax.set_ylabel("IAE of S_O,5 on the training days")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "new_p1_training.png", dpi=170)
    plt.close(fig)


def fig_profile():
    ilc = J("summary_ilc_paper1.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for w, c in zip(W3, ("#2a78d6", "#1baf7a", "#eb6834")):
        p = [x for x in ilc if x["weather"] == w and not x["noise"] and x["controller"] == "PIDx3+ILC"][0]["ilc_profile_KLa"]
        ax.plot(np.arange(len(p)) * 8 * 45 / 3600, p, color=c, lw=1.2, label=w)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("learned KLa,5 feedforward (1/d)")
    ax.legend(frameon=False, fontsize=7.5, title="profile at the end of the run", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "new_p1_ilc_profile.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = table()
    for r in rows:
        print(f"{r['case']:6s} {r['controller']:30s} O {r['IAE_SO5']:.5f} ({100 * r['vs_our_PID_SO5']:+.0f}%)  "
              f"N {r['IAE_SNO2']:.5f} ({100 * r['vs_our_PID_SNO2']:+.0f}%)  runs {r['runs']}")
    tr = final_traces(load_guard())
    fig_tracking(tr)
    fig_bars(rows)
    fig_daily(tr)
    fig_training()
    fig_profile()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
