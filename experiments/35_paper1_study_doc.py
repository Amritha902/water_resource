"""Figures and tables of the paper 1 study (docs/figures/study/, results/paper1_study_tables.json).

Reads results/paper1_tuning.json (32), curves_sacil_v3_seed*.json (28) and
summary_paper1_study.json / traces_paper1_study.json (33), and for the
midnight comparison the round 1 files (*_round1.json) when present.

    python experiments/35_paper1_study_doc.py
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
OUT = ROOT / "docs" / "figures" / "study"
J = lambda n: json.loads((RES / n).read_text())
W3 = ("dry", "rain", "storm")
CTRL = ["PID", "MAACC", "PID+ILC", "MAACC+ILC", "PIDx3", "PIDx3+ILC", "PIDx3+SAC-IL", "Ours (no gate)", "Ours"]
LABEL = {"PID": "paper's PID", "MAACC": "MAACC (paper, as printed)", "PID+ILC": "paper's PID + ILC",
         "MAACC+ILC": "MAACC + ILC", "PIDx3": "PID ×3", "PIDx3+ILC": "PID ×3 + ILC",
         "PIDx3+SAC-IL": "PID ×3 + SAC-IL (no ILC)", "Ours (no gate)": "ours without gate",
         "Ours": "OURS: PID ×3 + ILC + SAC-IL + gate"}
COL = {"PID": "#eb6834", "MAACC": "#c9a227", "PID+ILC": "#f2a37a", "MAACC+ILC": "#e3cf7a", "PIDx3": "#8a8a84",
       "PIDx3+ILC": "#1baf7a", "PIDx3+SAC-IL": "#9cc2ee", "Ours (no gate)": "#b9b9f0", "Ours": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.color": "#e4e3de",
                     "axes.spines.top": False, "axes.spines.right": False})


def agg(runs, ctrl, case, key):
    noise = case == "noise"
    w = "dry" if noise else case
    v = [r[key] for r in runs if r["controller"] == ctrl and r["weather"] == w and r["noise"] == noise]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def tables(study):
    runs = study["runs"]
    pub = J("published_paper1.json")
    keys = ["IAE_SO5", "IAE_SO5_test", "IAE_SNO2", "IAE_SNO2_test", "ISE_SO5", "DEVmax_SO5", "DEVmax_SO5_test",
            "DEVmax_SNO2", "AE", "PE_Qa", "KLa_move_per_day", "Qa_move_per_day", "at_limit_KLa", "at_limit_Qa",
            "eff_SNH", "eff_TN", "eff_SNH_over_4", "eff_TN_over_18", "gated_share", "wall_s"]
    T = {}
    for case in W3 + ("noise",):
        T[case] = {}
        for c in CTRL:
            T[case][c] = {k: agg(runs, c, case, k) for k in keys}
        pk = "noisy_dry" if case == "noise" else case
        T[case]["paper"] = {"PID_SO5": pub[pk]["PID"]["IAE_SO5"], "MAACC_SO5": pub[pk]["MAACC"]["IAE_SO5"],
                            "PID_SNO2": pub[pk]["PID"]["IAE_SNO2"], "MAACC_SNO2": pub[pk]["MAACC"]["IAE_SNO2"]}
    T["compute"] = study["compute"]
    T["tuning"] = study["tuning"]
    (RES / "paper1_study_tables.json").write_text(json.dumps(T, indent=1))
    return T


def midnight_stats(tr, ctrls=("PIDx3", "PIDx3+ILC", "Ours")):
    """Largest oxygen deviation within 15 minutes of midnight, and elsewhere (seed 0, no noise)."""
    out = {}
    for c in ctrls:
        for w in W3:
            k = f"{c}|{w}|clean"
            if k not in tr:
                continue
            e = np.abs(np.array(tr[k]["s"])[:, 0] - 2)
            t = np.arange(len(e)) * 4 * 45 / 86400
            near = np.minimum(t % 1, 1 - t % 1) <= 15 / 1440
            out[k] = {"midnight": float(e[near].max()), "elsewhere": float(e[~near].max())}
    return out


def fig_midnight(tr1, tr2):
    """Oxygen around three midnights, dry weather: round 1 (switch) and round 2 (bumpless)."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0), sharey=True)
    dt = 4 * 45 / 86400
    for ax, day in zip(axes, (2, 3, 9)):
        for tr, c, lab, col, lw in ((tr2, "PIDx3", "PID ×3", COL["PIDx3"], 0.9),
                                    (tr1, "Ours", "ours, round 1 (profile switched)", "#c0392b", 1.0),
                                    (tr2, "Ours", "ours, round 2 (bumpless)", COL["Ours"], 1.4)):
            if tr is None:
                continue
            s = np.array(tr[f"{c}|dry|clean"]["s"])[:, 0]
            t = np.arange(len(s)) * dt
            m = (t >= day - 0.08) & (t <= day + 0.08)
            ax.plot((t[m] - day) * 24, s[m], color=col, lw=lw, label=lab)
        ax.axvline(0, color="k", ls=":", lw=0.7)
        ax.set_title(f"midnight, start of day {day + 1}", loc="left", fontsize=9)
        ax.set_xlabel("hours from midnight")
    axes[0].set_ylabel("S_O,5 (mg/L), dry")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "midnight.png", dpi=170)
    plt.close(fig)


def fig_gain_ilc():
    t = J("paper1_tuning.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    sw = t["gain_sweep"]
    ax = axes[0]
    ax.semilogy([s["gain"] for s in sw], [s["IAE_SO5"] for s in sw], "o-", color="#eb6834", label="oxygen")
    ax.semilogy([s["gain"] for s in sw], [s["IAE_SNO2"] for s in sw], "s-", color="#6b7b83", label="nitrate")
    ax.axvline(t["gain"], color="#1baf7a", ls="--", lw=1)
    ax.axvline(t["gain_limit"], color="#c0392b", ls=":", lw=1)
    ax.text(t["gain"] + 0.08, ax.get_ylim()[1] * 0.5, f"chosen ×{t['gain']}", color="#1baf7a", fontsize=8)
    ax.text(t["gain_limit"] + 0.08, ax.get_ylim()[1] * 0.5, f"unstable ×{t['gain_limit']}", color="#c0392b", fontsize=8)
    ax.set_xlabel("gain factor on the paper's PID")
    ax.set_ylabel("IAE, dry week 1")
    ax.set_title("A. PID gain (dry week 1 only)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    g = [r for r in t["ilc_grid"] if r["ilc"]["smooth"] == t["ilc"]["smooth"]]
    betas = sorted({r["ilc"]["beta"][0] for r in g})
    leads = sorted({r["ilc"]["lead"][0] for r in g})
    M = np.array([[next(r["IAE_SO5"] for r in g if r["ilc"]["beta"][0] == b and r["ilc"]["lead"][0] == l)
                   for l in leads] for b in betas])
    im = ax.imshow(M * 1000, cmap="viridis_r", aspect="auto")
    dev_pid = next(s["DEVmax_SO5"] for s in sw if s["gain"] == t["gain"])
    for i, b in enumerate(betas):
        for j, l in enumerate(leads):
            r = next(r for r in g if r["ilc"]["beta"][0] == b and r["ilc"]["lead"][0] == l)
            mark = "" if r["DEVmax_SO5"] <= dev_pid else "\n(peak ↑)"
            if r["ilc"] == t["ilc"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2))
            ax.text(j, i, f"{M[i, j] * 1000:.2f}{mark}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > np.median(M) else "black")
    ax.set_xticks(range(len(leads)), [f"{l} ({l * 45 / 60:.1f} min)" for l in leads], fontsize=7.5)
    ax.set_yticks(range(len(betas)), betas)
    ax.set_xlabel("lead (control periods)")
    ax.set_ylabel("share β per day")
    ax.grid(False)
    ax.set_title(f"B. ILC: oxygen IAE ×1000, dry week 1 (smoothing ±{t['ilc']['smooth']})", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "tuning.png", dpi=170)
    plt.close(fig)


def fig_training():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for s in range(5):
        f = RES / f"curves_sacil_v3_seed{s}.json"
        if not f.exists():
            continue
        log = json.loads(f.read_text())
        ax.plot([x["episode"] for x in log], [x["IAE_SO5"] for x in log], "o-", ms=3.5, lw=1.1, label=f"run {s + 1}")
    ax.axhline(log[0]["IAE_SO5"], color="#1baf7a", ls="--", lw=1, label="after imitation (= expert)")
    ax.set_yscale("log")
    ax.set_xlabel("SAC episode (7 days of dry weather; 0 = after imitation)")
    ax.set_ylabel("oxygen IAE, dry week 1")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=170)
    plt.close(fig)


def fig_results(T):
    show = ["MAACC", "MAACC+ILC", "PIDx3", "PIDx3+ILC", "PIDx3+SAC-IL", "Ours"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6))
    for row, (key, name) in enumerate((("IAE_SO5", "oxygen S_O,5"), ("IAE_SNO2", "nitrate S_NO,2"))):
        for col, (suffix, title) in enumerate((("_test", "TEST: week 2 only (unseen)"), ("", "paper's protocol: 14 days"))):
            ax = axes[row, col]
            x = np.arange(3)
            for i, c in enumerate(show):
                v = [100 * (1 - T[w][c][key + suffix][0] / T[w]["PID"][key + suffix][0]) for w in W3]
                e = [100 * T[w][c][key + suffix][1] / T[w]["PID"][key + suffix][0] for w in W3]
                bars = ax.bar(x - 0.375 + 0.15 * i, v, 0.15, yerr=e, color=COL[c], label=LABEL[c], capsize=2,
                              error_kw={"lw": 0.8})
                if c == "Ours":
                    for b, val in zip(bars, v):
                        ax.text(b.get_x() + b.get_width() / 2, val + 2, f"{val:.0f}", ha="center", fontsize=7.5,
                                color=COL["Ours"], fontweight="bold")
            if suffix == "":
                claim = [100 * (1 - T[w]["paper"][("MAACC_" + key[4:])] / T[w]["paper"][("PID_" + key[4:])]) for w in W3]
                ax.scatter(x + 0.5, claim, marker="_", s=300, color="#5b5b57", lw=2, label="paper's claim", zorder=4)
            ax.set_xticks(x, W3)
            ax.set_ylim(-10, 100)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{name} · {title}", loc="left", fontsize=9)
            if col == 0:
                ax.set_ylabel("IAE reduction vs paper's PID (%)")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=170)
    plt.close(fig)


def fig_tracking():
    tr = J("traces_paper1_study.json")
    dt = 4 * 45 / 86400
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, w in zip(axes, W3):
        for c, lw in (("MAACC", 1.6), ("PIDx3", 0.9), ("Ours", 0.9)):
            k = f"{c}|{w}|clean"
            if k not in tr:
                continue
            s = np.array(tr[k]["s"])
            t = np.arange(len(s)) * dt
            m = (t >= 8) & (t <= 11)
            ax.plot(t[m], s[m, 0], color=COL[c], lw=lw, label=LABEL[c])
        ax.axhline(2, color="k", ls=":", lw=0.7)
        ax.set_ylabel(f"{w}\nS_O,5 (mg/L)")
        ax.set_ylim(1.84, 2.08)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
    axes[-1].set_xlabel("time (day) — days 8 to 11 of the test week")
    fig.tight_layout()
    fig.savefig(OUT / "tracking.png", dpi=170)
    plt.close(fig)


def fig_daily():
    tr = J("traces_paper1_study.json")
    per = int(round(86400 / (4 * 45)))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, w in zip(axes, W3):
        for c in ("MAACC", "PIDx3", "PIDx3+ILC", "Ours"):
            k = f"{c}|{w}|clean"
            if k not in tr:
                continue
            e = np.abs(np.array(tr[k]["s"])[:, 0] - 2)
            d = [e[i * per:(i + 1) * per].mean() for i in range(len(e) // per)]
            ax.plot(range(1, len(d) + 1), d, "o-", ms=3, lw=1, color=COL[c], label=LABEL[c])
        ax.set_yscale("log")
        ax.axvspan(7.5, 14.5, color="#eaf2f4", zorder=0)
        ax.set_title(w, loc="left", fontsize=9)
        ax.set_xlabel("day")
    axes[0].set_ylabel("daily IAE of S_O,5")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "daily.png", dpi=170)
    plt.close(fig)


def fig_tradeoffs(T):
    items = [("AE", "aeration energy (kWh/d)"), ("PE_Qa", "recycle pumping energy (kWh/d)"),
             ("KLa_move_per_day", "KLa,5 movement (1/d per day)"), ("Qa_move_per_day", "Q_a movement (m³/d per day)"),
             ("DEVmax_SO5", "largest oxygen deviation (mg/L)"), ("eff_SNH", "effluent ammonium (g N/m³)"),
             ("eff_TN", "effluent total N (g N/m³)"), ("eff_SNH_over_4", "time with effluent NH4 > 4")]
    show = ["MAACC", "PIDx3", "PIDx3+ILC", "Ours"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, (k, title) in zip(axes.flat, items):
        x = np.arange(3)
        for i, c in enumerate(show):
            ax.bar(x - 0.3 + 0.2 * i, [T[w][c][k][0] for w in W3], 0.2, color=COL[c], label=LABEL[c])
        lo = min(T[w][c][k][0] for w in W3 for c in show)
        hi = max(T[w][c][k][0] for w in W3 for c in show)
        if k in ("AE", "PE_Qa", "eff_TN"):
            ax.set_ylim(lo * 0.97, hi * 1.02)
        ax.set_xticks(x, W3, fontsize=8)
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoffs.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    study = J("summary_paper1_study.json")
    T = tables(study)
    fig_gain_ilc()
    fig_training()
    fig_results(T)
    fig_tracking()
    fig_daily()
    fig_tradeoffs(T)
    tr2 = J("traces_paper1_study.json")
    f1 = RES / "traces_paper1_study_round1.json"
    tr1 = json.loads(f1.read_text()) if f1.exists() else None
    fig_midnight(tr1, tr2)
    T["midnight"] = {"round2": midnight_stats(tr2)}
    if tr1 is not None:
        T["midnight"]["round1"] = midnight_stats(tr1)
    (RES / "paper1_study_tables.json").write_text(json.dumps(T, indent=1))
    print("midnight:", json.dumps(T["midnight"], indent=1))
    for case in W3 + ("noise",):
        print(case)
        for c in CTRL:
            v = T[case][c]
            print(f"  {c:15s} O {v['IAE_SO5'][0]:.5f}±{v['IAE_SO5'][1]:.5f} test {v['IAE_SO5_test'][0]:.5f}  "
                  f"N {v['IAE_SNO2'][0]:.5f} test {v['IAE_SNO2_test'][0]:.5f}  AE {v['AE'][0]:.0f} "
                  f"KLa mv {v['KLa_move_per_day'][0]:.0f} effNH {v['eff_SNH'][0]:.2f} TN {v['eff_TN'][0]:.2f} "
                  f"gate {100 * v['gated_share'][0]:.0f}% n={v['IAE_SO5'][2]}")
    print("compute", T["compute"])


if __name__ == "__main__":
    main()
