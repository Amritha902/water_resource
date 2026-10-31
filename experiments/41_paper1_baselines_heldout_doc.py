"""Tables and figures of the stronger baselines and the held-out test (docs/PAPER1_BASELINES_HELDOUT.md).

Reads results/summary_paper1_study.json (33), paper1_tuned_pid.json (38),
curves_maacc_rep_*.json (39) and summary_paper1_heldout.json (40).  Writes
results/paper1_heldout_tables.json and docs/figures/heldout/*.png.

    python experiments/41_paper1_baselines_heldout_doc.py
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
OUT = ROOT / "docs" / "figures" / "heldout"
J = lambda n: json.loads((RES / n).read_text())
W3 = ("dry", "rain", "storm")
CTRL = ["PID", "MAACC", "rMAACC", "PIDx3", "PIDx3+ILC", "PIDx3+ILC+rMAACC", "Ours",
        "PIDtuned", "PIDtuned+ILC", "OursT"]
LABEL = {"PID": "paper's PID", "MAACC": "MAACC as printed", "rMAACC": "repaired MAACC (paper's PID prior)",
         "PIDx3": "PID ×3", "PIDx3+ILC": "PID ×3 + ILC", "PIDx3+ILC+rMAACC": "PID ×3 + ILC + repaired MAACC",
         "Ours": "ours: PID ×3 + ILC + SAC-IL", "PIDtuned": "tuned PID", "PIDtuned+ILC": "tuned PID + ILC",
         "OursT": "tuned PID + ILC + SAC-IL"}
COL = {"PID": "#eb6834", "MAACC": "#c9a227", "rMAACC": "#e3cf7a", "PIDx3": "#8a8a84", "PIDx3+ILC": "#1baf7a",
       "PIDx3+ILC+rMAACC": "#9b7fd4", "Ours": "#2a78d6", "PIDtuned": "#5c5c58", "PIDtuned+ILC": "#127a50",
       "OursT": "#103f80"}
KEYS = ["IAE_SO5_test", "IAE_SNO2_test", "DEVmax_SO5_test", "KLa_move_per_day", "AE", "eff_SNH", "eff_TN",
        "at_limit_KLa", "gated_share"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.color": "#e4e3de",
                     "axes.spines.top": False, "axes.spines.right": False})


def mean(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)) if v else float("nan")


def study_tables():
    runs = J("summary_paper1_study.json")["runs"]
    T = {}
    for case in W3 + ("noise",):
        w, noise = ("dry", True) if case == "noise" else (case, False)
        T[case] = {}
        for c in CTRL:
            rows = [r for r in runs if r["controller"] == c and r["weather"] == w and r["noise"] == noise]
            T[case][c] = {k: mean(rows, k if case != "noise" else k.replace("_test", "")) for k in KEYS}
            T[case][c]["n"] = len(rows)
    return T


def heldout_tables():
    H = J("summary_paper1_heldout.json")
    runs = H["runs"]
    T = {"H1": {}, "H2": {}, "H3": {}}
    units = {"H1": [f"bsm2:{w}" for w in H["windows"]], "H2": list(W3), "H3": list(W3)}
    for cond in T:
        for c in CTRL:
            rows = [r for r in runs if r["condition"] == cond and r["controller"] == c]
            ok = [r for r in rows if not r["fail"]]
            d = {k: mean(ok, k) for k in KEYS}
            d.update(n=len(rows), failed=len(rows) - len(ok))
            # per unit (window or weather), mean over learned seeds and noise seeds
            d["per_unit"] = {u: mean([r for r in ok if r["weather"] == u], "IAE_SO5_test") for u in units[cond]}
            T[cond][c] = d

    def compare(cond, a, b):
        """Paired comparison of mean oxygen IAE per run unit (window, or weather x noise seed)."""
        rows = [r for r in runs if r["condition"] == cond]
        keyf = lambda r: (r["weather"], str(r["sensors"]))
        units_ = sorted({keyf(r) for r in rows})
        wins, total, pa, pb = 0, 0, [], []
        for u in units_:
            ra = [r for r in rows if keyf(r) == u and r["controller"] == a]
            rb = [r for r in rows if keyf(r) == u and r["controller"] == b]
            if not ra or not rb or any(r["fail"] for r in ra + rb):
                continue
            xa, xb = mean(ra, "IAE_SO5_test"), mean(rb, "IAE_SO5_test")
            pa.append(xa)
            pb.append(xb)
            total += 1
            wins += xa < xb
        change = 100 * (np.mean(pa) / np.mean(pb) - 1) if total else float("nan")
        better = bool(total and np.mean(pa) < np.mean(pb) and wins >= 2 * total / 3)
        return dict(a=a, b=b, change_pct=float(change), wins=int(wins), units=int(total), better=better)

    pairs = [("Ours", "PIDx3+ILC"), ("Ours", "PIDx3+ILC+rMAACC"), ("Ours", "PIDtuned+ILC"),
             ("Ours", "PID"), ("OursT", "PIDtuned+ILC"), ("PIDtuned+ILC", "PID"), ("rMAACC", "PID"),
             ("PIDx3+ILC", "PIDx3"), ("PIDtuned+ILC", "PIDtuned")]
    T["comparisons"] = {cond: [compare(cond, a, b) for a, b in pairs] for cond in ("H1", "H2", "H3")}
    return T


def bars(T, cases, key, fname, title, logy=True, labels=None):
    fig, axes = plt.subplots(1, len(cases), figsize=(3.9 * len(cases), 3.6), sharey=logy)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        vals = [T[case][c][key] for c in CTRL]
        ax.bar(range(len(CTRL)), vals, color=[COL[c] for c in CTRL])
        if logy:
            ax.set_yscale("log")
        for j, v in enumerate(vals):
            if v != v:
                ax.text(j, 0.02, "all runs\nfailed", transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6.5, color="#b03a2e", rotation=90)
        ax.set_xticks(range(len(CTRL)), [LABEL[c] for c in CTRL], rotation=70, ha="right", fontsize=7)
        ax.set_title((labels or {}).get(case, case), loc="left", fontsize=9)
    axes[0].set_ylabel(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def fig_tuned_grid():
    t = J("paper1_tuned_pid.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, name in zip(axes, ("oxygen_grid", "nitrate_grid"), ("oxygen loop", "nitrate loop")):
        g = t[key]
        ms = sorted({r["kp"] for r in g})
        M = np.array([[next(r["IAE"] for r in g if r["kp"] == p and r["ki"] == i) for i in ms] for p in ms])
        im = ax.imshow(np.log10(M), cmap="viridis_r", aspect="auto")
        best = t["terms"][0][0 if key == "oxygen_grid" else 1], t["terms"][1][0 if key == "oxygen_grid" else 1]
        for a, p in enumerate(ms):
            for b, i in enumerate(ms):
                r = next(r for r in g if r["kp"] == p and r["ki"] == i)
                ax.text(b, a, f"{r['IAE'] * 1000:.2f}" + ("" if r["admissible"] else "\n✗"), ha="center",
                        va="center", fontsize=6.5, color="white" if np.log10(M[a, b]) > np.median(np.log10(M)) else "black")
                if (p, i) == best:
                    ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2))
        ax.set_xticks(range(len(ms)), ms)
        ax.set_yticks(range(len(ms)), ms)
        ax.set_xlabel("I multiplier")
        ax.set_ylabel("P (= D) multiplier")
        ax.grid(False)
        ax.set_title(f"{name}: IAE ×1000, dry week 1 (✗ = fails the ×1.5 margin)", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "tuned_pid_grid.png", dpi=160)
    plt.close(fig)


def fig_rmaacc_training():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, mode, name in zip(axes, ("paper", "stack"), ("on the paper's PID", "on PID ×3 + ILC")):
        for s in range(5):
            f = RES / f"curves_maacc_rep_{mode}_seed{s}.json"
            if f.exists():
                log = json.loads(f.read_text())["log"]
                ax.plot([x["episode"] for x in log], [x["IAE_SO5"] for x in log], "o-", ms=3, lw=1, label=f"run {s + 1}")
        ax.set_yscale("log")
        ax.set_xlabel("episode (7 days of dry weather; 0 = the prior alone)")
        ax.set_title(f"repaired MAACC {name}: oxygen IAE, dry week 1", loc="left", fontsize=9)
    axes[0].set_ylabel("oxygen IAE")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "rmaacc_training.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = study_tables()
    H = heldout_tables()
    (RES / "paper1_heldout_tables.json").write_text(json.dumps({"study": S, "heldout": H}, indent=1))
    bars(S, W3, "IAE_SO5_test", "study_oxygen.png", "oxygen IAE, test week")
    bars(H, ("H1", "H2", "H3"), "IAE_SO5_test", "heldout_oxygen.png", "oxygen IAE, days 7-14",
         labels={"H1": "H1: BSM2 influent (6 windows)", "H2": "H2: official sensors (delay + noise)",
                 "H3": "H3: sensor delay only"})
    fig_tuned_grid()
    fig_rmaacc_training()
    for cond in ("H1", "H2", "H3"):
        print(cond)
        for c in CTRL:
            v = H[cond][c]
            print(f"  {c:17s} O {v['IAE_SO5_test']:.5f} N {v['IAE_SNO2_test']:.5f} dev {v['DEVmax_SO5_test']:.3f} "
                  f"KLa@limit {v['at_limit_KLa']:.2f} gate {v['gated_share']:.2f} fail {v['failed']}/{v['n']}")
        for cmp in H["comparisons"][cond]:
            print(f"  {cmp['a']} vs {cmp['b']}: {cmp['change_pct']:+.0f}% wins {cmp['wins']}/{cmp['units']} "
                  f"better={cmp['better']}")


if __name__ == "__main__":
    main()
