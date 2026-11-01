"""Tables, figure and the pre-declared answers of held-out test 2 (docs/PAPER1_HELDOUT2_PROTOCOL.md).

Reads results/summary_paper1_heldout2.json (43).  Writes
results/paper1_heldout2_tables.json and docs/figures/heldout/heldout2_oxygen.png.

    python experiments/44_paper1_heldout2_doc.py
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
CTRL = ["PID", "MAACC", "PIDtuned", "PIDtuned+ILC", "PIDtuned+ILC-AW", "Robust", "rMAACC", "Ours"]
LABEL = {"PID": "paper's PID", "MAACC": "MAACC as printed", "PIDtuned": "tuned PID",
         "PIDtuned+ILC": "tuned PID + ILC", "PIDtuned+ILC-AW": "tuned PID + ILC, anti-windup",
         "Robust": "robust: filtered, re-scaled + ILC-AW", "rMAACC": "repaired MAACC",
         "Ours": "ours (round 2): PID ×3 + ILC + SAC-IL"}
COL = {"PID": "#eb6834", "MAACC": "#c9a227", "PIDtuned": "#5c5c58", "PIDtuned+ILC": "#127a50",
       "PIDtuned+ILC-AW": "#1baf7a", "Robust": "#103f80", "rMAACC": "#e3cf7a", "Ours": "#2a78d6"}
CONDS = {"H4": "H4: new influent, clarified", "H5": "H5: new influent, raw (overload)",
         "H6": "H6: official sensors, new noise", "H7": "H7: new influent + official sensors"}
KEYS = ["IAE_SO5_test", "IAE_SNO2_test", "DEVmax_SO5_test", "KLa_move_per_day", "at_limit_KLa", "gated_share"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.color": "#e4e3de",
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    runs = json.loads((RES / "summary_paper1_heldout2.json").read_text())["runs"]
    unit = lambda r: (r["weather"], str(r["sensors"]))
    T = {}
    for cond in CONDS:
        T[cond] = {}
        for c in CTRL:
            rows = [r for r in runs if r["condition"] == cond and r["controller"] == c]
            ok = [r for r in rows if not r["fail"]]
            d = {k: float(np.mean([r[k] for r in ok])) if ok else float("nan") for k in KEYS}
            d.update(n=len(rows), failed=len(rows) - len(ok))
            T[cond][c] = d

    def compare(cond, a, b):
        rows = [r for r in runs if r["condition"] == cond]
        units = sorted({unit(r) for r in rows})
        wins, xa, xb = 0, [], []
        for u in units:
            ra = [r for r in rows if unit(r) == u and r["controller"] == a]
            rb = [r for r in rows if unit(r) == u and r["controller"] == b]
            if not ra or not rb or any(r["fail"] for r in ra + rb):
                continue
            ma, mb = np.mean([r["IAE_SO5_test"] for r in ra]), np.mean([r["IAE_SO5_test"] for r in rb])
            xa.append(ma)
            xb.append(mb)
            wins += ma < mb
        n = len(xa)
        fa = sum(r["fail"] for r in rows if r["controller"] == a)
        fb = sum(r["fail"] for r in rows if r["controller"] == b)
        return dict(a=a, b=b, change_pct=float(100 * (np.mean(xa) / np.mean(xb) - 1)) if n else float("nan"),
                    wins=int(wins), units=n, better=bool(n and np.mean(xa) < np.mean(xb) and wins >= 2 * n / 3),
                    failures_a=fa, failures_b=fb)

    Q = {
        "i": [compare("H5", "PIDtuned+ILC-AW", "PIDtuned+ILC"), compare("H4", "PIDtuned+ILC-AW", "PIDtuned+ILC")],
        "ii": [compare("H6", "Robust", "PID"), compare("H7", "Robust", "PID")],
        "iii": [compare("H4", "Robust", "PIDtuned+ILC-AW")],
        "iv": [compare("H4", "Robust", "PID"), compare("H5", "Robust", "PID")],
        "v": [compare(c, "rMAACC", "PID") for c in CONDS],
        "extra": [compare(c, "Ours", "PID") for c in CONDS] + [compare(c, "PIDtuned+ILC-AW", "PID") for c in CONDS],
    }
    (RES / "paper1_heldout2_tables.json").write_text(json.dumps({"tables": T, "questions": Q}, indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, cond in zip(axes, CONDS):
        vals = [T[cond][c]["IAE_SO5_test"] for c in CTRL]
        ax.bar(range(len(CTRL)), vals, color=[COL[c] for c in CTRL])
        ax.set_yscale("log")
        for j, (c, v) in enumerate(zip(CTRL, vals)):
            f = T[cond][c]["failed"]
            if f:
                ax.text(j, 0.02, f"{f}/{T[cond][c]['n']}\nfailed" if v == v else "all\nfailed",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.3, color="#b03a2e")
        ax.set_xticks(range(len(CTRL)), [LABEL[c] for c in CTRL], rotation=70, ha="right", fontsize=7)
        ax.set_title(CONDS[cond], loc="left", fontsize=9)
    axes[0].set_ylabel("oxygen IAE, days 7-14")
    fig.tight_layout()
    fig.savefig(OUT / "heldout2_oxygen.png", dpi=160)
    plt.close(fig)

    for cond in CONDS:
        print(cond)
        for c in CTRL:
            v = T[cond][c]
            print(f"  {c:16s} O {v['IAE_SO5_test']:.5f} N {v['IAE_SNO2_test']:.5f} dev {v['DEVmax_SO5_test']:.3f} "
                  f"mv {v['KLa_move_per_day']:.0f} lim {v['at_limit_KLa']:.2f} gate {v['gated_share']:.2f} "
                  f"fail {v['failed']}/{v['n']}")
    for q, items in Q.items():
        for x in items:
            print(f"({q}) {x}")


if __name__ == "__main__":
    main()
