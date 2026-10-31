"""Paper 1, a fairly tuned PID baseline: P and I gains tuned separately per loop.

The PID x3 of the study multiplies all of the paper's gains by one factor.  A
reviewer can ask whether a properly tuned PID would do as well, so here the
PID is tuned term by term, with a stated budget, on dry weather week 1 only
(days 0-7), like everything else.

Budget and rule, fixed before the run:
1. Oxygen loop first (nitrate loop at x3): P multiplier and I multiplier each
   in {1, 2, 3, 4, 6, 8}, D multiplier = P multiplier (the paper's D/P ratio
   kept): 36 settings.
2. Stability margin: a setting is admissible only if the same setting with
   all its multipliers x1.5 still works (no failure, and that loop's IAE at
   most twice the setting's own).  The study's x3 passes this rule.
3. The admissible setting with the lowest IAE of that loop is chosen.
4. Then the nitrate loop the same way, with the oxygen loop at its choice.
5. Daily ILC re-tuned for the tuned PID, with the grid and rule of
   experiments/32_paper1_tuning.py (oxygen loop, bumpless; lowest oxygen IAE
   among settings whose largest oxygen deviation is no larger than the tuned
   PID's alone).

Writes results/paper1_tuned_pid.json.

    python experiments/38_tuned_pid.py
"""

from __future__ import annotations

import itertools
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.sacil.ilc import DailyILC  # noqa: E402
from wwtp.sacil.sacil import Paper1Env, evaluate  # noqa: E402

TRAIN_DAYS = 7.0
MULTS = (1, 2, 3, 4, 6, 8)
MARGIN = 1.5
BETAS, LEADS, SMOOTHS = (0.25, 0.5, 0.75, 1.0), (1, 2, 4, 8, 16), (5, 15, 30)


def warm():
    return pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())


def run(job):
    terms, ilc = job
    st, u0 = warm()
    env = Paper1Env("dry", st, u0, days=TRAIN_DAYS, act_mask=(0.0, 0.0), terms=terms,
                    ilc=None if ilc is None else DailyILC(**ilc))
    r = evaluate(env, lambda o, e: np.zeros(2))
    return dict(terms=np.asarray(terms).tolist(), ilc=ilc, IAE_SO5=r["IAE_SO5"], IAE_SNO2=r["IAE_SNO2"],
                DEVmax_SO5=r["DEVmax_SO5"], fail=r["fail"])


def terms_for(loop, kp, ki, other):
    t = np.array(other, float).copy()
    t[:, loop] = (kp, ki, kp)
    return t


def tune_loop(pool, loop, other):
    key = ("IAE_SO5", "IAE_SNO2")[loop]
    grid = list(itertools.product(MULTS, MULTS))
    base = pool.map(run, [(terms_for(loop, p, i, other), None) for p, i in grid])
    pushed = pool.map(run, [(terms_for(loop, MARGIN * p, MARGIN * i, other), None) for p, i in grid])
    rows = []
    for (p, i), b, m in zip(grid, base, pushed):
        ok = not b["fail"] and not m["fail"] and m[key] <= 2 * b[key]
        rows.append(dict(kp=p, ki=i, IAE=b[key], IAE_other=b[("IAE_SNO2", "IAE_SO5")[loop]],
                         IAE_at_margin=m[key], DEVmax_SO5=b["DEVmax_SO5"], admissible=ok))
    best = min((r for r in rows if r["admissible"]), key=lambda r: r["IAE"])
    return rows, best


def main():
    out = {"train": "dry weather, days 0-7", "multipliers": MULTS, "margin": MARGIN,
           "rule": "lowest IAE of the loop among settings still working with all multipliers x1.5 "
                   "(no failure, IAE at most twice); D multiplier = P multiplier"}
    x3 = np.full((3, 2), 3.0)
    with Pool(4) as pool:
        rows_o, best_o = tune_loop(pool, 0, x3)
        print("oxygen loop:", best_o, f"{sum(r['admissible'] for r in rows_o)}/36 admissible", flush=True)
        after_o = terms_for(0, best_o["kp"], best_o["ki"], x3)
        rows_n, best_n = tune_loop(pool, 1, after_o)
        print("nitrate loop:", best_n, f"{sum(r['admissible'] for r in rows_n)}/36 admissible", flush=True)
        terms = terms_for(1, best_n["kp"], best_n["ki"], after_o)
        out.update(oxygen_grid=rows_o, nitrate_grid=rows_n, terms=terms.tolist())

        alone = run((terms, None))
        x3run = run((x3, None))
        out["tuned_alone"], out["x3_alone"] = alone, x3run
        print(f"tuned PID dry week 1: O {alone['IAE_SO5']:.5f} N {alone['IAE_SNO2']:.5f} "
              f"(PID x3: O {x3run['IAE_SO5']:.5f} N {x3run['IAE_SNO2']:.5f})", flush=True)

        grid = [dict(beta=[b, 0.0], lead=[l, 0], smooth=w, bumpless=True)
                for b, l, w in itertools.product(BETAS, LEADS, SMOOTHS)]
        res = pool.map(run, [(terms, g) for g in grid])
        ok = [r for r in res if not r["fail"] and r["DEVmax_SO5"] <= alone["DEVmax_SO5"]]
        out["ilc_grid"], out["ilc_admissible"] = res, len(ok)
        if ok:
            best = min(ok, key=lambda r: r["IAE_SO5"])
            out["ilc"] = best["ilc"]
            print(f"ILC: {len(ok)}/{len(res)} admissible; best {best['ilc']} O {best['IAE_SO5']:.5f}", flush=True)
        else:
            out["ilc"] = None
            print("ILC: no admissible setting", flush=True)
    (ROOT / "results" / "paper1_tuned_pid.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
