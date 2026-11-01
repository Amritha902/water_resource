"""Paper 1: a tuned PID + ILC made robust to sensor noise, tuned on dry weather week 1 only.

The held-out test showed that every high-gain controller fails with the
official BSM1 sensors (noise 0.25 g/m3).  Here the tuned PID + ILC
(experiments/38) gets two changes, chosen on dry days 0-7 with the official
sensor models (noise seeds 100 and 101, never used in any test):

* a first-order filter on the oxygen measurement the controller uses, time
  constant tau in {0, 0.75, 1.5, 3} min;
* the PID gains of each loop scaled: oxygen g_O in {0.125, 0.25, 0.5, 1},
  nitrate g_N in {0.25, 0.5, 1}.

A first search (results/paper1_robust_pid_v1.json) filtered both loops with
one gain scale; filtering the nitrate loop, whose sensor already lags by
10 min, made it oscillate even with ideal sensors, so the search was redone
as above (training data only, before held-out test 2).

The ILC (experiments/38's settings) gets anti-windup.  Rule, fixed before
the run: among settings with no failure in either noise seed and whose
oxygen error with ideal sensors is at most twice that of the tuned PID +
ILC, the lowest mean score with sensors, score = IAE_SO5 / 0.004 +
IAE_SNO2 / 0.002 (the learners' selection score).

Writes results/paper1_robust_pid.json.

    python experiments/42_robust_pid.py
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

TP = json.loads((ROOT / "results" / "paper1_tuned_pid.json").read_text())
TAUS, GO, GN = (0.0, 0.75, 1.5, 3.0), (0.125, 0.25, 0.5, 1.0), (0.25, 0.5, 1.0)
SEEDS = (100, 101)
score = lambda r: r["IAE_SO5"] / 0.004 + r["IAE_SNO2"] / 0.002


def run(job):
    tau, g, sensors = job
    st, u0 = pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())
    env = Paper1Env("dry", st, u0, days=7.0, act_mask=(0.0, 0.0), terms=np.array(TP["terms"]) * np.array(g),
                    ilc=DailyILC(**{**TP["ilc"], "antiwindup": True}), sensors=sensors,
                    meas_filter=(tau, 0.0))
    r = evaluate(env, lambda o, e: np.zeros(2))
    return dict(tau=tau, g=g, sensors=sensors, IAE_SO5=r["IAE_SO5"], IAE_SNO2=r["IAE_SNO2"],
                DEVmax_SO5=r["DEVmax_SO5"], KLa_move_per_day=r["KLa_move_per_day"], fail=r["fail"])


def main():
    grid = [(t, (go, gn)) for t, go, gn in itertools.product(TAUS, GO, GN)]
    jobs = [(t, g, s) for t, g in grid for s in SEEDS + (None,)]
    with Pool(4) as pool:
        res = pool.map(run, jobs)
    ref = run((0.0, (1.0, 1.0), None))
    rows = []
    for t, g in grid:
        noisy = [r for r in res if r["tau"] == t and r["g"] == g and r["sensors"] is not None]
        ideal = next(r for r in res if r["tau"] == t and r["g"] == g and r["sensors"] is None)
        ok = not any(r["fail"] for r in noisy) and not ideal["fail"] and ideal["IAE_SO5"] <= 2 * ref["IAE_SO5"]
        rows.append(dict(tau=t, g=g, admissible=ok,
                         score_sensors=float(np.mean([score(r) for r in noisy])) if not any(r["fail"] for r in noisy) else None,
                         IAE_SO5_sensors=float(np.mean([r["IAE_SO5"] for r in noisy])),
                         IAE_SNO2_sensors=float(np.mean([r["IAE_SNO2"] for r in noisy])),
                         failures=sum(r["fail"] for r in noisy), IAE_SO5_ideal=ideal["IAE_SO5"],
                         IAE_SNO2_ideal=ideal["IAE_SNO2"]))
        print(rows[-1], flush=True)
    best = min((r for r in rows if r["admissible"]), key=lambda r: r["score_sensors"])
    print("chosen:", best, flush=True)
    out = dict(train="dry weather, days 0-7, official sensors (noise seeds 100, 101) and ideal sensors",
               rule="no failure; ideal-sensor oxygen IAE <= 2x tuned PID + ILC; lowest mean score with sensors",
               reference_ideal=ref, grid=rows, tau=best["tau"], g=best["g"],
               terms=(np.array(TP["terms"]) * np.array(best["g"])).tolist(), ilc={**TP["ilc"], "antiwindup": True})
    (ROOT / "results" / "paper1_robust_pid.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
