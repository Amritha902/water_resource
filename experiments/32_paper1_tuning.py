"""Paper 1, new controller, stage A: every setting chosen on dry weather, week 1 only.

The test data (week 2 of every weather, and rain and storm altogether) is not
touched here.

1. PID gain: the paper's PID with all gains x g, g = 1..6, on dry days 0-7.
   The loop is called unstable at the first g whose oxygen IAE is more than
   twice that of g - 1; the chosen gain is the largest whole number below 70 %
   of that g.
2. Daily ILC on the oxygen loop, with the chosen gain: a grid over the share
   beta, the lead and the smoothing; the best IAE on dry days 0-7 is chosen and
   the whole grid is kept to show how sensitive the result is.
3. ILC on the nitrate loop as well, at the chosen oxygen settings.

Writes results/paper1_tuning.json.

    python experiments/32_paper1_tuning.py
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
BETAS, LEADS, SMOOTHS = (0.25, 0.5, 0.75, 1.0), (1, 2, 4, 8, 16), (5, 15, 30)


def warm():
    return pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())


def run(job):
    gain, ilc = job
    st, u0 = warm()
    env = Paper1Env("dry", st, u0, days=TRAIN_DAYS, base_gain=(gain, gain), act_mask=(0.0, 0.0),
                    ilc=None if ilc is None else DailyILC(**ilc))
    r = evaluate(env, lambda o, e: np.zeros(2))
    return dict(gain=gain, ilc=ilc, IAE_SO5=r["IAE_SO5"], IAE_SNO2=r["IAE_SNO2"], fail=r["fail"])


def main():
    out = {"train": "dry weather, days 0-7"}
    with Pool(4) as pool:
        sweep = list(pool.map(run, [(g, None) for g in range(1, 7)]))
        out["gain_sweep"] = sweep
        limit = next((s["gain"] for prev, s in zip(sweep, sweep[1:])
                      if s["fail"] or s["IAE_SO5"] > 2 * prev["IAE_SO5"]), 7)
        gain = int(np.ceil(0.7 * limit) - 1)
        out["gain_limit"], out["gain"] = limit, gain
        print("gain sweep:", [(s["gain"], round(s["IAE_SO5"], 5)) for s in sweep], "-> limit", limit, "chosen", gain)

        grid = [dict(beta=[b, 0.0], lead=[l, 0], smooth=w) for b, l, w in itertools.product(BETAS, LEADS, SMOOTHS)]
        res = list(pool.map(run, [(gain, g) for g in grid]))
        out["ilc_grid"] = res
        best = min((r for r in res if not r["fail"]), key=lambda r: r["IAE_SO5"])
        print("ILC best:", best["ilc"], round(best["IAE_SO5"], 5))

        nit = [dict(beta=[best["ilc"]["beta"][0], bn], lead=[best["ilc"]["lead"][0], ln], smooth=best["ilc"]["smooth"])
               for bn, ln in itertools.product((0.25, 0.5, 1.0), (0, 2, 8))]
        resn = list(pool.map(run, [(gain, g) for g in nit]))
        out["ilc_nitrate"] = resn
        bestn = min(resn + [best], key=lambda r: r["IAE_SO5"] / best["IAE_SO5"] + r["IAE_SNO2"] / best["IAE_SNO2"])
        print("nitrate ILC:", [(r["ilc"]["beta"][1], r["ilc"]["lead"][1], round(r["IAE_SNO2"], 5)) for r in resn],
              "-> chosen", bestn["ilc"])
        out["ilc"] = bestn["ilc"]
    (ROOT / "results" / "paper1_tuning.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
