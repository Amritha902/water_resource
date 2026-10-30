"""Paper 1 task: the PID prior with daily iterative learning control (ILC).

Controllers, each over the full 14 days of every weather (and dry with the
paper's actuator noise), from the common warm-up state:

* PID          the paper's PID prior
* PID+ILC      the paper's PID with the daily ILC feedforward
* PIDx3        the retuned prior (gains x3)
* PIDx3+ILC    the retuned prior with the daily ILC feedforward

The ILC learns online, from zero, during the run (see wwtp/sacil/ilc.py);
its settings were chosen on dry weather only (experiments/28_sac_il_paper1.py).

    python experiments/29_ilc_paper1.py
"""

from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.sacil.ilc import DailyILC  # noqa: E402
from wwtp.sacil.sacil import Paper1Env, evaluate, expert_action  # noqa: E402

ILC = dict(beta=[0.5, 0.0], lead=[4, 0], smooth=15)
CONTROLLERS = {"PID": (1, False), "PID+ILC": (1, True), "PIDx3": (3, False), "PIDx3+ILC": (3, True)}
NOISE_RUNS = 10


def warm():
    return pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())


def run(job):
    weather, noise, name, seed = job
    gain, use_ilc = CONTROLLERS[name]
    st, u0 = warm()
    # a plain PID with this gain in the loop and no learned correction (so nothing is bounded)
    env = Paper1Env(weather, st, u0, noise=noise, seed=seed, ilc=DailyILC(**ILC) if use_ilc else None,
                    base_gain=(gain, gain), act_mask=(0.0, 0.0))
    out = evaluate(env, lambda o, e: np.zeros(2))
    if use_ilc:
        out["ilc_profile_KLa"] = env.ilc.f[::8, 0].tolist()
    return dict(weather=weather, noise=noise, controller=name, seed=seed, **out)


def main():
    jobs = [(w, False, c, 0) for w in ("dry", "rain", "storm") for c in CONTROLLERS]
    jobs += [("dry", True, c, s) for c in CONTROLLERS for s in range(NOISE_RUNS)]
    rows, traces = [], {}
    with Pool(4) as pool:
        for r in pool.imap_unordered(run, jobs):
            tr = r.pop("_trace")
            if r["seed"] == 0:
                traces[f"{r['weather']}|{'noise' if r['noise'] else 'clean'}|{r['controller']}"] = tr
            rows.append(r)
            print(f"{r['weather']:5s} noise={r['noise']!s:5s} {r['controller']:10s} seed {r['seed']:2d}: "
                  f"IAE O {r['IAE_SO5']:.5f} N {r['IAE_SNO2']:.5f}", flush=True)
    (ROOT / "results" / "summary_ilc_paper1.json").write_text(json.dumps(rows, indent=1))
    (ROOT / "results" / "traces_ilc_paper1.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
