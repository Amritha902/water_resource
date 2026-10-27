"""Paper 3 (IOQL) on the official BSM1 plant: Tables I-IV and Figs. 4-7.

Fixed (r = 2) and dynamic (eq. 33) reference; dry, rain, storm; PID prior
alone, IOQL (omega = 1), and IOQL with priors K1-K3 (omega = 0.99^k).  Q_a at
the BSM1 default.  The plant is first run 14 days on dry weather under the
PID prior at r = 2.  Metrics (eq. 32) over the 14 days.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.ioql.ioql import IOQL, reference  # noqa: E402
from wwtp.official.plant import DT_DAYS, OfficialBSM1  # noqa: E402

QA = 55338.0
_WARM = {}


def warm():
    if "st" not in _WARM:
        p = OfficialBSM1("dry")
        c = IOQL(learn=False)
        c.A[:] = 0.0                                  # prior only
        x = p.measurement()[0]
        for _ in range(p.n - 1):
            x = p.step(np.array([c(x, 2.0), QA]))[0]
        _WARM["st"], _WARM["u"] = p.get_state(), float(c.pid.u[0])
    return _WARM["st"], _WARM["u"]


def run(job):
    weather, dynamic, method, seed = job
    st, u0 = warm()
    p = OfficialBSM1(weather)
    p.set_state(st)
    if method == "PID" or method.startswith("PIDx"):
        c = IOQL(seed=seed, u0=u0, learn=False)
        c.A[:] = 0.0
        if method.startswith("PIDx"):            # retuned prior: gains scaled
            g = float(method[4:])
            c.pid.kp, c.pid.ki, c.pid.kd = c.pid.kp * g, c.pid.ki * g, c.pid.kd * g
    else:
        prior = method.split("-")[1] if "-" in method else "default"
        c = IOQL(seed=seed, prior=prior, omega_decay=prior != "default", u0=u0)
    x = p.measurement()[0]
    E, X, R = [], [], []
    t0 = time.process_time()
    for k in range(p.n - 1):
        r = reference(k * DT_DAYS, dynamic)
        u = c(x, r)
        x = p.step(np.array([u, QA]))[0]
        r_next = reference((k + 1) * DT_DAYS, dynamic)
        E.append(x - r_next)
        X.append(x)
        R.append(r_next)
    cpu = time.process_time() - t0
    E = np.array(E)
    return dict(weather=weather, dynamic=dynamic, method=method, seed=seed,
                IAE=float(np.mean(np.abs(E))), ISE=float(np.mean(E ** 2)),
                DEVmax=float(np.max(np.abs(E))), cpu_s=cpu,
                finite=bool(np.all(np.isfinite(X)))), (np.array(X[::4]), np.array(R[::4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="ioql")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="run only these methods (e.g. PIDx3), for all cases, one seed")
    ap.add_argument("--weathers", nargs="*", default=["dry", "rain", "storm"])
    ap.add_argument("--fixed-only", action="store_true")
    a = ap.parse_args()
    if a.methods:
        jobs = [(w, dyn, m, 0) for dyn in ((False,) if a.fixed_only else (False, True))
                for w in a.weathers for m in a.methods]
        return _go(jobs, a)
    jobs = []
    for dyn in (False, True):
        for w in ("dry", "rain", "storm"):
            jobs.append((w, dyn, "PID", 0))
            for m in ("IOQL", "IOQL-K1", "IOQL-K2", "IOQL-K3"):
                jobs += [(w, dyn, m, s) for s in range(a.seeds)]
    return _go(jobs, a)


def _go(jobs, a):
    rows, traces = [], {}
    with Pool(a.workers) as pool:
        for i, (row, (X, R)) in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.append(row)
            if row["seed"] == 0:
                traces[f"{row['weather']}|{'dynamic' if row['dynamic'] else 'fixed'}|{row['method']}"] = {
                    "x": X.tolist(), "r": R.tolist()}
            print(f"[{i}/{len(jobs)}] {row['weather']} dyn={row['dynamic']} {row['method']} s{row['seed']} "
                  f"IAE={row['IAE']:.3e} ISE={row['ISE']:.3e} cpu={row['cpu_s']:.0f}s", flush=True)
            if i % 5 == 0 or i == len(jobs):
                (ROOT / "results" / f"summary_{a.tag}.json").write_text(json.dumps(rows, indent=1))
                (ROOT / "results" / f"traces_{a.tag}.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
