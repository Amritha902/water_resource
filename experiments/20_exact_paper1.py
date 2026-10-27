"""Paper 1 replicated as printed, on the official BSM1 plant and influent.

Plant: ``bsm2-python`` (port of the IWA benchmark code); influent: the official
dry and rain files (rain repaired, see ``wwtp/official/plant.py``); storm: the
official file via ``wwtp/official/plant.py``.  Controller: ``wwtp/maacc/exact.py`` with Table I values.

Protocol: the plant is first run 14 days on dry weather under the PID from the
benchmark steady state; every controller then starts from that state and is
evaluated over the full 14 days of the weather file (45 s steps).  Metrics as
eq. (36): IAE = mean |z|, ISE = mean z^2, DEVmax = max |z|.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.baselines.pid import IncrementalPID  # noqa: E402
from wwtp.maacc.exact import ExactMAACC, U_MAX, U_MIN  # noqa: E402
from wwtp.official.plant import OfficialBSM1, U_NOMINAL  # noqa: E402

#: Set points.  The text of paper 1 gives r = [1, 2] for [S_O,5, S_NO,2], but
#: its Figs. 4-6 and 8 show S_O,5 held at 2 and S_NO,2 at 1 (the standard BSM1
#: values), and with [2, 1] the paper's PID row is reproduced (dry oxygen IAE
#: 1.140e-2 vs 1.14e-2).  [1, 2] runs are kept in summary_paper1_exact.json.
R = np.array([2.0, 1.0])
NOISE = 0.03 * U_MAX
_WARM = {}


def warm_state():
    if "st" not in _WARM:
        p = OfficialBSM1("dry")
        pid = IncrementalPID(u_min=U_MIN, u_max=U_MAX, u0=U_NOMINAL.copy())
        s = p.measurement()
        for _ in range(p.n - 1):
            s = p.step(pid(R - s))
        _WARM["st"], _WARM["u"] = p.get_state(), pid.u.copy()
    return _WARM["st"], _WARM["u"]


INITS = [(1, 0.1, 1), (1, 0.1, 100), (1, 10, 1), (1, 10, 100), (10, 10, 100),
         (100, 1, 100), (1, 1, 1000), (1, 100, 100), (1, 1000, 100), (1, 10, 10000),
         (0.01, 10, 100), (0.001, 1, 100)]


def run(job):
    weather, noise, ctrl, seed = job[:4]
    init = job[4] if len(job) > 4 else (1.0, 0.1, 1.0)
    st, u_end = warm_state()
    p = OfficialBSM1(weather)
    p.set_state(st)
    if ctrl == "PID":
        pid = IncrementalPID(u_min=U_MIN, u_max=U_MAX, u0=u_end.copy())
        law = lambda s: pid(R - s)
    else:
        law = ExactMAACC(seed=seed, critic=ctrl.split("-")[1], u0=u_end.copy(), init=init)
        law = (lambda m: (lambda s: m(s, R)))(law)
    rng = np.random.default_rng(1000 + seed)
    s, E, ok = p.measurement(), [], True
    tr = {"s": [], "u": [], "u_ideal": []}
    for k in range(p.n - 1):
        u = law(s)
        if not np.all(np.isfinite(u)):
            ok = False
            u = np.nan_to_num(u, nan=0.0, posinf=1e12, neginf=-1e12)
        u_ideal = np.clip(u, U_MIN, U_MAX)
        if noise:
            u = u + rng.normal(0.0, NOISE)
        u = np.clip(u, U_MIN, U_MAX)
        s = p.step(u)
        E.append(s - R)
        if seed == 0 and k % 4 == 0:
            tr["s"].append(s.tolist()); tr["u"].append(u.tolist()); tr["u_ideal"].append(u_ideal.tolist())
    E = np.array(E)
    out = dict(weather=weather, noise=noise, controller=ctrl, seed=seed, finite=ok,
               init=list(init))
    if seed == 0:
        out["_trace"] = tr
    for i, n in enumerate(("SO5", "SNO2")):
        e = E[:, i]
        out[f"IAE_{n}"], out[f"ISE_{n}"], out[f"DEVmax_{n}"] = (
            float(np.mean(np.abs(e))), float(np.mean(e ** 2)), float(np.max(np.abs(e))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--weathers", nargs="*", default=["dry", "rain", "storm"])
    ap.add_argument("--critics", nargs="*", default=["paper", "td0", "residual"])
    ap.add_argument("--noise", action="store_true", help="also the Table III case (dry + noise)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="exact")
    ap.add_argument("--init-sweep", action="store_true",
                    help="dry, seed 0: every initial-weight scale in INITS, both critic readings")
    a = ap.parse_args()
    if a.init_sweep:
        jobs = [("dry", False, f"MAACC-{c}", 0, i) for c in ("paper", "td0") for i in INITS]
        a.tag = "exact_init_sweep"
        return _go(jobs, a)
    jobs = [(w, False, "PID", 0) for w in a.weathers]
    # paper Fig. 9 compares the ideal and the noisy control input of the same run
    jobs += [(w, False, f"MAACC-{c}", s) for w in a.weathers for c in a.critics for s in range(a.seeds)]
    if a.noise:
        jobs += [("dry", True, "PID", s) for s in range(a.seeds)]
        jobs += [("dry", True, f"MAACC-{c}", s) for c in a.critics for s in range(a.seeds)]
    return _go(jobs, a)


def _go(jobs, a):
    out = ROOT / "results" / f"summary_paper1_{a.tag}.json"
    rows, traces = [], {}
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            tr = r.pop("_trace", None)
            if tr is not None:
                traces[f"{r['weather']}|{'noise' if r['noise'] else 'clean'}|{r['controller']}"] = tr
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['weather']} noise={r['noise']} {r['controller']} s{r['seed']} "
                  f"IAE_O={r['IAE_SO5']:.3e} IAE_NO={r['IAE_SNO2']:.3e} finite={r['finite']}", flush=True)
            if i % 10 == 0 or i == len(jobs):
                out.write_text(json.dumps(rows, indent=1))
                if traces:
                    out.with_name(out.name.replace("summary_", "traces_")).write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
