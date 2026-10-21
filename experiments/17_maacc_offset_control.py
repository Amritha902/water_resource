"""Control experiment: is MAACC's nitrate gain learned control?

Under the paper's noise protocol MAACC improves IAE on S_NO,2 by 23-29 %
against the PID prior (results/summary_paper1.json).  Its learned term a_2
spends most of the run pinned at its clip, -10 % of the Q_a range
(-9 223 m3/d), so we test two learning-free controllers:

``PID+offset``   the PID prior plus a *constant* (0, -9 223), i.e. MAACC with
                 its learned term frozen at the clip;
``PID-clamp``    the PID prior with its internal Q_a state allowed down to
                 -9 223 instead of 0 (the plant still clips the applied flow
                 at 0).  The velocity-form PID is shift-invariant, so this is
                 the same controller as PID+offset up to the upper clamp.

Same harness, noise, seeds and burn-in as ``03_run_control.py``.  Writes
``results/summary_offset_control.json``.
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

from wwtp.baselines.pid_controller import PIDController
from wwtp.bsm1 import influent
from wwtp.envs import run_closed_loop

ACT = np.array([0.03 * 240.0, 0.03 * 92230.0])
SENSOR = np.array([0.05, 0.10])
OFFSET = np.array([0.0, -0.1 * 92230.0])


class OffsetPID(PIDController):
    """The PID prior plus a constant additive term (a frozen MAACC a_k)."""

    def __init__(self, offset):
        super().__init__()
        self.offset = np.asarray(offset, float)

    def act(self, k, s, r, context):
        return self.pid(r - s) + self.offset


def _job(args):
    name, weather, seed = args
    if name == "PID+offset":
        ctrl = OffsetPID(OFFSET)
    elif name == "PID-clamp":
        ctrl = PIDController(u_min=(0.0, OFFSET[1]))
    else:
        ctrl = PIDController()
    res = run_closed_loop(ctrl, influent.canonical_scenario(weather),
                          actuator_noise_std=ACT, sensor_noise_std=SENSOR,
                          seed=100 + seed)
    row = res.summary(burn_in_days=1.0)
    row.update(controller=name, weather=weather, seed=seed, condition="noisy",
               frac_qa_at_zero=float(np.mean(res.control[:, 1] <= 1.0)))
    print(f"{weather:5s} {name:10s} s{seed}  IAE_SNO2={row['IAE_SNO2']:.4f} "
          f"IAE_SO5={row['IAE_SO5']:.4f}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/summary_offset_control.json")
    args = ap.parse_args()
    jobs = [(n, w, s) for w in ("dry", "rain", "storm")
            for n in ("PID", "PID+offset", "PID-clamp") for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    out = ROOT / args.out
    out.write_text(json.dumps(rows, indent=2, default=float))

    ref = json.loads((ROOT / "results/summary_paper1.json").read_text())
    print("\nIAE S_NO,2, mean over seeds (noisy condition)")
    for w in ("dry", "rain", "storm"):
        line = f"  {w:5s}"
        for name, src in (("PID", rows), ("MAACC", ref), ("PID+offset", rows), ("PID-clamp", rows)):
            v = [r["IAE_SNO2"] for r in src if r["weather"] == w and r["controller"] == name
                 and r.get("condition", "noisy") == "noisy"]
            line += f"  {name} {np.mean(v):.4f}"
        print(line)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
