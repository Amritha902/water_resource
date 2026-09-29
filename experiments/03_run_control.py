"""Closed-loop benchmark: PID vs MAACC vs PANDA (+ ablations).

Three operating conditions are evaluated on the three canonical BSM1 weather
profiles:

``clean``    no instrumentation error; the idealised setting of the paper.
``noisy``    the paper's supplementary protocol -- zero-mean white noise with
             a standard deviation of 3 % of each actuator's upper bound --
             plus realistic DO / nitrate sensor noise.
``detuned``  the same noise, but the expert prior is a conservatively tuned
             PID (25 % of the published gains), which is what a plant that
             has not been re-tuned in years actually looks like.  This is the
             condition in which a learned correction has something to add.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.baselines.pid import PAPER_KD, PAPER_KI, PAPER_KP
from wwtp.baselines.pid_controller import PIDController
from wwtp.bsm1 import influent
from wwtp.envs import run_closed_loop
from wwtp.maacc.agent import MAACCController
from wwtp.panda.controller import PANDAController

WEATHERS = ("dry", "rain", "storm")

CONDITIONS = {
    "clean": dict(gain=1.0, actuator=None, sensor=None),
    "noisy": dict(gain=1.0,
                  actuator=np.array([0.03 * 240.0, 0.03 * 92230.0]),
                  sensor=np.array([0.05, 0.10])),
    "detuned": dict(gain=0.25,
                    actuator=np.array([0.03 * 240.0, 0.03 * 92230.0]),
                    sensor=np.array([0.05, 0.10])),
}


def make_controller(name: str, gain: float, seed: int, art: Path):
    kp, ki, kd = PAPER_KP * gain, PAPER_KI * gain, PAPER_KD * gain
    fc, tw = art / "forecaster.pt", art / "twin.pt"
    if name == "PID":
        return PIDController(kp=kp, ki=ki, kd=kd)
    if name == "MAACC":
        return MAACCController(seed=seed, kp=kp, ki=ki, kd=kd)
    if name == "PANDA":
        return PANDAController(fc, tw, seed=seed, kp=kp, ki=ki, kd=kd)
    if name == "PANDA-noFF":
        return PANDAController(fc, tw, seed=seed, kp=kp, ki=ki, kd=kd,
                               use_feedforward=False)
    if name == "PANDA-noCtx":
        return PANDAController(fc, tw, seed=seed, kp=kp, ki=ki, kd=kd,
                               use_context=False)
    if name == "PANDA-noGate":
        return PANDAController(fc, tw, seed=seed, kp=kp, ki=ki, kd=kd,
                               use_gate=False)
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--controllers", nargs="*",
                    default=["PID", "MAACC", "PANDA"])
    ap.add_argument("--ablations", action="store_true",
                    help="also run the PANDA ablations under 'noisy'")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    art = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(c, w, n, s) for c in args.conditions for w in WEATHERS
            for n in args.controllers for s in range(args.seeds)]
    if args.ablations:
        jobs += [("noisy", w, n, s) for w in WEATHERS
                 for n in ("PANDA-noFF", "PANDA-noCtx", "PANDA-noGate")
                 for s in range(min(args.seeds, 2))]

    series_cache = {w: influent.canonical_scenario(w) for w in WEATHERS}
    rows, traces = [], {}
    for i, (cond, weather, name, seed) in enumerate(jobs, 1):
        spec = CONDITIONS[cond]
        ctrl = make_controller(name, spec["gain"], seed, art)
        res = run_closed_loop(ctrl, series_cache[weather],
                              actuator_noise_std=spec["actuator"],
                              sensor_noise_std=spec["sensor"],
                              seed=100 + seed)
        row = res.summary(burn_in_days=1.0)
        row.update(condition=cond, controller=name, seed=seed)
        rows.append(row)
        print(f"[{i:3d}/{len(jobs)}] {cond:8s} {weather:6s} {name:12s} s{seed} "
              f"IAE1={row['IAE_SO5']:.5f} IAE2={row['IAE_SNO2']:.5f} "
              f"DEV2={row['DEVmax_SNO2']:.3f} EQ={row['EQ']:.0f} "
              f"OCI={row['OCI']:.0f}", flush=True)

        key = (cond, weather, name)
        if seed == 0 and cond in ("noisy", "detuned"):
            traces[f"{cond}|{weather}|{name}"] = {
                "t": res.time_days[::20].tolist(),
                "s": res.state[::20].tolist(),
                "u": res.control[::20].tolist(),
                "q_in": res.influent_flow[::20].tolist(),
                "nh_eff": res.effluent[::20, 9].tolist(),
                "extras": {k: v for k, v in res.extras.items()},
            }

    (out / f"summary_{args.tag}.json").write_text(json.dumps(rows, indent=2))
    (out / f"traces_{args.tag}.json").write_text(json.dumps(traces))

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out / f"summary_{args.tag}.csv", index=False)
        agg = (df.groupby(["condition", "weather", "controller"])
                 [["IAE_SO5", "ISE_SO5", "DEVmax_SO5", "IAE_SNO2", "ISE_SNO2",
                   "DEVmax_SNO2", "EQ", "OCI", "viol_NH"]]
                 .agg(["mean", "std"]))
        agg.to_csv(out / f"aggregate_{args.tag}.csv")
        print("\n" + agg.round(5).to_string())
    except ImportError:
        pass


if __name__ == "__main__":
    main()
