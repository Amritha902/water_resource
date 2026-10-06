"""Simulate the rollouts for the effluent-ammonium early-warning model.

Train, validation and test rollouts use disjoint seeds, so every reported
number is on plants and influent the model has never seen.  The canonical
BSM1 dry / rain / storm profiles are simulated separately at three fixed
operating points as a further held-out check.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.bsm1.influent import canonical_scenario
from wwtp.warning.dataset import DT_SECONDS, OperatingPoint, collect

SPLITS = {"train": (0, 96), "val": (1000, 16), "test": (2000, 24)}
#: fixed stress levels for the canonical profiles
CANONICAL_OPS = {
    "nominal": OperatingPoint(240.0, 15.0, 1.0),
    "trimmed": OperatingPoint(200.0, 14.0, 1.2),
    "cold": OperatingPoint(220.0, 12.5, 1.5),
}


def _job(args):
    seed, days, fc = args
    import torch
    torch.set_num_threads(1)
    return collect(seed, days=days, forecaster=fc)


def _canon(args):
    weather, op_name, fc = args
    import torch
    torch.set_num_threads(1)
    series = canonical_scenario(weather, dt_seconds=DT_SECONDS)
    roll = collect(7, forecaster=fc, op=CANONICAL_OPS[op_name], series=series)
    roll["name"] = f"{weather}|{op_name}"
    return roll


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/warning_rollouts.pkl")
    ap.add_argument("--forecaster", default="artifacts/forecaster.pt")
    ap.add_argument("--days", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the number of rollouts per split")
    args = ap.parse_args()

    out = {}
    with Pool(args.workers) as pool:
        for split, (base, n) in SPLITS.items():
            n = max(2, int(round(n * args.scale)))
            jobs = [(base + i, args.days, args.forecaster) for i in range(n)]
            out[split] = pool.map(_job, jobs)
            print(f"{split}: {n} rollouts", flush=True)
        jobs = [(w, o, args.forecaster) for w in ("dry", "rain", "storm")
                for o in CANONICAL_OPS]
        out["canonical"] = pool.map(_canon, jobs)
        print(f"canonical: {len(jobs)} rollouts", flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
