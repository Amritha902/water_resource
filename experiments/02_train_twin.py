"""Collect excitation rollouts and train the neural digital twin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.twin.train import train_twin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = train_twin(args.out, n_rollouts=args.rollouts, epochs=args.epochs,
                      seed=args.seed)
    best = min(info["history"], key=lambda h: h["val_free_rmse"])
    print(f"\nsequences={info['n_sequences']}  best free-running RMSE="
          f"{best['val_free_rmse']:.4f} mg/L")


if __name__ == "__main__":
    main()
