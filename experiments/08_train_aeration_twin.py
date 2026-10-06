"""Collect open-loop rollouts and fit the one-step aeration model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.rl.twin import train_aeration_twin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = train_aeration_twin(args.out, n_rollouts=args.rollouts,
                               epochs=args.epochs, seed=args.seed)
    best = min(info["history"], key=lambda h: h["val"])
    print(f"\ntransitions={info['n_transitions']}  "
          f"best val DO MAE={best['val_DO_mae']:.4f} mg/L  "
          f"NH MAE={best['val_NH_mae']:.4f} mg/L")


if __name__ == "__main__":
    main()
