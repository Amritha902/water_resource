"""Train the probabilistic influent forecaster."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.forecast.train import train_forecaster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--scenarios", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = train_forecaster(args.out, n_scenarios=args.scenarios,
                            epochs=args.epochs, seed=args.seed)
    best = min(info["history"], key=lambda h: h["val_pinball"])
    print(f"\nwindows={info['n_windows']}  best val pinball={best['val_pinball']:.4f}"
          f"  regime acc={best['val_regime_acc']:.3f}")


if __name__ == "__main__":
    main()
