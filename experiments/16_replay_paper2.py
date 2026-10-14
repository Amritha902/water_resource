"""Re-train one paper-2 agent and record its evaluation week in full.

Uses the exact construction and training code of
``06_aeration_benchmark.py`` (``make`` / ``train`` / ``evaluate``), so the
summary it prints is a reproducibility check against the committed
``results/summary_aeration.json``.  The only difference is the trace
resolution: one sample per decision instead of one per 20 decisions, so the
website can draw the trajectories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wwtp.bsm1 import influent
from wwtp.rl.env import AerationEnvConfig
from wwtp.rl.train import evaluate, train

_spec = importlib.util.spec_from_file_location(
    "bench", ROOT / "experiments" / "06_aeration_benchmark.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method")
    ap.add_argument("--weather", default="dry")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/site/aeration")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    series = influent.canonical_scenario(args.weather)
    agent, reward, cfg, trainable = bench.make(args.method, args.seed,
                                               ROOT / "artifacts")
    curves = None
    if trainable:
        curves = train(agent, series, reward, cfg=cfg, episodes=args.episodes,
                       eval_every=5, seed=args.seed)["history"]
    eval_cfg = cfg if trainable else AerationEnvConfig(action_interval=1)
    stride = 1 if trainable else 20          # one sample per 15 min either way
    res = evaluate(agent, series, reward, cfg=eval_cfg, seed=args.seed,
                   record_stride=stride)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blob = {"method": args.method, "weather": args.weather, "seed": args.seed,
            "episodes": args.episodes, "summary": res["summary"],
            "trace": res["trace"], "curves": curves}
    path = out / f"replay_{args.weather}_{args.method}_s{args.seed}.json"
    path.write_text(json.dumps(blob, default=float))
    s = res["summary"]
    print(f"{args.method}: AE={s['AE']:.1f} DO={s['S_O5_mean']:.3f} "
          f"violNH={s['viol_NH']:.3f} violTN={s['viol_TN']:.3f} -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
