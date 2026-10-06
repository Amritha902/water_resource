"""Does the learned plant model buy sample efficiency?

The comparison holds the number of *real* interactions fixed and varies only
how many synthetic batches the twin supplies per real batch.  That is the
claim being tested: the reproduction needs roughly fifteen simulated weeks to
converge against the seven days Du et al. allow, and on a real plant those
weeks are spent discharging illegal effluent, so the extra experience has to
come from a model.

Three arms:

``dyna=0``                model-free, the reference
``dyna=3``                three synthetic batches per real one
``dyna=3 + online dual``  the same, with the Lagrange multiplier stepped
                          inside the episode

The third arm exists because of what the first run of this experiment showed:
with ``dyna=3`` the policy gets about four times the gradient steps per real
interaction and learns the energy objective much faster, but a multiplier
stepped only at the episode boundary cannot keep up, and the agent runs the
ammonium limit far past its budget until the dual catches up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wwtp.bsm1 import influent
from wwtp.rl.env import AerationEnvConfig
from wwtp.rl.panda_rl import PandaRLAgent, PandaRLConfig
from wwtp.rl.rewards import RewardSpec
from wwtp.rl.train import train
from wwtp.rl.twin import load_aeration_twin

ENERGY_ONLY = RewardSpec("energy", lambda info: -info["aeration_fraction"])

ARMS = (
    ("dyna=0", dict(dyna_ratio=0, lambda_every=0)),
    ("dyna=3", dict(dyna_ratio=3, lambda_every=0)),
    ("dyna=3+online-dual", dict(dyna_ratio=3, lambda_every=20,
                                lambda_lr_online=0.05)),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weather", default="dry")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    series = influent.canonical_scenario(args.weather)
    cfg = AerationEnvConfig(obs_mode="augmented", action_interval=20,
                            action_mode="absolute")

    rows, curves = [], {}
    for name, overrides in ARMS:
        for seed in range(args.seeds):
            twin = (load_aeration_twin(art / "aeration_twin.pt")
                    if overrides["dyna_ratio"] else None)
            agent = PandaRLAgent(
                PandaRLConfig(seed=seed, n_obs=4, gamma=0.99, ou_sigma=0.35,
                              ou_sigma_final=0.10, warmup_steps=400,
                              buffer_size=80000, **overrides),
                forecaster=str(art / "forecaster.pt"), twin=twin)
            print(f"\n=== {name}  seed {seed}  "
                  f"{args.episodes} training weeks ===", flush=True)
            t0 = time.time()
            hist = train(agent, series, ENERGY_ONLY, cfg=cfg,
                         episodes=args.episodes, eval_every=2, seed=seed)
            final = [h for h in hist["history"] if "eval_AE" in h][-1]
            row = {"arm": name, "seed": seed, "weather": args.weather,
                   "episodes": args.episodes,
                   "wall_seconds": round(time.time() - t0, 1),
                   "lambda": agent.lmbda.tolist()}
            row.update({k[5:]: v for k, v in final.items()
                        if k.startswith("eval_")})
            rows.append(row)
            curves[f"{name}|s{seed}"] = hist["history"]
            print(f"  AE={row['AE']:.1f} EQ={row['EQ']:.0f} "
                  f"DO={row['S_O5_mean']:.3f} NH={row['S_NH']:.2f} "
                  f"violNH={row['viol_NH']:.3f} "
                  f"lambda={np.round(agent.lmbda, 2).tolist()} "
                  f"[{row['wall_seconds']:.0f}s]", flush=True)

            (out / "summary_dyna.json").write_text(
                json.dumps(rows, indent=2, default=float))
            (out / "curves_dyna.json").write_text(
                json.dumps(curves, default=float))

    print("\n| arm | AE | NH>4 | mean S_O,5 | lambda |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['arm']} | {r['AE']:.1f} | {100 * r['viol_NH']:.1f}% | "
              f"{r['S_O5_mean']:.2f} | {np.round(r['lambda'], 2).tolist()} |")


if __name__ == "__main__":
    main()
