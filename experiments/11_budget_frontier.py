"""The energy / violation frontier.

A single operating point cannot settle whether letting dissolved oxygen float
actually saves anything, because the saving and the discharge-limit violation
rate move together.  Du et al. report a 5-7 % aeration saving; our
reproduction gets 5.8-7.8 % on dry weather, but at 30-46 % of the week above
the 4 g N/m3 ammonium limit, against 12 % for the PID comparator at a fixed
2 mg/L set point.  The saving is real and so is its cost.

A constrained formulation can answer the question a fixed penalty weight
cannot: *at a violation rate the operator is willing to accept, how much
energy is actually available?*  This script sweeps the ammonium budget and
traces the frontier, with PID and DDPG-B placed on the same axes.

Each arm is one PANDA-RL run with its ammonium budget set to the sweep value;
the Lagrange multiplier finds the weight that delivers it.
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
from wwtp.rl.env import NH_LIMIT, TN_LIMIT, AerationEnvConfig
from wwtp.rl.panda_rl import Constraint, PandaRLAgent, PandaRLConfig
from wwtp.rl.rewards import RewardSpec
from wwtp.rl.train import train

ENERGY_ONLY = RewardSpec("energy", lambda info: -info["aeration_fraction"])
DEFAULT_BUDGETS = (0.05, 0.15, 0.25, 0.40)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weather", default="dry")
    ap.add_argument("--budgets", type=float, nargs="*",
                    default=list(DEFAULT_BUDGETS))
    ap.add_argument("--tn-budget", type=float, default=0.45,
                    help="total-nitrogen budget, default is PID's own rate")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="frontier")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    series = influent.canonical_scenario(args.weather)
    cfg = AerationEnvConfig(obs_mode="augmented", action_interval=20,
                            action_mode="absolute")

    rows: list[dict] = []
    for budget in args.budgets:
        for seed in range(args.seeds):
            constraints = (
                Constraint("NH_peak", NH_LIMIT, budget, "S_NH,e"),
                Constraint("TN_peak", TN_LIMIT, args.tn_budget, "N_tot,e"))
            agent = PandaRLAgent(
                PandaRLConfig(seed=seed, n_obs=4, gamma=0.99, ou_sigma=0.35,
                              ou_sigma_final=0.05, warmup_steps=400,
                              buffer_size=80000, constraints=constraints,
                              lambda_every=20, lambda_lr_online=0.05),
                forecaster=str(art / "forecaster.pt"))
            print(f"\n=== NH budget {100 * budget:.0f}%  seed {seed} ===",
                  flush=True)
            t0 = time.time()
            hist = train(agent, series, ENERGY_ONLY, cfg=cfg,
                         episodes=args.episodes, eval_every=5, seed=seed)
            final = [h for h in hist["history"] if "eval_AE" in h][-1]
            row = {"weather": args.weather, "nh_budget": budget, "seed": seed,
                   "episodes": args.episodes,
                   "wall_seconds": round(time.time() - t0, 1),
                   "lambda": agent.lmbda.tolist()}
            row.update({k[5:]: v for k, v in final.items()
                        if k.startswith("eval_")})
            rows.append(row)
            print(f"  AE={row['AE']:.1f} NH>4={100 * row['viol_NH']:.1f}% "
                  f"(budget {100 * budget:.0f}%) DO={row['S_O5_mean']:.2f} "
                  f"lambda={np.round(agent.lmbda, 2).tolist()} "
                  f"[{row['wall_seconds']:.0f}s]", flush=True)
            (out / f"summary_{args.tag}.json").write_text(
                json.dumps(rows, indent=2, default=float))

    print("\n| NH budget | achieved NH>4 | AE | mean S_O,5 | lambda NH |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {100 * r['nh_budget']:.0f}% | {100 * r['viol_NH']:.1f}% | "
              f"{r['AE']:.1f} | {r['S_O5_mean']:.2f} | {r['lambda'][0]:.1f} |")


if __name__ == "__main__":
    main()
