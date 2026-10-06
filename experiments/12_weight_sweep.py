"""Isolate the forecast and CVaR changes from the Lagrangian machinery.

The budget-swept frontier (``11_budget_frontier.py``) turned out not to be
monotone at one seed and twenty training weeks: a looser ammonium budget
produced *both* more energy and more violations.  The dual is still moving
when training stops, so the evaluated policy is a snapshot of a
non-stationary process, and the budget does not reliably index the operating
point.

This script removes that confound.  Both arms use the paper's own reward,
``-(KLa_5/240 + beta2 max(S_NH,e - 4, 0))``, with no Lagrange multiplier at
all, and ``beta2`` is swept.  A fixed weight is a stationary objective, so
each run has a well-defined target.  The two arms differ *only* in the
network:

``DDPG-B``      the published agent, scalar critic, no forecast
``PANDA-fw``    forecast *and* CVaR -- modifications 1 and 2 together
``PANDA-fcast`` forecast only, scalar critic -- modification 1 alone
``PANDA-cvar``  CVaR only, no forecast -- modification 2 alone

The last two exist because ``PANDA-fw`` enables both changes at once, so a
gap between it and ``DDPG-B`` cannot be attributed to either one.

Sweeping ``beta2`` traces a frontier for each.  If modifications 1 and 2 earn
their place, PANDA-fw's frontier sits below-and-left of DDPG-B's: less energy
at the same violation rate, or fewer violations at the same energy.  That is a
claim about the network, independent of how the trade-off is parameterised.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wwtp.bsm1 import influent
from wwtp.rl.ddpg import DDPGAgent, DDPGConfig
from wwtp.rl.env import NH_LIMIT, AerationEnvConfig
from wwtp.rl.panda_rl import PandaRLAgent, PandaRLConfig
from wwtp.rl.rewards import RewardSpec
from wwtp.rl.train import train

DEFAULT_WEIGHTS = (0.2, 0.42, 1.0, 2.5)


def make_reward(beta2: float) -> RewardSpec:
    """The paper's eq. (23) with beta2 as the swept parameter."""
    def fn(info: dict) -> float:
        return -(info["aeration_fraction"]
                 + beta2 * max(info["S_NH"] - NH_LIMIT, 0.0))
    return RewardSpec(f"AE_EQ(b2={beta2})", fn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weather", default="dry")
    ap.add_argument("--weights", type=float, nargs="*",
                    default=list(DEFAULT_WEIGHTS))
    ap.add_argument("--arms", nargs="*", default=["DDPG-B", "PANDA-fw"],
                    help="any of DDPG-B, PANDA-fw, PANDA-fcast, PANDA-cvar")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="weights")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    series = influent.canonical_scenario(args.weather)
    cfg = AerationEnvConfig(obs_mode="augmented", action_interval=20,
                            action_mode="absolute")
    base = dict(n_obs=4, gamma=0.99, ou_sigma=0.35, ou_sigma_final=0.05,
                warmup_steps=400, buffer_size=80000)

    rows: list[dict] = []
    for beta2 in args.weights:
        reward = make_reward(beta2)
        for arm in args.arms:
            for seed in range(args.seeds):
                fc = str(art / "forecaster.pt")
                if arm == "DDPG-B":
                    agent = DDPGAgent(DDPGConfig(seed=seed, **base))
                elif arm == "PANDA-fw":          # modifications 1 + 2
                    agent = PandaRLAgent(
                        PandaRLConfig(seed=seed, constraints=(), **base),
                        forecaster=fc)
                elif arm == "PANDA-fcast":       # modification 1 alone
                    agent = PandaRLAgent(
                        PandaRLConfig(seed=seed, constraints=(),
                                      n_quantiles=1, cvar_alpha=1.0, **base),
                        forecaster=fc)
                elif arm == "PANDA-cvar":        # modification 2 alone
                    agent = PandaRLAgent(
                        PandaRLConfig(seed=seed, constraints=(),
                                      use_forecast=False, **base),
                        forecaster=None)
                else:
                    raise ValueError(arm)

                print(f"\n=== {arm}  beta2={beta2}  seed {seed} ===",
                      flush=True)
                t0 = time.time()
                hist = train(agent, series, reward, cfg=cfg,
                             episodes=args.episodes, eval_every=5, seed=seed)
                final = [h for h in hist["history"] if "eval_AE" in h][-1]
                row = {"weather": args.weather, "arm": arm, "beta2": beta2,
                       "seed": seed, "episodes": args.episodes,
                       "wall_seconds": round(time.time() - t0, 1)}
                row.update({k[5:]: v for k, v in final.items()
                            if k.startswith("eval_")})
                rows.append(row)
                print(f"  AE={row['AE']:.1f} NH>4={100 * row['viol_NH']:.1f}% "
                      f"EQ={row['EQ']:.0f} DO={row['S_O5_mean']:.2f} "
                      f"[{row['wall_seconds']:.0f}s]", flush=True)
                (out / f"summary_{args.tag}.json").write_text(
                    json.dumps(rows, indent=2, default=float))

    print("\n| arm | beta2 | NH>4 | AE | mean S_O,5 |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["arm"], x["beta2"])):
        print(f"| {r['arm']} | {r['beta2']} | {100 * r['viol_NH']:.1f}% | "
              f"{r['AE']:.1f} | {r['S_O5_mean']:.2f} |")


if __name__ == "__main__":
    main()
