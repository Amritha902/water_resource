"""Aeration benchmark: PID / fuzzy / DDPG-A / DDPG-B / PANDA-RL.

Protocol of Du et al. (2023): learn on week 1 of each influent file,
evaluate with exploration off on week 2, report means over that week.

The two fixed-set-point comparators need no training.  The three learners are
run for several seeds because DDPG on this problem is seed-sensitive -- that
sensitivity is part of the result, so every seed is reported, not just the
best one.

Methods
-------
``PID``, ``Fuzzy``        hold S_O,5 at 2 mg/L (the paper's comparators)
``DDPG-A``                reward eq. (19), effluent quality only
``DDPG-B``                reward eq. (23), energy + ammonium
``PANDA-RL``              ours: forecast-conditioned, CVaR, Lagrangian limits
``PANDA-noForecast``      ablation: drop the forecast context
``PANDA-noCVaR``          ablation: optimise the mean return instead of the tail
``PANDA-noConstraint``    ablation: back to a fixed penalty weight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wwtp.bsm1 import influent
from wwtp.rl.baselines import FuzzyAeration, PIDAeration
from wwtp.rl.ddpg import DDPGAgent, DDPGConfig
from wwtp.rl.env import AerationEnvConfig
from wwtp.rl.panda_rl import DEFAULT_CONSTRAINTS, PandaRLAgent, PandaRLConfig
from wwtp.rl.rewards import REWARDS, RewardSpec
from wwtp.rl.train import evaluate, train

WEATHERS = ("dry", "rain", "storm")

#: the corrected environment: Markov observation, absolute action, 15-min
#: decisions.  See docs/04_second_paper.md Sec. 5 for why each is needed.
CORRECTED = dict(obs_mode="augmented", action_interval=20,
                 action_mode="absolute")
#: the environment exactly as the paper specifies it
AS_PUBLISHED = dict(obs_mode="paper", action_interval=1,
                    action_mode="incremental")

#: PANDA-RL optimises energy only; the discharge limits are constraints
ENERGY_ONLY = RewardSpec("energy", lambda info: -info["aeration_fraction"])

LEARNERS = ("DDPG-A", "DDPG-B", "PANDA-RL")
ABLATIONS = ("PANDA-noForecast", "PANDA-noCVaR", "PANDA-noConstraint")


def _base_cfg(seed: int, n_obs: int) -> dict:
    return dict(seed=seed, n_obs=n_obs, gamma=0.99, ou_sigma=0.35,
                ou_sigma_final=0.05, warmup_steps=500, buffer_size=80000)


def make(method: str, seed: int, artifacts: Path, published: bool = False):
    """Return ``(agent, reward, env_config, trainable)``."""
    env_kwargs = AS_PUBLISHED if published else CORRECTED
    cfg = AerationEnvConfig(**env_kwargs)
    n_obs = 4 if cfg.obs_mode == "augmented" else 3
    fc = str(artifacts / "forecaster.pt")

    if method == "PID":
        return PIDAeration(), REWARDS["AE_EQ"], cfg, False
    if method == "Fuzzy":
        return FuzzyAeration(), REWARDS["AE_EQ"], cfg, False
    if method == "DDPG-A":
        return DDPGAgent(DDPGConfig(**_base_cfg(seed, n_obs))), \
            REWARDS["EQ"], cfg, True
    if method == "DDPG-B":
        return DDPGAgent(DDPGConfig(**_base_cfg(seed, n_obs))), \
            REWARDS["AE_EQ"], cfg, True

    panda = dict(_base_cfg(seed, n_obs))
    forecaster: str | None = fc
    if method == "PANDA-RL":
        pass
    elif method == "PANDA-noForecast":
        forecaster = None
        panda["use_forecast"] = False
    elif method == "PANDA-noCVaR":
        panda["cvar_alpha"] = 1.0
    elif method == "PANDA-noConstraint":
        panda["constraints"] = ()
    else:
        raise ValueError(method)

    agent = PandaRLAgent(PandaRLConfig(**panda), forecaster=forecaster)
    reward = (REWARDS["AE_EQ"] if method == "PANDA-noConstraint"
              else ENERGY_ONLY)
    return agent, reward, cfg, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weathers", nargs="*", default=list(WEATHERS))
    ap.add_argument("--methods", nargs="*",
                    default=["PID", "Fuzzy", *LEARNERS])
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--published-env", action="store_true",
                    help="run the learners in the paper's exact environment")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="aeration")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    methods = list(args.methods) + (list(ABLATIONS) if args.ablations else [])
    series = {w: influent.canonical_scenario(w) for w in args.weathers}

    rows: list[dict] = []
    traces: dict[str, dict] = {}
    curves: dict[str, list] = {}

    jobs = []
    for weather in args.weathers:
        for method in methods:
            trainable = method not in ("PID", "Fuzzy")
            for seed in range(args.seeds if trainable else 1):
                jobs.append((weather, method, seed))

    for i, (weather, method, seed) in enumerate(jobs, 1):
        agent, reward, cfg, trainable = make(method, seed, art,
                                             published=args.published_env)
        tag = f"{weather}|{method}|s{seed}"
        print(f"\n[{i}/{len(jobs)}] {tag}", flush=True)
        if trainable:
            hist = train(agent, series[weather], reward, cfg=cfg,
                         episodes=args.episodes, eval_every=5, seed=seed)
            curves[tag] = hist["history"]
        # the fixed-set-point comparators run at the native 45 s period
        eval_cfg = cfg if trainable else AerationEnvConfig(action_interval=1)
        result = evaluate(agent, series[weather], reward, cfg=eval_cfg,
                          seed=seed)
        row = dict(result["summary"])
        row.update(weather=weather, method=method, seed=seed)
        if hasattr(agent, "lmbda") and len(getattr(agent, "lmbda", [])):
            for con, lam in zip(agent.constraints, agent.lmbda):
                row[f"lambda_{con.name}"] = float(lam)
        rows.append(row)
        if seed == 0:
            traces[tag] = result["trace"]
        print(f"    AE={row['AE']:.1f} EQ={row['EQ']:.0f} "
              f"DO={row['S_O5_mean']:.3f} NH={row['S_NH']:.2f} "
              f"TN={row['N_tot']:.2f} violNH={row['viol_NH']:.3f} "
              f"violTN={row['viol_TN']:.3f}", flush=True)

        (out / f"summary_{args.tag}.json").write_text(
            json.dumps(rows, indent=2, default=float))

    (out / f"traces_{args.tag}.json").write_text(json.dumps(traces, default=float))
    (out / f"curves_{args.tag}.json").write_text(json.dumps(curves, default=float))

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out / f"summary_{args.tag}.csv", index=False)
        cols = ["AE", "EQ", "S_O5_mean", "S_NH", "N_tot", "viol_NH",
                "viol_TN", "OCI"]
        agg = df.groupby(["weather", "method"])[cols].mean()
        # aeration energy saved against the PID comparator, per weather
        for weather in df["weather"].unique():
            base = df[(df.weather == weather) & (df.method == "PID")]["AE"]
            if len(base):
                mask = agg.index.get_level_values(0) == weather
                agg.loc[mask, "AE_saving_%"] = (
                    100.0 * (float(base.iloc[0]) - agg.loc[mask, "AE"])
                    / float(base.iloc[0]))
        agg.to_csv(out / f"aggregate_{args.tag}.csv")
        print("\n" + agg.round(3).to_string())
    except ImportError:
        pass


if __name__ == "__main__":
    main()
