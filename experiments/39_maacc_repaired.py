"""Train paper 1's MAACC with repaired training (src/wwtp/sacil/maacc_repaired.py).

Same data and budget as SAC-IL (experiments/28_sac_il_paper1.py): dry weather
days 0-7, the expert's run first put in the replay buffer, then 5 episodes of
7 days, the policy kept being the best on the training week (same score as
SAC-IL); the starting point (the prior alone) is a candidate too.  Two set-ups:

  --mode paper   as paper 1: both loops, on the paper's PID prior
  --mode stack   in our controller: oxygen loop only, on PID x3 + daily ILC
                 (the safety gate is added at test time, as for SAC-IL)

    python experiments/39_maacc_repaired.py --mode paper --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.sacil.ilc import DailyILC  # noqa: E402
from wwtp.sacil.maacc_repaired import RepairedMAACC  # noqa: E402
from wwtp.sacil.sacil import Paper1Env, evaluate  # noqa: E402

T = json.loads((ROOT / "results" / "paper1_tuning.json").read_text())
START_STEPS = 1920          # one day of exploration before the first update
GAIN = float(T["gain"])


def make_env(mode, weather, warm, **kw):
    st, u0 = warm
    if mode == "paper":
        return Paper1Env(weather, st, u0, weights=Paper1Env.W_BALANCED, **kw)
    return Paper1Env(weather, st, u0, weights=Paper1Env.W_BALANCED, base_gain=(GAIN, GAIN),
                     act_mask=(1.0, 0.0), ilc=DailyILC(**T["ilc"]), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("paper", "stack"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--train-days", type=float, default=7.0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    warm = pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())
    tag = f"maacc_rep_{a.mode}"
    E = lambda w, **kw: make_env(a.mode, w, warm, **kw)
    agent = RepairedMAACC(Paper1Env.obs_dim, active=(True, a.mode == "paper"), seed=a.seed)
    train_env = E("dry", days=a.train_days, seed=a.seed)
    t0 = time.time()

    # the expert's run (the prior alone: action 0) goes into the replay buffer; as in
    # TD3, learning starts only after a first day of exploring around the prior (an
    # earlier version also learned offline from the expert run alone, where the action
    # never varies: the critic then extrapolates and the actor drifted before acting)
    o, done = train_env.reset(), False
    while not done:
        o2, r, done, info = train_env.step(np.zeros(2))
        agent.buf.add(o, np.zeros(2), r, o2, float(info["fail"]))
        o = o2
    pol = lambda o, env: agent.act(o)
    cost = lambda ev: ev["IAE_SO5"] / 0.004 + ev["IAE_SNO2"] / 0.002 + (100 if ev["fail"] else 0)
    ev = evaluate(E("dry", days=a.train_days), pol)
    log = [dict(episode=0, **{k: v for k, v in ev.items() if k != "_trace"})]
    best = (cost(ev), copy.deepcopy(agent.state_dict()), 0)
    print(f"[{tag} {a.seed}] start IAE O {ev['IAE_SO5']:.5f} N {ev['IAE_SNO2']:.5f} ({time.time() - t0:.0f}s)", flush=True)

    for ep in range(1, a.episodes + 1):
        o, done, ret = train_env.reset(), False, 0.0
        while not done:
            act = agent.act(o, deterministic=False)
            o2, r, done, info = train_env.step(act)
            agent.buf.add(o, act, r, o2, float(info["fail"]))
            if ep > 1 or train_env.k >= START_STEPS:
                agent.update()
            o, ret = o2, ret + r
        ev = evaluate(E("dry", days=a.train_days), pol)
        if cost(ev) < best[0]:
            best = (cost(ev), copy.deepcopy(agent.state_dict()), ep)
        log.append(dict(episode=ep, train_return=ret, **{k: v for k, v in ev.items() if k != "_trace"}))
        print(f"[{tag} {a.seed}] ep {ep}: return {ret:.0f}  eval IAE O {ev['IAE_SO5']:.5f} "
              f"N {ev['IAE_SNO2']:.5f} fail {ev['fail']} ({time.time() - t0:.0f}s)", flush=True)

    agent.load_state_dict(best[1])
    print(f"[{tag} {a.seed}] kept policy of episode {best[2]}", flush=True)
    torch.save(best[1], ROOT / "artifacts" / f"{tag}_actor_seed{a.seed}.pt")
    (ROOT / "results" / f"curves_{tag}_seed{a.seed}.json").write_text(
        json.dumps(dict(kept_episode=best[2], log=log), indent=1))


if __name__ == "__main__":
    main()
