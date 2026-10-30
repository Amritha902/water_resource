"""SAC-IL outside its training range: input clipping and a safety gate.

The SAC-IL policy is trained on dry weather.  In rain and storm some of its
inputs (influent flow, recycle setting) leave the range it has seen, and the
network extrapolates.  Two safeguards, applied to the trained policies
without retraining:

* clip   each input is clipped to the range seen in training;
* gate   the learned correction is switched off for a period, and the
         demonstrator (PID x3 + ILC, unbounded) acts alone, when any input
         leaves the training range by more than ``MARGIN`` of its width, or
         when the correction the demonstrator would need is larger than the
         learner's bound (this happens under actuator noise).

    python experiments/30_sacil_guard.py
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.sacil.ilc import DailyILC  # noqa: E402
from wwtp.sacil.sacil import RES_SCALE, Actor, Paper1Env, evaluate, expert_action, expert_increment  # noqa: E402

spec = importlib.util.spec_from_file_location("e28", ROOT / "experiments" / "28_sac_il_paper1.py")
E28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E28)

MARGIN = 0.10
LOOPS = dict(base_gain=(1.0, 3.0), act_mask=(1.0, 0.0))


def env(w, **kw):
    st, u0 = pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())
    return Paper1Env(w, st, u0, ilc=DailyILC(**E28.ILC), weights=Paper1Env.W_BALANCED, **LOOPS, **kw)


def training_range():
    """Input range of the training data (the demonstrator on dry days 0-7)."""
    e, obs = env("dry", days=7.0), []
    o, done = e.reset(), False
    while not done:
        obs.append(o)
        o, _, done, _ = e.step(expert_action(e, 3.0))
    obs = np.array(obs)
    return obs.min(0), obs.max(0)


def policy(seed, mode, lo, hi):
    actor = Actor(Paper1Env.obs_dim, Paper1Env.act_dim)
    actor.load_state_dict(torch.load(ROOT / "artifacts" / f"sacil_actor_seed{seed}.pt"))
    width = np.maximum(hi - lo, 1e-6)
    stats = {"gated": 0, "n": 0}

    def act(o, e):
        stats["n"] += 1
        if mode in ("gate", "clip+gate"):
            ood = np.any((o < lo - MARGIN * width) | (o > hi + MARGIN * width))
            need = abs(expert_increment(e.e, e.e1, e.e2, 3.0 - e.base_gain)[0]) > RES_SCALE[0]
            if ood or need:
                stats["gated"] += 1
                e.fallback_gain = 3.0
                return np.zeros(2)
        x = np.clip(o, lo, hi) if mode in ("clip", "clip+gate") else o
        with torch.no_grad():
            mu, _ = actor(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))
        return torch.tanh(mu)[0].numpy()
    return act, stats


def run(job):
    weather, seed, mode, noise_seed = job
    lo, hi = RANGE
    pol, stats = policy(seed, mode, lo, hi)
    noise = noise_seed is not None
    out = evaluate(env(weather, noise=noise, seed=noise_seed or 0), pol)
    out.pop("_trace")
    return dict(weather=weather, noise=noise, noise_seed=noise_seed, seed=seed, mode=mode,
                gated_share=stats["gated"] / max(stats["n"], 1), **out)


RANGE = training_range()


def main():
    modes = ("raw", "clip", "gate", "clip+gate")
    jobs = [(w, s, m, None) for w in ("dry", "rain", "storm") for s in (0, 1, 2) for m in modes]
    jobs += [("dry", s, m, n) for s in (0, 1, 2) for m in modes for n in range(5)]
    rows = []
    with mp.get_context("spawn").Pool(4) as pool:          # torch does not survive a fork here
        for r in pool.imap_unordered(run, jobs):
            rows.append(r)
            print(f"{r['weather']:5s} noise={r['noise']!s:5s} seed {r['seed']} {r['mode']:9s}: "
                  f"IAE O {r['IAE_SO5']:.5f} N {r['IAE_SNO2']:.5f} gated {100 * r['gated_share']:.1f}%",
                  flush=True)
    (ROOT / "results" / "summary_sacil_guard.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
