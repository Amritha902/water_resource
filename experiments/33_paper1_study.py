"""Paper 1, new controller, stage C: test every controller on the same data.

Settings come from stage A (experiments/32_paper1_tuning.py, dry week 1 only)
and the SAC-IL policies from experiments/28_sac_il_paper1.py (trained on dry
week 1 only).  Every controller starts from the same warm-up state and runs
the full 14 days of the official dry, rain and storm files, and dry weather
with paper 1's actuator noise.  Scores are kept for the whole run (the
paper's protocol) and for week 2 alone (days 7-14: never used for tuning or
training; it holds the rain and storm events).

Controllers (the ablation):
  PID                 paper 1's PID prior
  MAACC               paper 1's controller, exactly as printed
  PID+ILC             paper's PID + our daily ILC
  MAACC+ILC           paper's MAACC + our daily ILC (a fair add-on to the paper's own method)
  PIDx3               retuned prior
  PIDx3+ILC           retuned prior + daily ILC
  PIDx3+SAC-IL        retuned prior + SAC-IL correction (no ILC), with the safety gate
  Ours                retuned prior + daily ILC + SAC-IL correction, with the safety gate
  Ours (no gate)      the same without clipping and gate

    python experiments/33_paper1_study.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.maacc.exact import ExactMAACC  # noqa: E402
from wwtp.sacil.ilc import DailyILC  # noqa: E402
from wwtp.sacil.sacil import RES_SCALE, Actor, Paper1Env, evaluate, expert_increment  # noqa: E402

T = json.loads((ROOT / "results" / "paper1_tuning.json").read_text())
ILC, GAIN = T["ilc"], float(T["gain"])
MARGIN = 0.10
NOISE_RUNS, NOISE_RUNS_SAC = 10, 5
SAC_SEEDS = {"sacil_v2": (0, 1, 2, 3, 4), "sacil_noilc": (0, 1, 2)}


def warm():
    return pickle.loads((ROOT / "demo" / "cache" / "warm_paper1.pkl").read_bytes())


def make_env(weather, ctrl, seed=0, noise=False, noise_seed=0, days=None):
    st, u0 = warm()
    use_ilc = "ILC" in ctrl or ctrl.startswith("Ours")
    ilc = DailyILC(**ILC) if use_ilc else None
    kw = dict(days=days, noise=noise, seed=noise_seed, ilc=ilc)
    if ctrl.startswith("MAACC"):
        return Paper1Env(weather, st, u0, base_law=lambda: ExactMAACC(seed=seed, critic="paper", u0=u0.copy()),
                         act_mask=(0.0, 0.0), **kw)
    if "SAC" in ctrl or ctrl.startswith("Ours"):
        return Paper1Env(weather, st, u0, base_gain=(1.0, GAIN), act_mask=(1.0, 0.0),
                         weights=Paper1Env.W_BALANCED, **kw)
    g = GAIN if "x3" in ctrl else 1.0
    return Paper1Env(weather, st, u0, base_gain=(g, g), act_mask=(0.0, 0.0), **kw)


_RANGE = {}


def training_range(ctrl):
    """Input range of the SAC-IL training data: the demonstrator on dry days 0-7."""
    if ctrl not in _RANGE:
        e = make_env("dry", ctrl, days=7.0)
        o, done, obs = e.reset(), False, []
        while not done:
            obs.append(o)
            o, _, done, _ = e.step(expert_increment(e.e, e.e1, e.e2, GAIN - e.base_gain) * e.mask / RES_SCALE)
        obs = np.array(obs)
        _RANGE[ctrl] = (obs.min(0), obs.max(0))
    return _RANGE[ctrl]


def sac_policy(tag, seed, ctrl, gated):
    actor = Actor(Paper1Env.obs_dim, Paper1Env.act_dim)
    actor.load_state_dict(torch.load(ROOT / "artifacts" / f"{tag}_actor_seed{seed}.pt"))
    lo, hi = training_range(ctrl)
    width = np.maximum(hi - lo, 1e-6)
    stats = {"gated": 0, "n": 0}

    def act(o, e):
        stats["n"] += 1
        if gated:
            ood = np.any((o < lo - MARGIN * width) | (o > hi + MARGIN * width))
            need = abs(expert_increment(e.e, e.e1, e.e2, GAIN - e.base_gain)[0]) > RES_SCALE[0]
            if ood or need:
                stats["gated"] += 1
                e.fallback_gain = GAIN
                return np.zeros(2)
            o = np.clip(o, lo, hi)
        with torch.no_grad():
            mu, _ = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
        return torch.tanh(mu)[0].numpy()
    return act, stats


def run(job):
    ctrl, weather, seed, noise, noise_seed, keep_trace = job
    env = make_env(weather, ctrl, seed=seed, noise=noise, noise_seed=noise_seed)
    stats = {"gated": 0, "n": 1}
    if "SAC" in ctrl or ctrl.startswith("Ours"):
        tag = "sacil_noilc" if ctrl == "PIDx3+SAC-IL" else "sacil_v2"
        pol, stats = sac_policy(tag, seed, ctrl, gated=ctrl != "Ours (no gate)")
    else:
        pol = lambda o, e: np.zeros(2)
    t0 = time.perf_counter()
    out = evaluate(env, pol)
    out["wall_s"] = time.perf_counter() - t0
    tr = out.pop("_trace")
    return dict(controller=ctrl, weather=weather, noise=noise, seed=seed, noise_seed=noise_seed,
                gated_share=stats["gated"] / max(stats["n"], 1), _trace=tr if keep_trace else None, **out)


def jobs():
    J = []
    for c in ("PID", "PID+ILC", "PIDx3", "PIDx3+ILC"):
        J += [(c, w, 0, False, 0, True) for w in ("dry", "rain", "storm")]
        J += [(c, "dry", 0, True, n, n == 0) for n in range(NOISE_RUNS)]
    for c in ("MAACC", "MAACC+ILC"):
        J += [(c, w, s, False, 0, s == 0) for w in ("dry", "rain", "storm") for s in (0, 1, 2)]
        J += [(c, "dry", n, True, n, n == 0) for n in range(NOISE_RUNS)]
    for c, tag in (("PIDx3+SAC-IL", "sacil_noilc"), ("Ours", "sacil_v2"), ("Ours (no gate)", "sacil_v2")):
        for s in SAC_SEEDS[tag]:
            J += [(c, w, s, False, 0, s == 0) for w in ("dry", "rain", "storm")]
            if c != "Ours (no gate)":
                J += [(c, "dry", s, True, n, s == 0 and n == 0) for n in range(NOISE_RUNS_SAC)]
    return J


def compute_costs():
    """Computing time per control period of each building block (this machine, one core)."""
    st, u0 = warm()
    r = np.array([2.0, 1.0])
    s = np.array([2.01, 0.99])
    out = {}
    m = ExactMAACC(seed=0, critic="paper", u0=u0.copy())
    t0 = time.perf_counter()
    for _ in range(2000):
        m(s, r)
    out["MAACC_ms"] = 1000 * (time.perf_counter() - t0) / 2000
    e = make_env("dry", "PID", days=0.01)
    e.reset()
    t0 = time.perf_counter()
    for _ in range(20000):
        expert_increment(e.e, e.e1, e.e2, GAIN)
    out["PID_ms"] = 1000 * (time.perf_counter() - t0) / 20000
    ilc = DailyILC(**ILC)
    t0 = time.perf_counter()
    for k in range(1920):
        ilc.record(k, np.zeros(2), u0)
    out["ILC_ms_per_period_incl_daily_update"] = 1000 * (time.perf_counter() - t0) / 1920
    actor = Actor(Paper1Env.obs_dim, Paper1Env.act_dim)
    x = torch.zeros(1, Paper1Env.obs_dim)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(5000):
            actor(x)
    out["SAC_actor_ms"] = 1000 * (time.perf_counter() - t0) / 5000
    out["SAC_actor_parameters"] = int(sum(p.numel() for p in actor.parameters()))
    return out


def main():
    J = jobs()
    print(f"{len(J)} runs", flush=True)
    rows, traces = [], {}
    with mp.get_context("spawn").Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(run, J), 1):
            tr = r.pop("_trace")
            if tr is not None:
                traces[f"{r['controller']}|{r['weather']}|{'noise' if r['noise'] else 'clean'}"] = tr
            rows.append(r)
            print(f"[{i}/{len(J)}] {r['controller']:15s} {r['weather']:5s} noise={r['noise']!s:5s} s{r['seed']} "
                  f"O {r['IAE_SO5']:.5f} (test {r['IAE_SO5_test']:.5f})  N {r['IAE_SNO2']:.5f}  "
                  f"gate {100 * r['gated_share']:.0f}%", flush=True)
    out = {"tuning": {"gain": GAIN, "ilc": ILC}, "compute": compute_costs(), "runs": rows}
    (ROOT / "results" / "summary_paper1_study.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results" / "traces_paper1_study.json").write_text(json.dumps(traces))
    print("compute:", out["compute"])


if __name__ == "__main__":
    main()
