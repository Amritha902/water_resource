"""Paper 1 task with SAC + imitation learning (SAC-IL) instead of MAACC.

Protocol as in experiments/20_exact_paper1.py: official BSM1, 14-day dry
warm-up under the paper's PID, then 14 days of evaluation at 45 s.

Training uses dry weather only (days 0-7 of the dry file).  The expert is the
paper's own PID.  Rain, storm and the actuator-noise case are not seen in
training.  The policy kept is the one with the lowest training-weather cost.

    python experiments/28_sac_il_paper1.py --seed 0
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
from wwtp.sacil.sacil import Paper1Env, SACIL, SACILConfig, evaluate, expert_action, run_expert  # noqa: E402

#: daily ILC (beta, lead in periods, smoothing half-width) and PID gain, chosen
#: on dry weather, week 1 only, by experiments/32_paper1_tuning.py
_TUNING = ROOT / "results" / "paper1_tuning.json"
_T = json.loads(_TUNING.read_text()) if _TUNING.exists() else {}
ILC = _T.get("ilc", dict(beta=[0.5, 0.0], lead=[4, 0], smooth=15))
GAIN = float(_T.get("gain", 3.0))


def warm():
    f = ROOT / "demo" / "cache" / "warm_paper1.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    import importlib.util
    spec = importlib.util.spec_from_file_location("e20", ROOT / "experiments" / "20_exact_paper1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.warm_state()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--train-days", type=float, default=7.0)
    ap.add_argument("--noise-runs", type=int, default=20)
    ap.add_argument("--tag", default="sacil")
    ap.add_argument("--demo-gain", type=float, default=GAIN, help="gain of the demonstrator PID (1 = paper's PID)")
    ap.add_argument("--ilc", action="store_true", help="daily iterative learning control in the loop")
    ap.add_argument("--balanced", action="store_true", help="weight both loops by the retuned prior's error")
    ap.add_argument("--oxygen-only", action="store_true",
                    help="the learner corrects KLa,5 only; the nitrate loop keeps the demonstrator PID")
    ap.add_argument("--tuned", action="store_true",
                    help="on the PID tuned term by term (experiments/38) and its ILC: the learner corrects "
                         "KLa,5 only, the demonstrator is that PID + ILC itself")
    a = ap.parse_args()
    torch.set_num_threads(1)
    st, u0 = warm()
    ilc = DailyILC(**ILC) if a.ilc else None
    wts = Paper1Env.W_BALANCED if a.balanced else None
    loops = dict(base_gain=(1.0, a.demo_gain), act_mask=(1.0, 0.0)) if a.oxygen_only else {}
    if a.tuned:
        tp = json.loads((ROOT / "results" / "paper1_tuned_pid.json").read_text())
        ilc = DailyILC(**tp["ilc"]) if a.ilc else None
        loops = dict(terms=tp["terms"], act_mask=(1.0, 0.0))
        a.demo_gain = 1.0              # the demonstrator is the base itself: action 0
    E = lambda w, **kw: Paper1Env(w, st, u0, ilc=ilc, weights=wts, **loops, **kw)
    agent = SACIL(Paper1Env.obs_dim, Paper1Env.act_dim, SACILConfig(), seed=a.seed)
    train_env = E("dry", days=a.train_days, seed=a.seed)

    # 1. imitation: demonstrations of the paper's PID, then behaviour cloning
    t0 = time.time()
    demo_cost = run_expert(train_env, agent, gain=a.demo_gain)
    bc_loss = agent.pretrain_bc(4000)
    agent.pretrain_critic(3000)
    pol = lambda o, env: agent.act(o)
    bc = evaluate(E("dry", days=a.train_days), pol)
    print(f"[seed {a.seed}] expert cost {demo_cost:.0f}; BC loss {bc_loss:.2e}; "
          f"BC policy IAE O {bc['IAE_SO5']:.5f} N {bc['IAE_SNO2']:.5f} ({time.time() - t0:.0f}s)", flush=True)
    log = [dict(episode=0, **{k: v for k, v in bc.items() if k != "_trace"})]
    # policy selection on the training weather: both loops scaled by the retuned prior's error
    cost = lambda ev: ev["IAE_SO5"] / 0.004 + ev["IAE_SNO2"] / 0.002 + (100 if ev["fail"] else 0)
    best = (cost(bc), copy.deepcopy(agent.actor.state_dict()), 0)

    # 2. soft actor-critic on dry weather
    for ep in range(1, a.episodes + 1):
        o, done, ret = train_env.reset(), False, 0.0
        while not done:
            act = agent.act(o, deterministic=False)
            o2, r, done, info = train_env.step(act)
            agent.buf.add(o, act, r, o2, float(info["fail"]))
            agent.update()
            o, ret = o2, ret + r
        ev = evaluate(E("dry", days=a.train_days), pol)
        score = cost(ev)
        if score < best[0]:
            best = (score, copy.deepcopy(agent.actor.state_dict()), ep)
        log.append(dict(episode=ep, train_return=ret, alpha=float(agent.log_alpha.exp().detach()),
                        **{k: v for k, v in ev.items() if k != "_trace"}))
        print(f"[seed {a.seed}] ep {ep}: return {ret:.0f}  eval IAE O {ev['IAE_SO5']:.5f} "
              f"N {ev['IAE_SNO2']:.5f} fail {ev['fail']}  alpha {float(agent.log_alpha.exp()):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    agent.actor.load_state_dict(best[1])
    print(f"[seed {a.seed}] kept policy of episode {best[2]}", flush=True)
    out_dir = ROOT / "results"
    torch.save(agent.actor.state_dict(), ROOT / "artifacts" / f"{a.tag}_actor_seed{a.seed}.pt")

    # 3. evaluation: 14 days, every weather, plus actuator noise (dry)
    rows, traces = [], {}
    for w in ("dry", "rain", "storm"):
        ev = evaluate(E(w), pol)
        traces[f"{w}|clean"] = ev.pop("_trace")
        rows.append(dict(weather=w, noise=False, controller="SAC-IL", seed=a.seed, kept_episode=best[2], **ev))
        print(f"[seed {a.seed}] {w}: IAE O {ev['IAE_SO5']:.5f} N {ev['IAE_SNO2']:.5f}", flush=True)
    for ns in range(a.noise_runs):
        ev = evaluate(E("dry", noise=True, seed=ns), pol)
        tr = ev.pop("_trace")
        if ns == 0:
            traces["dry|noise"] = tr
        rows.append(dict(weather="dry", noise=True, controller="SAC-IL", seed=a.seed, noise_seed=ns,
                         kept_episode=best[2], **ev))
    nz = [r for r in rows if r["noise"]]
    print(f"[seed {a.seed}] dry+noise: IAE O {np.mean([r['IAE_SO5'] for r in nz]):.5f} "
          f"N {np.mean([r['IAE_SNO2'] for r in nz]):.5f}", flush=True)
    (out_dir / f"summary_{a.tag}_seed{a.seed}.json").write_text(json.dumps(rows, indent=1))
    (out_dir / f"curves_{a.tag}_seed{a.seed}.json").write_text(json.dumps(log, indent=1))
    (out_dir / f"traces_{a.tag}_seed{a.seed}.json").write_text(json.dumps(traces))
    print(f"[seed {a.seed}] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
