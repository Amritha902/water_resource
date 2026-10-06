"""Training and evaluation loops for the aeration agents.

Protocol of Du et al. (2023), Section 4.1: the agent learns on the first
week of an influent file and is evaluated, with exploration off, on the
second week.  Reported figures are means over the evaluation week.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import metrics
from ..bsm1 import asm1
from ..bsm1.influent import InfluentSeries
from .env import AerationEnv, AerationEnvConfig, NH_LIMIT, TN_LIMIT
from .rewards import RewardSpec

SEC_PER_DAY = 86400.0


def evaluate(agent, series: InfluentSeries, reward: RewardSpec,
             cfg: AerationEnvConfig | None = None, start_day: float = 7.0,
             days: float = 7.0, seed: int = 0,
             record_stride: int = 20) -> dict:
    """Run a policy with exploration off and summarise the week."""
    env = AerationEnv(series, cfg, start_day=start_day, days=days, seed=seed)
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    rows: list[dict] = []
    eff: list[np.ndarray] = []
    q_eff: list[float] = []
    rewards: list[float] = []
    trace: dict[str, list] = {k: [] for k in
                              ("t", "S_O5", "KLa5", "S_NH", "N_tot", "q_in")}
    while True:
        action = agent.act(obs, info=None, explore=False)
        obs, info, done = env.step(action)
        rewards.append(reward(info))
        rows.append(info)
        eff.extend(info["eff_samples"])
        q_eff.extend(info["q_eff_samples"])
        if (info["step"] - 1) % record_stride == 0:
            trace["t"].append(info["t_days"])
            for key in ("S_O5", "KLa5", "S_NH", "N_tot", "q_in"):
                trace[key].append(info[key])
        if done:
            break

    eff_arr = np.array(eff)
    dt_days = env.cfg.dt_seconds / SEC_PER_DAY
    out = {
        "AE": float(np.mean([r["AE"] for r in rows])),
        "PE": float(np.mean([r["PE"] for r in rows])),
        "EQ": metrics.effluent_quality_index(eff_arr, np.array(q_eff), dt_days),
        "S_NH": float(np.mean([r["S_NH"] for r in rows])),
        "N_tot": float(np.mean([r["N_tot"] for r in rows])),
        "COD": float(np.mean([r["COD"] for r in rows])),
        "BOD5": float(np.mean([r["BOD5"] for r in rows])),
        "TSS": float(np.mean([r["TSS"] for r in rows])),
        "S_O5_mean": float(np.mean([r["S_O5"] for r in rows])),
        "S_O5_std": float(np.std([r["S_O5"] for r in rows])),
        "KLa5_mean": float(np.mean([r["KLa5"] for r in rows])),
        "viol_NH": float(np.mean(eff_arr[:, asm1.S_NH] > NH_LIMIT)),
        "viol_TN": float(np.mean(asm1.total_nitrogen(eff_arr) > TN_LIMIT)),
        "NH_p95": float(np.percentile(eff_arr[:, asm1.S_NH], 95)),
        "reward": float(np.mean(rewards)),
    }
    out["OCI"] = out["AE"] + out["PE"]
    return {"summary": out, "trace": trace}


def train(agent, series: InfluentSeries, reward: RewardSpec,
          cfg: AerationEnvConfig | None = None, episodes: int = 6,
          train_start_day: float = 0.0, train_days: float = 7.0,
          eval_every: int = 1, seed: int = 0, verbose: bool = True,
          sigma_schedule: bool = True) -> dict:
    """Online learning on the training week, periodic held-out evaluation."""
    history: list[dict] = []
    c = agent.cfg
    for ep in range(episodes):
        env = AerationEnv(series, cfg, start_day=train_start_day,
                          days=train_days, seed=seed + ep)
        obs = env.reset()
        agent.reset()
        frac = ep / max(episodes - 1, 1)
        sigma = (c.ou_sigma + frac * (c.ou_sigma_final - c.ou_sigma)
                 if sigma_schedule else c.ou_sigma)
        ep_reward, n_decisions = 0.0, 0
        while True:
            action = agent.act(obs, info=None, explore=True, sigma=sigma)
            nxt, info, done = env.step(action)
            r = reward(info)
            agent.observe(obs, action, r, nxt, done)
            agent.update()
            obs = nxt
            ep_reward += r
            n_decisions += 1
            if done:
                break
        record = {"episode": ep, "sigma": sigma, "decisions": n_decisions,
                  "train_reward": ep_reward / max(n_decisions, 1)}
        if (ep + 1) % eval_every == 0 or ep == episodes - 1:
            ev = evaluate(agent, series, reward, cfg, seed=seed)["summary"]
            record.update({f"eval_{k}": v for k, v in ev.items()})
        history.append(record)
        if verbose:
            msg = (f"  ep {ep:2d} sigma={sigma:.3f} "
                   f"train_r={record['train_reward']:+.4f}")
            if "eval_AE" in record:
                msg += (f" | eval AE={record['eval_AE']:.1f} "
                        f"EQ={record['eval_EQ']:.0f} "
                        f"DO={record['eval_S_O5_mean']:.3f} "
                        f"NH={record['eval_S_NH']:.2f} "
                        f"TN={record['eval_N_tot']:.2f}")
            print(msg, flush=True)
    return {"history": history}


def save_history(path: str | Path, blob: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(blob, indent=2, default=float))
