"""Learned one-step model of the aeration MDP, for Dyna-style updates.

Reproducing Du et al. showed the agent needs roughly fifteen simulated weeks
of interaction to converge, against the seven days of online learning the
paper describes (``docs/04_second_paper.md`` Sec. 5.3).  On a real plant those
extra weeks are spent exploring at the actuator rails, discharging illegal
effluent the whole time.  A learned model is the only way to buy that
experience without running it.

The model is deliberately **feed-forward**, not recurrent, unlike
``wwtp.twin.DigitalTwin``: Dyna needs to predict a successor for an arbitrary
state drawn from the replay buffer, and a recurrent model has no hidden state
to attach to such a state.  One decision interval (15 min) is long enough
that a residual MLP on

    [ s_t (3 tank-5 concentrations), KLa_5 / 240, d_t (3 inlet channels) ]

predicts the next observation well; the effluent heads give the ammonium and
total nitrogen a synthetic transition needs for its reward and its constraint
costs.

As with the tracking twin, the training data must contain **open-loop**
excitation.  A model fitted on data from a closed-loop aeration policy learns
a near-zero aeration gain, because the controller raises ``KLa_5`` exactly
when the oxygen demand rises.  ``collect_corpus`` therefore drives ``KLa_5``
with a piecewise-constant pseudo-random sequence over its whole range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bsm1.influent import random_scenario
from .env import AerationEnv, AerationEnvConfig

#: physical scales; the observation is already divided by ``obs_scale``
D_SCALE = np.array([20000.0, 40.0, 90.0])      # Q_in, S_NH,in, S_S,in
EFF_SCALE = np.array([5.0, 20.0])              # effluent S_NH, N_tot


class AerationTwin(nn.Module):
    """``s_{t+1} = s_t + f(s_t, u_t, d_t)`` with effluent heads."""

    def __init__(self, n_obs: int = 3, hidden: int = 128):
        super().__init__()
        self.n_obs = n_obs
        n_in = n_obs + 1 + 3
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.delta_head = nn.Linear(hidden, n_obs)
        self.eff_head = nn.Linear(hidden, 2)

    def forward(self, obs: torch.Tensor, u: torch.Tensor,
                d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``obs`` scaled as the env emits it, ``u`` = KLa_5/240, ``d`` scaled."""
        h = self.trunk(torch.cat([obs, u, d], dim=-1))
        return obs + self.delta_head(h), self.eff_head(h)


def collect_corpus(n_rollouts: int = 12, seed: int = 0, days: float = 14.0,
                   cfg: AerationEnvConfig | None = None) -> dict:
    """Open-loop PRBS rollouts in the aeration environment."""
    cfg = cfg or AerationEnvConfig(obs_mode="paper", action_interval=20,
                                   action_mode="absolute")
    rng = np.random.default_rng(seed)
    O, U, D, O2, E = [], [], [], [], []
    for i in range(n_rollouts):
        series = random_scenario(rng, days=days)
        env = AerationEnv(series, cfg, seed=seed + i)
        obs = env.reset()
        hold, level = 0, 0.0
        while True:
            if hold <= 0:                      # dwell 1-8 decisions (15-120 min)
                hold = int(rng.integers(1, 9))
                level = float(rng.uniform(-1.0, 1.0))
            hold -= 1
            d_prev = np.array([series.flow[min(env.k0 + env.k,
                                               series.n_steps - 1)],
                               series.composition[min(env.k0 + env.k,
                                                      series.n_steps - 1), 9],
                               series.composition[min(env.k0 + env.k,
                                                      series.n_steps - 1), 1]])
            O.append(obs.copy())
            U.append([0.5 * (level + 1.0)])     # the absolute KLa_5/240 applied
            D.append(d_prev / D_SCALE)
            obs, info, done = env.step(level)
            O2.append(obs.copy())
            E.append([info["S_NH"] / EFF_SCALE[0], info["N_tot"] / EFF_SCALE[1]])
            if done:
                break
        print(f"  rollout {i + 1}/{n_rollouts} ({series.weather})", flush=True)
    return {"obs": np.asarray(O, np.float32), "u": np.asarray(U, np.float32),
            "d": np.asarray(D, np.float32), "obs2": np.asarray(O2, np.float32),
            "eff": np.asarray(E, np.float32)}


def train_aeration_twin(out_dir: str | Path, n_rollouts: int = 12,
                        epochs: int = 60, batch_size: int = 512,
                        lr: float = 2e-3, seed: int = 0, val_frac: float = 0.15,
                        eff_weight: float = 1.0, device: str = "cpu",
                        corpus: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    corpus = corpus or collect_corpus(n_rollouts=n_rollouts, seed=seed)
    n = len(corpus["obs"])
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_frac * n))
    val, train = idx[:n_val], idx[n_val:]
    tensors = {k: torch.from_numpy(v) for k, v in corpus.items()}

    model = AerationTwin(n_obs=corpus["obs"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def batches(sel, shuffle):
        order = np.random.permutation(sel) if shuffle else sel
        for i in range(0, len(order), batch_size):
            j = order[i:i + batch_size]
            yield (tensors["obs"][j], tensors["u"][j], tensors["d"][j],
                   tensors["obs2"][j], tensors["eff"][j])

    history = []
    for epoch in range(epochs):
        model.train()
        tr, seen = 0.0, 0
        for o, u, d, o2, e in batches(train, True):
            po2, pe = model(o, u, d)
            loss = F.smooth_l1_loss(po2, o2) + eff_weight * F.smooth_l1_loss(pe, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss) * len(o)
            seen += len(o)
        sched.step()

        model.eval()
        va, do_err, nh_err, vseen = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for o, u, d, o2, e in batches(val, False):
                po2, pe = model(o, u, d)
                va += float(F.smooth_l1_loss(po2, o2)
                            + eff_weight * F.smooth_l1_loss(pe, e)) * len(o)
                # physical errors: DO is channel 1 (scale 4), effluent NH scale 5
                do_err += float(((po2[:, 1] - o2[:, 1]) * 4.0).abs().mean()) * len(o)
                nh_err += float(((pe[:, 0] - e[:, 0])
                                 * EFF_SCALE[0]).abs().mean()) * len(o)
                vseen += len(o)
        history.append({"epoch": epoch, "train": tr / max(seen, 1),
                        "val": va / max(vseen, 1),
                        "val_DO_mae": do_err / max(vseen, 1),
                        "val_NH_mae": nh_err / max(vseen, 1)})
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train {history[-1]['train']:.5f}  "
                  f"val {history[-1]['val']:.5f}  DO MAE "
                  f"{history[-1]['val_DO_mae']:.4f} mg/L  NH MAE "
                  f"{history[-1]['val_NH_mae']:.4f} mg/L", flush=True)

    torch.save({"state_dict": model.state_dict(), "n_obs": model.n_obs,
                "d_scale": D_SCALE, "eff_scale": EFF_SCALE},
               out_dir / "aeration_twin.pt")
    (out_dir / "aeration_twin_history.json").write_text(
        json.dumps(history, indent=2))
    return {"history": history, "n_transitions": n}


def load_aeration_twin(path: str | Path, device: str = "cpu") -> AerationTwin:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = AerationTwin(n_obs=int(blob["n_obs"])).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
