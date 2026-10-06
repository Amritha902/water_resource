"""PANDA-RL -- the modified reinforcement-learning network.

Starting point is the DDPG agent of Du et al. (2023).  Reproducing it exposed
three problems (``docs/04_second_paper.md`` Sec. 5): the reward that should
punish under-aeration arrives hours after the action, the penalty weights that
trade energy against discharge limits are hand-tuned by trial and error, and
a scalar critic optimises the *mean* return although what a discharge consent
constrains is the tail.  PANDA-RL changes the network in four ways, each
aimed at one of those problems.

**1. Forecast-conditioned actor and critic.**  Both networks additionally
receive a six-dimensional context from the probabilistic influent forecaster
of ``wwtp.forecast``:

    c_k = [ dQ/Q at 30 min, dQ/Q at 2 h, dNH/NH at 2 h, dCOD/COD at 2 h,
            relative flow uncertainty, P(rain) + P(storm) ]

The policy becomes ``mu(s_k, c_k)`` -- a *disturbance-scheduled* law rather
than pure state feedback.  The agent can cut aeration before the load falls
instead of discovering the consequence hours later.

**2. Distributional critic with a CVaR objective.**  The critic outputs
``n_quantiles`` quantiles of the return distribution ``Z(s, a)`` and is
trained with the quantile Huber loss, against the distributional Bellman
target ``r + gamma Z'(s', mu'(s'))``.  The actor then maximises the
conditional value at risk -- the mean of the worst ``cvar_alpha`` fraction of
returns -- instead of the expectation.  Optimising the mean is what lets a
policy accept rare large exceedances; optimising the tail does not.

**3. Constraints with Lagrange multipliers instead of guessed weights.**  The
reward keeps only the term that is genuinely an objective,

    r_k = -KLa_5 / 240            (aeration energy, affine in AE)

and every discharge limit becomes a *constraint* with its own cost critic
``Q_c`` and multiplier ``lambda``.  Each multiplier is updated by dual ascent
on the measured violation rate,

    lambda <- clip( lambda + eta (rate - budget), 0, lambda_max )

so the operator specifies an allowed violation rate -- "ammonium above
4 g N/m3 at most 5 % of the time" -- rather than guessing the ``0.38`` and
``0.42`` of eqs. (19) and (23).  Because the cost critic is scaled by
``1 - gamma`` its output reads directly as a violation rate, which keeps the
multiplier interpretable.  Unlike reward B of the paper, which had to drop
the total-nitrogen term to stay tractable, several constraints can be active
at once.

**4. Twin-assisted (Dyna) updates.**  A learned one-step model of the aeration
MDP (``wwtp.rl.twin.AerationTwin``) supplies ``dyna_ratio`` synthetic batches
per real one.  Each synthetic transition takes a real state out of the replay
buffer, applies a random action, and lets the twin predict the successor and
the effluent, from which the energy reward and both constraint costs follow
analytically.  This is where sample efficiency has to come from if the agent
is to stay inside a realistic online budget: the reproduction needs about
fifteen simulated weeks of interaction against the seven days the paper
allows, and on a real plant those weeks are spent discharging illegal
effluent.  Requires ``action_mode="absolute"`` (the twin predicts the effect
of an applied ``KLa_5``, not of an increment) and is off by default.

Setting ``n_quantiles = 1``, ``cvar_alpha = 1`` and no constraints and no
forecast recovers plain DDPG, so every component can be ablated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..forecast.predictor import InfluentPredictor
from .ddpg import DDPGAgent, DDPGConfig, _mlp
from .env import NH_LIMIT, TN_LIMIT
from .twin import D_SCALE, EFF_SCALE

#: dimension of the forecast context vector
CONTEXT_DIM = 6


@dataclass
class Constraint:
    """One discharge limit, expressed as a rate budget."""

    key: str                 # field of the env info dict
    limit: float             # discharge limit in physical units
    budget: float            # allowed fraction of time above the limit
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.key

    def cost(self, info: dict) -> float:
        return float(info[self.key] > self.limit)


DEFAULT_CONSTRAINTS = (
    Constraint("NH_peak", NH_LIMIT, 0.05, "S_NH,e"),
    Constraint("TN_peak", TN_LIMIT, 0.15, "N_tot,e"),
)


@dataclass
class PandaRLConfig(DDPGConfig):
    n_quantiles: int = 32
    cvar_alpha: float = 0.3            # fraction of the worst returns to optimise
    huber_kappa: float = 1.0
    constraints: tuple = DEFAULT_CONSTRAINTS
    lambda_lr: float = 2.0             # per-episode dual step
    #: optional intra-episode dual ascent, every this many agent decisions.
    #: Needed whenever the policy gets more gradient steps than the episode
    #: boundary provides dual steps -- with ``dyna_ratio = 3`` the policy
    #: learns roughly four times faster and a once-per-week multiplier simply
    #: cannot keep up: the agent drives aeration down and the constraint only
    #: catches up an episode later.  See docs/03_novelty.md.
    lambda_every: int = 0
    lambda_lr_online: float = 0.05     # dual step for the intra-episode update
    rate_ema_beta: float = 2e-3        # ~500-decision window on the violation rate
    lambda_init: float = 1.0
    lambda_max: float = 30.0
    use_forecast: bool = True
    lr_cost: float = 3e-4
    dyna_ratio: int = 0                # synthetic batches per real batch


class _CostBuffer:
    """Parallel store for the per-constraint costs of each transition."""

    def __init__(self, capacity: int, n_cost: int):
        self.cost = np.zeros((capacity, max(n_cost, 1)), dtype=np.float32)

    def add(self, ptr: int, cost: np.ndarray) -> None:
        if cost.size:
            self.cost[ptr] = cost


class PandaRLAgent(DDPGAgent):
    """Forecast-conditioned, risk-sensitive, constrained actor-critic."""

    name = "PANDA-RL"

    def __init__(self, cfg: PandaRLConfig | None = None,
                 forecaster: str | Path | None = None, device: str = "cpu",
                 twin=None):
        self.pcfg = cfg or PandaRLConfig()
        c = self.pcfg
        self.constraints = tuple(c.constraints)
        self.n_constraints = len(self.constraints)
        self.use_forecast = bool(c.use_forecast and forecaster is not None)
        self.n_context = CONTEXT_DIM if self.use_forecast else 0

        # the extended observation is what the networks and buffer see
        base_cfg = DDPGConfig(**{k: v for k, v in c.__dict__.items()
                                 if k in DDPGConfig.__dataclass_fields__})
        base_cfg.n_obs = c.n_obs + self.n_context
        super().__init__(base_cfg, device=device)
        self.cfg_panda = c
        self.twin = twin

        self.predictor = (InfluentPredictor(forecaster, device=device)
                          if self.use_forecast else None)

        # distributional critic: one output per quantile
        n_in = base_cfg.n_obs + c.n_action
        self.critic = _mlp([n_in, *c.hidden, c.n_quantiles]).to(device)
        self.critic_target = _mlp([n_in, *c.hidden, c.n_quantiles]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                           lr=c.lr_critic)

        # cost critics (one output per constraint) and multipliers
        if self.n_constraints:
            self.cost_critic = _mlp([n_in, *c.hidden, self.n_constraints]).to(device)
            self.cost_critic_target = _mlp([n_in, *c.hidden,
                                            self.n_constraints]).to(device)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.opt_cost = torch.optim.Adam(self.cost_critic.parameters(),
                                             lr=c.lr_cost)
            self.costs = _CostBuffer(c.buffer_size, self.n_constraints)
            self.lmbda = np.full(self.n_constraints, float(c.lambda_init))
        else:
            self.lmbda = np.zeros(0)

        # inlet observation per transition, needed to roll the twin forward
        self.inlets = np.zeros((c.buffer_size, 3), dtype=np.float32)

        taus = (torch.arange(c.n_quantiles, dtype=torch.float32, device=device)
                + 0.5) / c.n_quantiles
        self.taus = taus
        self.n_tail = max(int(round(c.cvar_alpha * c.n_quantiles)), 1)
        self.cost_scale = 1.0 - c.gamma
        self._ctx = np.zeros(self.n_context)
        # running estimate of each violation rate, for the intra-episode dual
        self._rate_ema = np.zeros(max(self.n_constraints, 1))
        self._dual_steps = 0

    # -- forecast context --------------------------------------------------
    def reset(self) -> None:
        super().reset()
        if getattr(self, "predictor", None) is not None:
            self.predictor.reset()
        self._ctx = np.zeros(self.n_context)

    def push_inlet(self, inlet: np.ndarray) -> None:
        if self.predictor is None:
            return
        self.predictor.push(inlet)
        self._ctx = self.predictor.context() if self.predictor.ready else \
            np.zeros(self.n_context)

    def context(self) -> np.ndarray:
        return self._ctx

    def constraint_costs(self, info: dict) -> np.ndarray:
        if not self.n_constraints:
            return np.zeros(0)
        return np.array([con.cost(info) for con in self.constraints])

    # -- storage -----------------------------------------------------------
    def observe(self, o, a, r, o2, done, cost=None, inlet=None) -> None:
        ptr = self.buffer.ptr
        super().observe(o, a, r, o2, done)
        if self.n_constraints:
            c = np.zeros(self.n_constraints) if cost is None else np.asarray(cost)
            self.costs.add(ptr, c.astype(np.float32) * self.cost_scale)
        if inlet is not None:
            self.inlets[ptr] = np.asarray(inlet, np.float32) / D_SCALE
        if self.n_constraints and cost is not None:
            beta = self.cfg_panda.rate_ema_beta
            self._rate_ema = (1.0 - beta) * self._rate_ema + beta * np.asarray(cost)

    def _sample(self, n: int):
        idx = self.rng.integers(0, self.buffer.size, size=min(n, self.buffer.size))
        b = self.buffer
        out = [torch.from_numpy(a[idx]) for a in
               (b.obs, b.act, b.rew, b.nxt, b.done)]
        cost = (torch.from_numpy(self.costs.cost[idx])
                if self.n_constraints else None)
        return (*out, cost)

    # -- risk-sensitive value ---------------------------------------------
    def _cvar(self, z: torch.Tensor) -> torch.Tensor:
        """Mean of the worst ``cvar_alpha`` fraction of the return quantiles."""
        if self.n_tail >= z.shape[-1]:
            return z.mean(dim=-1)
        worst, _ = torch.topk(z, self.n_tail, dim=-1, largest=False)
        return worst.mean(dim=-1)

    def _quantile_huber(self, pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        """``pred``: (B, N) current quantiles; ``target``: (B, N) sampled."""
        c = self.cfg_panda
        td = target.unsqueeze(1) - pred.unsqueeze(2)           # (B, N_pred, N_tgt)
        abs_td = td.abs()
        huber = torch.where(abs_td <= c.huber_kappa,
                            0.5 * td ** 2,
                            c.huber_kappa * (abs_td - 0.5 * c.huber_kappa))
        taus = self.taus.view(1, -1, 1)
        loss = (taus - (td.detach() < 0).float()).abs() * huber / c.huber_kappa
        return loss.sum(dim=1).mean()

    # -- learning ----------------------------------------------------------
    def _learn_batch(self, o, a, r, o2, d, cost) -> dict:
        """One critic / cost-critic / actor step on a batch of transitions."""
        c = self.cfg_panda

        # distributional Bellman target
        with torch.no_grad():
            a2 = self.actor_target(o2)
            z2 = self.critic_target(torch.cat([o2, a2], dim=-1))
            target_z = r + c.gamma * (1.0 - d) * z2
        z = self.critic(torch.cat([o, a], dim=-1))
        loss_c = self._quantile_huber(z, target_z)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
        self.opt_critic.step()

        # cost critics: ordinary TD on the discounted violation indicator
        loss_q = None
        if self.n_constraints and cost is not None:
            with torch.no_grad():
                qc2 = self.cost_critic_target(
                    torch.cat([o2, self.actor_target(o2)], dim=-1))
                target_c = cost + c.gamma * (1.0 - d) * qc2
            qc = self.cost_critic(torch.cat([o, a], dim=-1))
            loss_q = ((qc - target_c) ** 2).mean()
            self.opt_cost.zero_grad(set_to_none=True)
            loss_q.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), c.grad_clip)
            self.opt_cost.step()

        # actor: maximise CVaR(Z) - sum_k lambda_k Q_c,k
        pre = self.actor.pre_activation(o)
        a_pi = torch.tanh(pre)
        z_pi = self.critic(torch.cat([o, a_pi], dim=-1))
        objective = self._cvar(z_pi)
        if self.n_constraints:
            lam = torch.as_tensor(self.lmbda, dtype=torch.float32,
                                  device=self.device)
            qc_pi = self.cost_critic(torch.cat([o, a_pi], dim=-1))
            objective = objective - (qc_pi * lam).sum(dim=-1)
        loss_a = (-objective.mean() + c.pre_act_penalty * (pre ** 2).mean())
        self.opt_actor.zero_grad(set_to_none=True)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
        self.opt_actor.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        if self.n_constraints:
            self._soft_update(self.cost_critic, self.cost_critic_target)

        stats = {"loss_critic": float(loss_c.detach()),
                 "loss_actor": float(loss_a.detach()),
                 "q_mean": float(z.detach().mean()),
                 "cvar": float(self._cvar(z.detach()).mean())}
        if loss_q is not None:
            stats["loss_cost"] = float(loss_q.detach())
        return stats

    def _dyna_batch(self, n: int):
        """Synthetic transitions from the twin, starting at real states.

        The forecast context is carried forward unchanged: over one 15-minute
        decision it barely moves, and the twin cannot predict it anyway.
        """
        c = self.cfg_panda
        k = c.n_obs                     # env observation width, context follows
        idx = self.rng.integers(0, self.buffer.size, size=n)
        x = self.buffer.obs[idx]
        inlet = self.inlets[idx]
        obs_env = torch.from_numpy(x[:, :3])
        d_t = torch.from_numpy(inlet)
        ctx = x[:, k:]

        a_syn = self.rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
        kla = 0.5 * (a_syn + 1.0)                       # absolute action mode
        with torch.no_grad():
            obs2, eff = self.twin(obs_env, torch.from_numpy(kla), d_t)
        obs2 = obs2.clamp(min=0.0).numpy()
        nh = eff[:, 0].numpy() * EFF_SCALE[0]
        tn = eff[:, 1].numpy() * EFF_SCALE[1]

        parts = [obs2]
        if k == 4:
            parts.append(kla)                            # the new actuator state
        if ctx.shape[1]:
            parts.append(ctx)
        x2 = np.concatenate(parts, axis=1).astype(np.float32)

        r = (-kla * self.reward_scale).astype(np.float32)  # energy-only reward
        done = np.zeros((n, 1), np.float32)
        cost = None
        if self.n_constraints:
            raw = {"NH_peak": nh, "TN_peak": tn}
            cols = []
            for con in self.constraints:
                value = raw.get(con.key)
                if value is None:
                    cols.append(np.zeros(n, np.float32))
                else:
                    cols.append((value > con.limit).astype(np.float32))
            cost = (np.stack(cols, axis=1) * self.cost_scale).astype(np.float32)

        to_t = lambda arr: torch.from_numpy(arr).to(self.device)
        return (to_t(x), to_t(a_syn), to_t(r), to_t(x2), to_t(done),
                to_t(cost) if cost is not None else None)

    def update(self) -> dict:
        c = self.cfg_panda
        if self.buffer.size < max(c.batch_size, c.warmup_steps):
            return {}
        stats: dict = {}
        for _ in range(c.updates_per_step):
            o, a, r, o2, d, cost = self._sample(c.batch_size)
            o, a, r, o2, d = (t.to(self.device) for t in (o, a, r, o2, d))
            if cost is not None:
                cost = cost.to(self.device)
            stats = self._learn_batch(o, a, r, o2, d, cost)

            if c.dyna_ratio and self.twin is not None:
                for _ in range(c.dyna_ratio):
                    syn = self._learn_batch(*self._dyna_batch(c.batch_size))
                stats["loss_critic_dyna"] = syn["loss_critic"]

            self._dual_steps += 1
            if (self.n_constraints and c.lambda_every
                    and self._dual_steps % c.lambda_every == 0):
                self._dual_step(self._rate_ema, c.lambda_lr_online)
                stats["lambda_0"] = float(self.lmbda[0])
        return stats

    def _dual_step(self, rates, lr: float) -> None:
        """One projected dual-ascent step on every multiplier."""
        for i, con in enumerate(self.constraints):
            self.lmbda[i] = float(np.clip(
                self.lmbda[i] + lr * (float(rates[i]) - con.budget),
                0.0, self.cfg_panda.lambda_max))

    # -- dual ascent on the multipliers -----------------------------------
    def end_episode(self, stats: dict) -> dict:
        """Update each Lagrange multiplier from the measured violation rate."""
        if not self.n_constraints:
            return {}
        c = self.cfg_panda
        rates = np.array([float(stats.get(f"rate_{con.key}", 0.0))
                          for con in self.constraints])
        # when the intra-episode dual is running it already tracks the rate;
        # stepping again here would double-count the same episode
        if not c.lambda_every:
            self._dual_step(rates, c.lambda_lr)
        out = {}
        for i, con in enumerate(self.constraints):
            out[f"lambda_{con.name}"] = float(self.lmbda[i])
            out[f"rate_{con.name}"] = float(rates[i])
        return out

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        blob = {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "lambda": self.lmbda.tolist()}
        if self.n_constraints:
            blob["cost_critic"] = self.cost_critic.state_dict()
        return blob

    def load_state_dict(self, blob: dict) -> None:
        self.actor.load_state_dict(blob["actor"])
        self.critic.load_state_dict(blob["critic"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.n_constraints and "cost_critic" in blob:
            self.cost_critic.load_state_dict(blob["cost_critic"])
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.lmbda = np.asarray(blob.get("lambda", self.lmbda), dtype=float)
