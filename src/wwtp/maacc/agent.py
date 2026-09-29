"""Multiagent adaptive critic control (MAACC) -- reproduction of the baseline.

Re-implementation of the algorithm in

    D. Wang, X. Li, J. Ren and J. Qiao, "Multiagent Adaptive Critic Control
    With Expert Knowledge for Wastewater Treatment Plants", IEEE Trans. Ind.
    Informat., 2026.

Per subsystem ``i`` one agent holds

* an action network  ``pi_i(z_{k,i}) -> Delta a_{k,i}``     (eqs. 8, 14)
* a critic network   ``Q_i(z_k, Delta a_{k,i}, a_{k,i})``   (eqs. 9, 13)

and the applied input is ``u_{k,i} = a_{k,i} + b_{k,i}`` (eq. 7), where
``b_{k,i}`` comes from the incremental PID expert prior.  The critic sees the
*full* tracking-error vector ``z_k`` so that the utility (eq. 10) charges each
agent for the error it causes in the other subsystem -- the paper's mechanism
for coping with the unknown interconnection term ``G_i(s_k)``.

Timing inside one control period, following Algorithm 1:

    z_k  ->  Delta a_k = pi(z_k)  ->  a_k = a_{k-1} + Delta a_k
         ->  Q_k = Q(z_k, Delta a_k, a_k),  U_k = U(z_k, Delta a_k, a_k)
         ->  critic step on the Bellman residual of eq. (9)
         ->  actor step on   L_a = 0.5 Q_k^2                      (eq. 17)
         ->  u_k = a_k + b_k                                      (eq. 7)

Four implementation choices, all discussed in ``docs/02_reproduction.md``:

1. Network inputs and the control-effort terms of the utility are normalised
   by the actuator ranges.  The paper keeps raw units and absorbs the scale
   into the weight matrices; normalising is equivalent and lets both agents
   share learning rates although ``KLa_5`` spans 0..240 and ``Q_a`` 0..92230.
2. The actor gradient uses the full Jacobian ``dQ/d(Delta a) + dQ/da``, since
   ``a_k = a_{k-1} + Delta a_k`` makes both critic inputs depend on the
   action-network output.
3. Both networks carry a constant bias input.  Without it ``tanh`` networks
   are pinned to zero at the origin and cannot represent a Q-function that is
   strictly positive at the set point.
4. **Critic update.**  Equation (15) writes the approximation error as
   ``e_k = U_k + lambda Q(x_k) - Q(x_{k-1})`` and eq. (18) descends it in the
   weights that produce ``Q(x_k)`` -- i.e. it corrects the *later* prediction.
   For the slowly varying states of a WWTP that recursion has homogeneous
   gain ``1 + l_c (1 - lambda) ||vartheta||^2 > 1`` and the critic diverges;
   we reproduce this in ``tests/test_critic_update.py``.  Theorem 1 only
   bounds the weight error (UUB), which does not exclude a large bound.  We
   therefore descend the *same* Bellman residual of eq. (9) in its standard
   TD(0) direction -- correcting ``Q(x_{k-1})`` towards ``U_k + lambda
   Q(x_k)`` -- which converges to the Bellman fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baselines.pid import IncrementalPID, PAPER_KP, PAPER_KI, PAPER_KD
from .networks import ShallowNet


@dataclass
class AgentConfig:
    """Hyper-parameters of a single adaptive-critic agent."""

    u_min: float
    u_max: float
    u_nominal: float
    da_max: float                  # max |Delta a| per control step [physical]
    a_max: float                   # authority of the learned correction
    z_scale: float = 0.2           # typical tracking-error magnitude [physical]
    w: float = 1.0                 # W_i -- own tracking error
    p: float = 0.25                # P_i -- coupled error of all subsystems
    y: float = 0.02                # Y_i -- increment penalty (normalised)
    r: float = 0.05                # R_i -- control-effort penalty (normalised)
    leak: float = 0.999            # leaky integration of a_k (realises R_i)
    n_hidden_critic: int = 16
    n_hidden_actor: int = 12
    lr_critic: float = 0.02
    lr_actor: float = 0.002
    wd_critic: float = 1e-4        # outer-weight decay of the critic network
    wd_actor: float = 2.0          # outer-weight decay of the action network
    discount: float = 0.90


#: configuration used for every experiment in this repository
def default_agent_configs() -> tuple[AgentConfig, AgentConfig]:
    return (
        AgentConfig(u_min=0.0, u_max=240.0, u_nominal=84.0,
                    da_max=4.0, a_max=24.0, z_scale=0.2),      # agent 1: KLa_5 -> S_O,5
        AgentConfig(u_min=0.0, u_max=92230.0, u_nominal=55338.0,
                    da_max=1500.0, a_max=9223.0, z_scale=0.5),  # agent 2: Q_a -> S_NO,2
    )


class AdaptiveCriticAgent:
    """One agent: an action network plus a critic network."""

    def __init__(self, index: int, cfg: AgentConfig, n_states: int,
                 rng: np.random.Generator, extra_context: int = 0):
        self.i = index
        self.cfg = cfg
        self.n_states = n_states
        self.extra_context = extra_context
        self.z_scale = np.ones(n_states)

        # AN input: own tracking error (+ optional predictive context)
        self.actor = ShallowNet(1 + extra_context, cfg.n_hidden_actor, 1,
                                rng=rng, inner_scale=1.5, zero_outer=True)
        # CN input: full error vector, Delta a, a (+ optional predictive context)
        self.critic = ShallowNet(n_states + 2 + extra_context,
                                 cfg.n_hidden_critic, 1, rng=rng,
                                 inner_scale=0.8)
        self.critic.beta *= 0.05
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self.a = 0.0
        self.da = 0.0
        self.q_prev: float | None = None
        self.x_prev: np.ndarray | None = None

    # -- feature assembly --------------------------------------------------
    def _critic_input(self, z: np.ndarray, da: float, a: float,
                      ctx: np.ndarray | None) -> np.ndarray:
        parts = [np.asarray(z, dtype=float).reshape(-1) / self.z_scale,
                 np.array([da / self.cfg.da_max, a / self.cfg.a_max])]
        if ctx is not None and self.extra_context:
            parts.append(np.asarray(ctx, dtype=float).reshape(-1))
        return np.concatenate(parts)

    def _actor_input(self, z_i: float, ctx: np.ndarray | None) -> np.ndarray:
        zn = z_i / self.cfg.z_scale
        if ctx is not None and self.extra_context:
            return np.concatenate([[zn], np.asarray(ctx, float).reshape(-1)])
        return np.array([zn])

    def utility(self, z: np.ndarray, da: float, a: float) -> float:
        """Utility function of eq. (10)."""
        c = self.cfg
        zn = np.asarray(z, dtype=float) / self.z_scale
        return float(c.w * zn[self.i] ** 2
                     + c.y * (da / c.da_max) ** 2
                     + c.r * (a / c.a_max) ** 2
                     + c.p * float(zn @ zn))

    # -- control -----------------------------------------------------------
    def act(self, z: np.ndarray, ctx: np.ndarray | None = None) -> float:
        """Produce ``Delta a_k`` and integrate it into ``a_k`` (eq. 8)."""
        y = float(self.actor.forward(self._actor_input(z[self.i], ctx))[0])
        self._squash = np.tanh(y)
        self.da = float(self._squash * self.cfg.da_max)
        # leaky integration: the R_i term of eq. (10) penalises a standing
        # offset, so the increment is accumulated with a small decay
        self.a = float(np.clip(self.cfg.leak * self.a + self.da,
                               -self.cfg.a_max, self.cfg.a_max))
        return self.a

    # -- learning ----------------------------------------------------------
    def learn(self, z: np.ndarray, ctx: np.ndarray | None = None) -> dict:
        """Critic step (Bellman residual of eq. 9) then actor step (eqs. 17, 19)."""
        c = self.cfg
        x = self._critic_input(z, self.da, self.a, ctx)
        q = float(self.critic.forward(x)[0])
        util = self.utility(z, self.da, self.a)

        td = 0.0
        if self.x_prev is not None:
            target = util + c.discount * q                      # bootstrapped target
            q_old = float(self.critic.forward(self.x_prev)[0])
            td = q_old - target
            self.critic.sgd_outer(np.array([td]), c.lr_critic,
                                  weight_decay=c.wd_critic)
            q = float(self.critic.forward(x)[0])                # refresh at x_k

        # actor: dL_a/dalpha = Q * dQ/d(Delta a) * d(Delta a)/dalpha
        dq_dx = self.critic.grad_wrt_input()[0]
        n = self.n_states
        dq_dda = dq_dx[n] / c.da_max + dq_dx[n + 1] / c.a_max   # wrt physical increment
        dda_dy = c.da_max * (1.0 - self._squash ** 2)           # tanh squashing
        self.actor.forward(self._actor_input(z[self.i], ctx))
        self.actor.sgd_outer(np.array([q * dq_dda * dda_dy]), c.lr_actor,
                             weight_decay=c.wd_actor)

        self.q_prev = q
        self.x_prev = x
        return {"q": q, "td": td, "utility": util}


class MAACCController:
    """The complete multiagent controller (Algorithm 1)."""

    name = "MAACC"

    def __init__(self, configs=None, seed: int = 0,
                 kp=PAPER_KP, ki=PAPER_KI, kd=PAPER_KD,
                 learn: bool = True, extra_context: int = 0):
        self.configs = tuple(configs) if configs is not None else default_agent_configs()
        self.n_agents = len(self.configs)
        self.seed = seed
        self.learn_enabled = learn
        self.extra_context = extra_context
        self._pid_kwargs = dict(
            kp=kp, ki=ki, kd=kd,
            u_min=np.array([c.u_min for c in self.configs]),
            u_max=np.array([c.u_max for c in self.configs]),
            u0=np.array([c.u_nominal for c in self.configs]))
        self.reset()

    def reset(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.agents = [AdaptiveCriticAgent(i, c, self.n_agents, rng,
                                           self.extra_context)
                       for i, c in enumerate(self.configs)]
        z_scale = np.array([c.z_scale for c in self.configs])
        for agent in self.agents:
            agent.z_scale = z_scale
        self.pid = IncrementalPID(**self._pid_kwargs)
        self.diagnostics = {"a": [], "q": [], "gate": []}

    # -- hooks overridden by the predictive controller ---------------------
    def _context(self, context: dict) -> np.ndarray | None:
        return None

    def _gate(self, context: dict) -> np.ndarray:
        return np.ones(self.n_agents)

    def _feedforward(self, context: dict) -> np.ndarray:
        return np.zeros(self.n_agents)

    # -- control law -------------------------------------------------------
    def act(self, k: int, s: np.ndarray, r: np.ndarray,
            context: dict) -> np.ndarray:
        z = np.asarray(s, dtype=float) - np.asarray(r, dtype=float)   # eq. (6)
        ctx = self._context(context)

        a = np.array([agent.act(z, ctx) for agent in self.agents])    # eq. (8)
        if self.learn_enabled:
            for agent in self.agents:
                agent.learn(z, ctx)

        b = self.pid(r - s)                                           # expert prior
        gate = self._gate(context)
        ff = self._feedforward(context)

        if k % 400 == 0:
            self.diagnostics["a"].append(a.tolist())
            self.diagnostics["q"].append([ag.q_prev for ag in self.agents])
            self.diagnostics["gate"].append(gate.tolist())
        return b + gate * a + ff                                      # eq. (7)
