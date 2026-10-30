"""SAC with imitation learning (SAC-IL) for paper 1's two loops on the official BSM1.

Same task as paper 1 (MAACC): KLa,5 -> S_O,5 = 2 mg/L and Q_a -> S_NO,2 = 1 mg/L,
45 s control period, actuators limited to [0, 240] 1/d and [0, 92 230] m3/d.
Same expert knowledge as paper 1: its incremental PID.  Different learner:

1. Imitation.  The expert PID is run on dry weather and its increments are
   recorded.  The actor is trained to reproduce them (behaviour cloning), so
   learning starts from the expert and not from random actions.
2. Soft actor-critic.  The policy is then improved by SAC (twin critics,
   entropy term) from the expert demonstrations plus its own experience.  A
   behaviour-cloning term, strong at first and then kept at a floor, keeps
   the policy close to the expert (without it the policy drifts and the
   loops become unstable after a few episodes).

The policy outputs the control increment, as the PID does
(u_k = u_{k-1} + du_k), from the last three errors of both loops, the
previous inputs and the measured influent flow.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines.pid import PAPER_KD, PAPER_KI, PAPER_KP, IncrementalPID
from ..maacc.exact import U_MAX, U_MIN
from ..official.plant import DT_DAYS, OfficialBSM1, load_influent

R = np.array([2.0, 1.0])                 # set points [S_O,5, S_NO,2]
RES_SCALE = np.array([6.0, 3000.0])      # action 1 = this extra increment per period
E_SCALE = np.array([0.05, 0.1])          # error normalisation
NOISE_STD = 0.03 * U_MAX                 # paper 1, Table III


class Paper1Env:
    """Paper 1 control task.  The paper's PID (its expert knowledge) stays in
    the loop; the action in [-1, 1]^2 is the learned extra increment of
    [KLa,5, Q_a] on top of the PID increment, as in paper 1 (u = PID + learned)."""

    obs_dim, act_dim = 11, 2

    #: reward weights on |S_O,5 - 2| and |S_NO,2 - 1|
    W_DEFAULT = (100.0, 50.0)
    #: each loop scaled by the retuned prior's own error, so neither is traded for the other
    W_BALANCED = (1 / 0.004, 1 / 0.002)

    def __init__(self, weather, warm_state, u0, days=None, noise=False, seed=0, ilc=None, weights=None,
                 base_gain=(1.0, 1.0), act_mask=(1.0, 1.0), base_law=None):
        self.w = np.array(self.W_DEFAULT if weights is None else weights, float)
        #: gain of the PID kept in the loop, per loop, and which loops the learner may correct
        self.base_gain, self.mask = np.asarray(base_gain, float), np.asarray(act_mask, float)
        self.fallback_gain = None             # set by a safety layer for one period
        self.weather, self.warm, self.u0 = weather, warm_state, np.asarray(u0, float)
        self.ilc_proto = ilc                  # DailyILC feedforward (copied at every reset)
        #: optional factory of another base controller (e.g. MAACC as printed): law(s, r) -> u
        self.base_law_factory = base_law
        self.days, self.noise = days, noise
        self.rng = np.random.default_rng(1000 + seed)
        d = load_influent(weather)
        self._tq, self._q = d[:, 0], d[:, 15]

    def _qin(self, k):
        i = np.searchsorted(self._tq, k * DT_DAYS, side="right") - 1
        return float(self._q[max(i, 0)])

    def reset(self):
        self.p = OfficialBSM1(self.weather)
        self.p.set_state(self.warm)
        self.n = self.p.n - 1 if self.days is None else min(self.p.n - 1, int(round(self.days / DT_DAYS)))
        self.k = 0
        self.u = self.u0.copy()
        s = self.p.measurement()
        e = s - R
        self.e1, self.e2 = np.zeros(2), np.zeros(2)          # as a freshly started PID
        self.e = e
        self.q1 = self._qin(0)
        self.ilc = None if self.ilc_proto is None else copy.deepcopy(self.ilc_proto)
        self.base_law = None if self.base_law_factory is None else self.base_law_factory()
        return self._obs()

    def _obs(self):
        q = self._qin(self.k)
        o = np.concatenate([self.e / E_SCALE, self.e1 / E_SCALE, self.e2 / E_SCALE,
                            self.u / U_MAX, [q / 20000.0 - 1.0, (q - self.q1) / 2000.0],
                            [0.0]])
        return o[: self.obs_dim].astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        if self.base_law is not None:
            # another base controller computes the input from the current measurement
            self.u = np.clip(self.base_law(self.e + R, R), U_MIN, U_MAX)
            du = None
        elif self.fallback_gain is not None:
            # safety fallback for this period: the demonstrator PID alone, unbounded
            g = np.where(self.mask > 0, self.fallback_gain, self.base_gain)
            du = expert_increment(self.e, self.e1, self.e2, g)
            self.fallback_gain = None
        else:
            du = expert_increment(self.e, self.e1, self.e2, self.base_gain) + self.mask * a * RES_SCALE
        if du is not None:
            self.u = np.clip(self.u + du, U_MIN, U_MAX)
        u_apply = self.u if self.ilc is None else np.clip(self.u + self.ilc.ff(self.k), U_MIN, U_MAX)
        if self.noise:
            u_apply = np.clip(u_apply + self.rng.normal(0.0, NOISE_STD), U_MIN, U_MAX)
        self.q1 = self._qin(self.k)
        s = self.p.step(u_apply)
        if self.ilc is not None:
            self.ilc.record(self.k, R - s, self.u)
        self.k += 1
        self.e2, self.e1, self.e = self.e1, self.e, s - R
        r = -float(np.sum(np.abs(self.e) * self.w)) - 0.05 * float(np.sum(a ** 2))
        fail = not (0.3 < s[0] < 4.0) or not np.all(np.isfinite(s))
        done = fail or self.k >= self.n
        return self._obs(), max(r, -20.0) - (50.0 if fail else 0.0), done, {"s": s, "u": u_apply, "fail": fail}


def expert_increment(e, e1, e2, gain=1.0):
    """Paper 1's incremental PID increment, from errors s - r (the PID uses r - s)."""
    E, E1, E2 = -e, -e1, -e2
    return gain * (PAPER_KP * (E - E1) + PAPER_KI * E + PAPER_KD * (E - 2 * E1 + E2))


def expert_action(env, gain=1.0):
    """Action that makes PID + action equal a demonstrator PID with this gain
    (gain 1: the paper's PID alone, action 0)."""
    du = expert_increment(env.e, env.e1, env.e2, gain - env.base_gain) * env.mask
    return np.clip(du / RES_SCALE, -1.0, 1.0)


# ------------------------------------------------------------------ networks
def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, od, ad):
        super().__init__()
        self.body = mlp(od, 2 * ad)

    def forward(self, o):
        mu, log_std = self.body(o).chunk(2, -1)
        return mu, (log_std - 4.0).clamp(-8, -1)      # small exploration around the policy

    def sample(self, o):
        mu, log_std = self(o)
        std = log_std.exp()
        z = mu + std * torch.randn_like(mu)
        a = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, od, ad):
        super().__init__()
        self.q1, self.q2 = mlp(od + ad, 1), mlp(od + ad, 1)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@dataclass
class SACILConfig:
    gamma: float = 0.97
    tau: float = 0.005
    lr: float = 3e-4
    batch: int = 256
    alpha0: float = 0.005
    target_entropy: float = -6.0
    bc_weight0: float = 2.0
    bc_decay_steps: int = 40000
    bc_floor: float = 0.5          # the imitation term never fades below this (as in TD3+BC)
    buffer: int = 400000
    reward_scale: float = 0.1


class Buffer:
    def __init__(self, n, od, ad):
        self.o, self.a = np.zeros((n, od), np.float32), np.zeros((n, ad), np.float32)
        self.r, self.o2, self.d = np.zeros(n, np.float32), np.zeros((n, od), np.float32), np.zeros(n, np.float32)
        self.n, self.i, self.full = n, 0, False

    def add(self, o, a, r, o2, d):
        j = self.i
        self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j] = o, a, r, o2, d
        self.i = (j + 1) % self.n
        self.full |= self.i == 0

    def __len__(self):
        return self.n if self.full else self.i

    def sample(self, b, rng):
        j = rng.integers(0, len(self), b)
        return [torch.as_tensor(x[j]) for x in (self.o, self.a, self.r, self.o2, self.d)]


class SACIL:
    def __init__(self, od, ad, cfg: SACILConfig | None = None, seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg or SACILConfig()
        self.rng = np.random.default_rng(seed)
        self.actor, self.critic = Actor(od, ad), Critic(od, ad)
        self.critic_t = copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.log_alpha = torch.tensor(np.log(self.cfg.alpha0), requires_grad=True)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        self.buf = Buffer(self.cfg.buffer, od, ad)
        self.demo = Buffer(self.cfg.buffer, od, ad)
        self.updates = 0

    @torch.no_grad()
    def act(self, o, deterministic=True):
        a, _, mean = self.actor.sample(torch.as_tensor(o).unsqueeze(0))
        return (mean if deterministic else a)[0].numpy()

    def pretrain_bc(self, steps=4000):
        """Behaviour cloning on the expert demonstrations."""
        for _ in range(steps):
            o, a, *_ = self.demo.sample(self.cfg.batch, self.rng)
            mu, _ = self.actor(o)
            loss = F.mse_loss(torch.tanh(mu), a)
            self.opt_a.zero_grad()
            loss.backward()
            self.opt_a.step()
        return float(loss.detach())

    def pretrain_critic(self, steps=4000):
        for _ in range(steps):
            self._critic_step(*self._mixed_batch())
            self._soft_update()

    def _mixed_batch(self):
        b = self.cfg.batch
        if len(self.buf) < b:
            return self.demo.sample(b, self.rng)
        o1 = self.demo.sample(b // 4, self.rng)
        o2 = self.buf.sample(b - b // 4, self.rng)
        return [torch.cat([x, y]) for x, y in zip(o1, o2)]

    def _critic_step(self, o, a, r, o2, d):
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(o2)
            q1t, q2t = self.critic_t(o2, a2)
            y = self.cfg.reward_scale * r + self.cfg.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(o, a)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        loss.backward()
        self.opt_c.step()

    def _soft_update(self):
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    def update(self):
        o, a, r, o2, d = self._mixed_batch()
        self._critic_step(o, a, r, o2, d)
        alpha = self.log_alpha.exp().detach()
        an, logp, _ = self.actor.sample(o)
        q1, q2 = self.critic(o, an)
        loss_a = (alpha * logp - torch.min(q1, q2)).mean()
        w = max(self.cfg.bc_floor, self.cfg.bc_weight0 * (1.0 - self.updates / self.cfg.bc_decay_steps))
        if w > 0:
            od, ad, *_ = self.demo.sample(self.cfg.batch, self.rng)
            mu, _ = self.actor(od)
            loss_a = loss_a + w * F.mse_loss(torch.tanh(mu), ad) * 100.0
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        loss_al = -(self.log_alpha * (logp.detach() + self.cfg.target_entropy)).mean()
        self.opt_al.zero_grad()
        loss_al.backward()
        self.opt_al.step()
        self._soft_update()
        self.updates += 1


def run_expert(env, agent=None, gain=1.0):
    """Run the expert PID through an episode; store its transitions as demonstrations."""
    o, done, total = env.reset(), False, 0.0
    while not done:
        a = expert_action(env, gain)
        o2, r, done, _ = env.step(a)
        if agent is not None:
            agent.demo.add(o, a, r, o2, float(done))
        o, total = o2, total + r
    return total


def evaluate(env, policy, record_every=4, test_from_day=7.0):
    """Run a policy over the episode.

    Tracking (IAE, ISE, DEVmax) of both loops over the whole run and over the
    test window (from ``test_from_day``: the second week, never used for tuning
    or training), and the trade-offs: aeration and pumping energy, actuator
    movement, time at the actuator limits, effluent ammonium and total
    nitrogen, and the controller's computing time per period."""
    import time
    o, done = env.reset(), False
    E, U, S, EFF, T = [], [], [], [], []
    while not done:
        t0 = time.perf_counter()
        a = policy(o, env)
        T.append(time.perf_counter() - t0)
        o, _, done, info = env.step(a)
        E.append(info["s"] - R)
        U.append(info["u"])
        S.append(info["s"])
        y = env.p.sim.ys_eff
        EFF.append((y[9], y[8] + y[9] + y[10] + y[11] + 0.08 * (y[4] + y[5]) + 0.06 * (y[6] + y[2])))
    E, U, EFF = np.array(E), np.array(U), np.array(EFF)
    k0 = min(int(round(test_from_day / DT_DAYS)), len(E) - 1)
    out = {"steps": len(E), "fail": bool(info["fail"])}
    for i, n in enumerate(("SO5", "SNO2")):
        for tag, x in (("", E[:, i]), ("_test", E[k0:, i])):
            out[f"IAE_{n}{tag}"] = float(np.mean(np.abs(x)))
            out[f"ISE_{n}{tag}"] = float(np.mean(x ** 2))
            out[f"DEVmax_{n}{tag}"] = float(np.max(np.abs(x)))
    days = len(U) * DT_DAYS
    out["KLa_move_per_day"] = float(np.abs(np.diff(U[:, 0])).sum() / days)
    out["Qa_move_per_day"] = float(np.abs(np.diff(U[:, 1])).sum() / days)
    out["AE"] = float(np.mean(8 / 1800 * (1333 * 240 + 1333 * 240 + 1333 * U[:, 0])))   # kWh/d
    out["PE_Qa"] = float(np.mean(0.004 * U[:, 1]))                                       # kWh/d, recycle part
    out["at_limit_KLa"] = float(np.mean((U[:, 0] <= U_MIN[0] + 1e-6) | (U[:, 0] >= U_MAX[0] - 1e-6)))
    out["at_limit_Qa"] = float(np.mean((U[:, 1] <= U_MIN[1] + 1e-6) | (U[:, 1] >= U_MAX[1] - 1e-6)))
    out["eff_SNH"] = float(EFF[:, 0].mean())
    out["eff_TN"] = float(EFF[:, 1].mean())
    out["eff_SNH_over_4"] = float(np.mean(EFF[:, 0] > 4.0))
    out["eff_TN_over_18"] = float(np.mean(EFF[:, 1] > 18.0))
    out["ms_per_step"] = float(1000 * np.mean(T))
    out["_trace"] = {"s": np.array(S)[::record_every].tolist(), "u": U[::record_every].tolist()}
    return out
