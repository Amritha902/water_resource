"""Paper 1's MAACC with its training repaired: a fair, strong learning baseline.

Built as MAACC is: one agent per loop; each agent's action network sees its
own loop and outputs an increment of the input (incremental control); each
agent's critic sees both loops and both actions (the coupling); the action
network's last layer starts at zero, so the controller starts as its PID
prior (u = a + b, expert knowledge).

What is repaired (the standard fixes for an actor-critic that does not learn,
as in MATD3):
* replay buffer: every transition is kept and re-used (MAACC uses each step once);
* target networks for actors and critics;
* twin critics, the smaller value used, and target-policy smoothing;
* inputs and reward normalised (MAACC's learning signal was ~1e-4);
* the learned increment is bounded (tanh, at most RES_SCALE per period);
* delayed actor updates.

What it does not have, by design, because those are what SAC-IL adds:
imitation of the expert (behaviour cloning) and the entropy term.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sacil import Buffer, mlp

#: observation entries of each loop (Paper1Env._obs): e, e1, e2, u of that loop, flow, flow change
OWN = ((0, 2, 4, 6, 8, 9), (1, 3, 5, 7, 8, 9))


@dataclass
class RepairedMAACCConfig:
    gamma: float = 0.97
    tau: float = 0.005
    lr: float = 3e-4
    batch: int = 256
    buffer: int = 400000
    reward_scale: float = 0.1
    explore_std: float = 0.1       # Gaussian exploration on the action, in [-1, 1] units
    target_std: float = 0.2        # target-policy smoothing
    target_clip: float = 0.5
    policy_delay: int = 2


class AgentActor(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.body = mlp(n_in, 1)
        nn.init.zeros_(self.body[-1].weight)          # starts as the PID prior (MAACC: outer weights 0)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, o):
        return torch.tanh(self.body(o))


class RepairedMAACC:
    """Two agents (oxygen, nitrate); ``active`` selects the loops the learner may correct."""

    def __init__(self, od, active=(True, True), cfg: RepairedMAACCConfig | None = None, seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg or RepairedMAACCConfig()
        self.rng = np.random.default_rng(seed)
        self.active = [i for i in (0, 1) if active[i]]
        self.actors = {i: AgentActor(len(OWN[i])) for i in self.active}
        self.actors_t = {i: copy.deepcopy(a) for i, a in self.actors.items()}
        # each agent's critic sees the whole state and both actions: Q_i(o, a_1, a_2), twin
        self.critics = {i: nn.ModuleList([mlp(od + 2, 1), mlp(od + 2, 1)]) for i in self.active}
        self.critics_t = {i: copy.deepcopy(c) for i, c in self.critics.items()}
        self.opt_a = {i: torch.optim.Adam(self.actors[i].parameters(), lr=self.cfg.lr) for i in self.active}
        self.opt_c = {i: torch.optim.Adam(self.critics[i].parameters(), lr=self.cfg.lr) for i in self.active}
        self.buf = Buffer(self.cfg.buffer, od, 2)
        self.updates = 0

    def _joint(self, o, actors):
        a = torch.zeros(o.shape[0], 2)
        for i in self.active:
            a[:, i] = actors[i](o[:, OWN[i]]).squeeze(-1)
        return a

    @torch.no_grad()
    def act(self, o, deterministic=True):
        a = self._joint(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), self.actors)[0].numpy()
        if not deterministic:
            a[self.active] += self.rng.normal(0.0, self.cfg.explore_std, len(self.active))
        return np.clip(a, -1.0, 1.0)

    def _q(self, critic, o, a):
        x = torch.cat([o, a], -1)
        return critic[0](x).squeeze(-1), critic[1](x).squeeze(-1)

    def update(self):
        if len(self.buf) < self.cfg.batch:
            return
        c = self.cfg
        o, a, r, o2, d = self.buf.sample(c.batch, self.rng)
        with torch.no_grad():
            a2 = self._joint(o2, self.actors_t)
            noise = (torch.randn_like(a2) * c.target_std).clamp(-c.target_clip, c.target_clip)
            mask = torch.zeros(2)
            mask[self.active] = 1.0
            a2 = (a2 + noise * mask).clamp(-1.0, 1.0)
        for i in self.active:
            with torch.no_grad():
                q1t, q2t = self._q(self.critics_t[i], o2, a2)
                y = c.reward_scale * r + c.gamma * (1 - d) * torch.min(q1t, q2t)
            q1, q2 = self._q(self.critics[i], o, a)
            loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.opt_c[i].zero_grad()
            loss.backward()
            self.opt_c[i].step()
        self.updates += 1
        if self.updates % c.policy_delay:
            return
        for i in self.active:
            # agent i changes its own action; the other agent's action is taken from the data
            ai = a.clone()
            ai[:, i] = self.actors[i](o[:, OWN[i]]).squeeze(-1)
            q1, _ = self._q(self.critics[i], o, ai)
            loss = -q1.mean()
            self.opt_a[i].zero_grad()
            loss.backward()
            self.opt_a[i].step()
        with torch.no_grad():
            for i in self.active:
                for net, tgt in ((self.actors[i], self.actors_t[i]), (self.critics[i], self.critics_t[i])):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - c.tau).add_(c.tau * p)

    def state_dict(self):
        return {str(i): self.actors[i].state_dict() for i in self.active}

    def load_state_dict(self, sd):
        for i in self.active:
            self.actors[i].load_state_dict(sd[str(i)])
