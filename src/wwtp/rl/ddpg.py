"""DDPG agent, reproducing Du et al. (2023), Sections 3.3 and 4.2.

Network sizes, optimiser settings and update rules are the published ones:

* actor  ``3 -> 256 -> 256 -> 1``, ReLU hidden, ``tanh`` output (Figure 5);
* critic ``4 -> 256 -> 256 -> 1``, ReLU hidden, linear output;
* ``gamma = 0.99``, batch ``N = 256``, soft-update rate ``eps = 0.001``,
  replay buffer ``R = 30000``, actor lr ``1e-4``, critic lr ``3e-4``;
* critic loss eq. (8)/(9), policy gradient eq. (10), soft updates eq. (11).

The only addition is an Ornstein-Uhlenbeck exploration process, which the
paper refers to only as "noise"; OU is the choice in the original DDPG paper
it cites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DDPGConfig:
    n_obs: int = 3
    n_action: int = 1
    hidden: tuple[int, int] = (256, 256)
    gamma: float = 0.99
    batch_size: int = 256
    tau: float = 0.001                 # eps in eq. (11)
    buffer_size: int = 30000
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    actor_final_gain: float = 0.01     # small last layer -> "do nothing" start
    warmup_steps: int = 1000           # transitions collected before updating
    updates_per_step: int = 1
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_sigma_final: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0


def _mlp(sizes: list[int], out_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Fixed-capacity ring buffer of ``(s, a, r, s', done)`` tuples."""

    def __init__(self, capacity: int, n_obs: int, n_action: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_obs), dtype=np.float32)
        self.act = np.zeros((capacity, n_action), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.nxt = np.zeros((capacity, n_obs), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, d) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.nxt[i], self.done[i] = o2, float(d)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=min(n, self.size))
        return tuple(torch.from_numpy(a[idx]) for a in
                     (self.obs, self.act, self.rew, self.nxt, self.done))


class OUNoise:
    """Ornstein-Uhlenbeck exploration process."""

    def __init__(self, n: int, theta: float, sigma: float,
                 rng: np.random.Generator):
        self.n, self.theta, self.sigma, self.rng = n, theta, sigma, rng
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(self.n)

    def __call__(self, sigma: float | None = None) -> np.ndarray:
        s = self.sigma if sigma is None else sigma
        self.state += -self.theta * self.state + s * self.rng.normal(0.0, 1.0, self.n)
        return self.state.copy()


class DDPGAgent:
    """Deterministic actor-critic agent with target networks."""

    name = "DDPG"

    def __init__(self, cfg: DDPGConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or DDPGConfig()
        c = self.cfg
        torch.manual_seed(c.seed)
        self.device = device
        self.rng = np.random.default_rng(c.seed)

        self.actor = _mlp([c.n_obs, *c.hidden, c.n_action], nn.Tanh()).to(device)
        self.actor_target = _mlp([c.n_obs, *c.hidden, c.n_action], nn.Tanh()).to(device)
        self.critic = _mlp([c.n_obs + c.n_action, *c.hidden, 1]).to(device)
        self.critic_target = _mlp([c.n_obs + c.n_action, *c.hidden, 1]).to(device)
        # The action is an *increment* on KLa_5, so a biased initial policy
        # integrates the actuator straight to a rail.  Shrinking the final
        # actor layer makes the initial policy "hold position", which is the
        # same safeguard the first base paper gets from initialising its
        # action-network outer weights to zero (eq. 7-8).
        with torch.no_grad():
            last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(c.actor_final_gain)
            last.bias.mul_(0.0)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=c.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=c.lr_critic)
        self.buffer = ReplayBuffer(c.buffer_size, c.n_obs, c.n_action)
        self.noise = OUNoise(c.n_action, c.ou_theta, c.ou_sigma, self.rng)
        self.total_steps = 0

    # -- acting ------------------------------------------------------------
    def reset(self) -> None:
        self.noise.reset()

    @torch.no_grad()
    def policy(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self.actor(o)[0].cpu().numpy()

    def act(self, obs: np.ndarray, info: dict | None = None,
            explore: bool = False, sigma: float | None = None) -> float:
        a = self.policy(obs)
        if explore:
            a = a + self.noise(sigma)
        return float(np.clip(a, -1.0, 1.0)[0])

    # -- learning ----------------------------------------------------------
    def observe(self, o, a, r, o2, done) -> None:
        self.buffer.add(np.asarray(o, np.float32), np.asarray([a], np.float32),
                        r, np.asarray(o2, np.float32), done)
        self.total_steps += 1

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.mul_(1.0 - tau).add_(tau * p)

    def update(self) -> dict:
        c = self.cfg
        if self.buffer.size < max(c.batch_size, c.warmup_steps):
            return {}
        stats = {}
        for _ in range(c.updates_per_step):
            o, a, r, o2, d = (t.to(self.device) for t in
                              self.buffer.sample(c.batch_size, self.rng))
            # eq. (9): y = r + gamma Q'(s', mu'(s'))
            with torch.no_grad():
                y = r + c.gamma * (1.0 - d) * self.critic_target(
                    torch.cat([o2, self.actor_target(o2)], dim=-1))
            q = self.critic(torch.cat([o, a], dim=-1))
            loss_c = ((q - y) ** 2).mean()                      # eq. (8)
            self.opt_critic.zero_grad(set_to_none=True)
            loss_c.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
            self.opt_critic.step()

            # eq. (10): maximise Q(s, mu(s))
            loss_a = -self.critic(torch.cat([o, self.actor(o)], dim=-1)).mean()
            self.opt_actor.zero_grad(set_to_none=True)
            loss_a.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
            self.opt_actor.step()

            self._soft_update(self.actor, self.actor_target)     # eq. (11)
            self._soft_update(self.critic, self.critic_target)
            stats = {"loss_critic": float(loss_c.detach()),
                     "loss_actor": float(loss_a.detach()),
                     "q_mean": float(q.detach().mean())}
        return stats

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "cfg": self.cfg.__dict__}

    def load_state_dict(self, blob: dict) -> None:
        self.actor.load_state_dict(blob["actor"])
        self.critic.load_state_dict(blob["critic"])
        # The action is an *increment* on KLa_5, so a biased initial policy
        # integrates the actuator straight to a rail.  Shrinking the final
        # actor layer makes the initial policy "hold position", which is the
        # same safeguard the first base paper gets from initialising its
        # action-network outer weights to zero (eq. 7-8).
        with torch.no_grad():
            last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(c.actor_final_gain)
            last.bias.mul_(0.0)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
