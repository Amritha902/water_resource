"""IOQL exactly as printed in Zhao et al., IEEE T-II 21(2), 2025.

* eq. (4)   ``u_k = omega * alpha(e_k) + beta(e_k)``; ``alpha`` = PID prior
  (Kp, Ki, Kd = 100, 20, 1), ``beta`` = action network.  ``omega = 1``, or
  ``omega = 0.99**k`` for the prior-policy study (Figs. 5 and 7).
* Sec. IV  ``e_k = x_k - r_k`` with ``x = S_O,5``; utility
  ``G_k = e_k^2 + 0.01 u_k^2``; ``gamma = 1``; ``h_c = h_a = 15``;
  ``l_c = 0.01``, ``l_a = 0.005``; ``T_c = 50`` critic and ``T_a = 60``
  action-network training steps per control period.
* eq. (7)   ``Q = V^T Psi(W^T [e, u])``, tanh, only ``V`` learns.
* eq. (9)   ``V <- V - l_c gamma Psi_k (gamma V^T Psi_k + G_k - V_{k-1}^T Psi_{k-1})``.
* eq. (10)  ``beta = A^T Theta(S^T e)``, tanh, only ``A`` learns.
* eqs. (12), (13)  ``A <- A - l_a Theta_k (V^T C_k)``,
  ``C_p = 0.5 (1 - Psi_p^2) W_{p, n+1}`` (as printed, with the 1/2).

Not stated in the paper, chosen by us: weights drawn uniformly on [-1, 1];
the action applied in period k is the one after that period's updates;
``u`` limited to the KLa_5 range [0, 240].
"""

from __future__ import annotations

import numpy as np

from ..baselines.pid import IncrementalPID

GAMMA, LC, LA, H, TC, TA = 1.0, 0.01, 0.005, 15, 50, 60
PRIORS = {"default": (100.0, 20.0, 1.0), "K1": (80.0, 15.0, 10.0),
          "K2": (100.0, 30.0, 10.0), "K3": (120.0, 25.0, 10.0)}


class IOQL:
    def __init__(self, seed: int = 0, prior: str = "default", omega_decay: bool = False,
                 u0: float = 84.0, learn: bool = True):
        rng = np.random.default_rng(seed)
        kp, ki, kd = PRIORS[prior]
        self.pid = IncrementalPID(kp=np.array([kp]), ki=np.array([ki]), kd=np.array([kd]),
                                  u_min=np.array([0.0]), u_max=np.array([240.0]),
                                  u0=np.array([u0]))
        self.W = rng.uniform(-1, 1, (2, H))          # critic inner, inputs [e, u]
        self.V = rng.uniform(-1, 1, H)               # critic outer
        self.S = rng.uniform(-1, 1, (1, H))          # action inner, input e
        self.A = rng.uniform(-1, 1, H)               # action outer
        self.omega_decay, self.learn = omega_decay, learn
        self.k = 0
        self.prev = None                             # V_{k-1}^T Psi_{k-1}

    def omega(self) -> float:
        return 0.99 ** self.k if self.omega_decay else 1.0

    def __call__(self, x: float, r: float) -> float:
        e = x - r
        alpha = float(self.pid(np.array([r - x]))[0])
        w = self.omega()
        theta = np.tanh(e * self.S[0])
        beta = float(self.A @ theta)
        u = w * alpha + beta
        if self.learn:
            psi = np.tanh(np.array([e, u]) @ self.W)
            g = e * e + 0.01 * u * u
            if self.prev is not None:
                for _ in range(TC):                  # eq. (9), T_c times
                    err = GAMMA * self.V @ psi + g - self.prev
                    self.V = self.V - LC * GAMMA * psi * err
            for _ in range(TA):                      # eqs. (11)-(13), T_a times
                beta = float(self.A @ theta)
                psi_u = np.tanh(np.array([e, w * alpha + beta]) @ self.W)
                c = 0.5 * (1.0 - psi_u ** 2) * self.W[1]
                self.A = self.A - LA * theta * float(self.V @ c)
            beta = float(self.A @ theta)
            u = w * alpha + beta
            psi = np.tanh(np.array([e, u]) @ self.W)
            self.prev = float(self.V @ psi)
        self.k += 1
        return float(np.clip(u, 0.0, 240.0))


def reference(t_days: float, dynamic: bool) -> float:
    """eq. (33) for the dynamic case, 2 mg/L otherwise."""
    if not dynamic:
        return 2.0
    d = t_days
    if 2 <= d < 4 or 8 <= d < 10:
        return 2.2
    if 4 <= d < 6 or 10 <= d < 12:
        return 1.8
    return 2.0
