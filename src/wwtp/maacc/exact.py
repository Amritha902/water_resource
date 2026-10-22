"""MAACC exactly as printed in Wang et al. (2026) -- no tuning of our own.

Everything below is taken from the paper:

* eq. (7)/(8)  ``u = a + b``, ``Delta a = pi(z_i)``, ``a_k = a_{k-1} + Delta a``;
  ``a`` is not clipped (only the applied input is limited to the actuator range).
* eq. (10) and Table I  ``U_i = W_i z_i^2 + Y_i Delta a^2 + R_i a^2 + z^T P_i z`` in raw,
  unnormalised units: W = (1, 2), Y = (0.5, 0.05), R = (1e-4, 1e-8),
  P_1 = 2 I, P_2 = I.
* eqs. (13)/(14)  single-hidden-layer networks without bias,
  ``Q = beta^T vartheta(theta^T [z_i, Delta a_i, a_i])``,
  ``pi = alpha^T delta(eta^T z_i)``; only the outer weights learn; the AN outer
  weights start at zero.  The activation is the bipolar sigmoid
  ``(1 - e^-x) / (1 + e^-x)``, whose derivative ``(1 - vartheta^2) / 2`` is the
  one in the matrix ``C`` of eq. (27).
* Table I  lambda = 0.95, l_c = 0.01, l_a = 0.005, h_c = (12, 14), h_a = (10, 12).
* eqs. (15), (16), (18)  ``e_k = U_k + lambda Q(x_k) - Q(x_{k-1})``, descend ``e^2 / 2``.
  As printed (and as Lemma 1 analyses it) the gradient is taken through
  ``Q(x_k)``: ``beta <- beta - l_c e lambda vartheta_k``  (``critic="paper"``).
  Because that recursion has gain > 1 (docs/REPRODUCTION.md), two standard
  readings are offered for comparison: ``"td0"`` (through ``Q(x_{k-1})``) and
  ``"residual"`` (through both).
* eqs. (17), (19), (27)  ``L_a = Q^2 / 2``; the gradient reaches ``alpha`` through
  ``dQ/da`` (the ``a``-columns of ``theta`` in ``C``).

Not given in the paper, chosen by us: the initial inner weights (uniform on
[-1, 1], the paper says "any nonzero value") and the initial critic outer
weights (uniform on [-0.1, 0.1]).
"""

from __future__ import annotations

import numpy as np

from ..baselines.pid import IncrementalPID, PAPER_KD, PAPER_KI, PAPER_KP

W = (1.0, 2.0)
Y = (0.5, 0.05)
R = (1e-4, 1e-8)
P = (2.0 * np.eye(2), np.eye(2))
LAMBDA, LC, LA = 0.95, 0.01, 0.005
HC, HA = (12, 14), (10, 12)
U_MIN, U_MAX = np.zeros(2), np.array([240.0, 92230.0])


def bipolar(x):
    """(1 - e^-x) / (1 + e^-x), written as tanh(x/2) so it cannot overflow."""
    return np.tanh(0.5 * np.asarray(x))


class ExactAgent:
    def __init__(self, i: int, rng: np.random.Generator, critic: str = "paper",
                 init: tuple[float, float, float] = (1.0, 0.1, 1.0)):
        self.i, self.critic = i, critic
        th, be, et = init                                # not given by the paper
        self.theta = rng.uniform(-th, th, (HC[i], 3))    # CN inner, inputs [z_i, da, a]
        self.beta = rng.uniform(-be, be, HC[i])          # CN outer
        self.eta = rng.uniform(-et, et, HA[i])           # AN inner, input z_i
        self.alpha = np.zeros(HA[i])                     # AN outer: starts at zero
        self.a = 0.0
        self.prev = None                                 # vartheta_{k-1}, Q_{k-1}

    def utility(self, z, da, a):
        i = self.i
        return W[i] * z[i] ** 2 + Y[i] * da ** 2 + R[i] * a ** 2 + z @ P[i] @ z

    def step(self, z: np.ndarray) -> float:
        """One period of Algorithm 1 for this agent; returns ``a_k``."""
        zi = z[self.i]
        delta = bipolar(self.eta * zi)
        da = float(self.alpha @ delta)                   # eq. (14)
        a = self.a + da                                  # eq. (8)
        x = np.array([zi, da, a])
        vt = bipolar(self.theta @ x)
        q = float(self.beta @ vt)                        # eq. (13)
        u_k = self.utility(z, da, a)
        if self.prev is not None:                        # eqs. (15), (16), (18)
            vt_prev, q_prev = self.prev
            e = u_k + LAMBDA * q - q_prev
            if self.critic == "paper":
                g = e * LAMBDA * vt
            elif self.critic == "td0":
                g = -e * vt_prev
            else:                                        # residual gradient
                g = e * (LAMBDA * vt - vt_prev)
            self.beta = self.beta - LC * g
            q = float(self.beta @ vt)
        # eqs. (17), (19), (27): dL_a/dalpha = Q * (beta^T C) * delta
        dq_da = float(self.beta @ (0.5 * (1.0 - vt ** 2) * self.theta[:, 2]))
        self.alpha = self.alpha - LA * q * dq_da * delta
        self.prev = (vt, q)
        self.a = a
        return a


class ExactMAACC:
    """Two exact agents on the paper's incremental PID prior."""

    def __init__(self, seed: int = 0, critic: str = "paper", u0=None,
                 init: tuple[float, float, float] = (1.0, 0.1, 1.0)):
        rng = np.random.default_rng(seed)
        self.agents = [ExactAgent(i, rng, critic, init) for i in range(2)]
        self.pid = IncrementalPID(PAPER_KP, PAPER_KI, PAPER_KD, U_MIN, U_MAX,
                                  u0=np.array([84.0, 55338.0]) if u0 is None else u0)

    def __call__(self, s: np.ndarray, r: np.ndarray) -> np.ndarray:
        z = s - r                                        # eq. (6)
        a = np.array([ag.step(z) for ag in self.agents])
        b = self.pid(r - s)
        return b + a                                     # eq. (7)
