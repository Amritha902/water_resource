"""Incremental (velocity form) PID -- the expert prior control strategy.

This is the ``b_{k,i}`` term of eq. (7) in the reference paper.  The gains are
the ones reported in the paper's experimental section:

    Kp = diag{100, 100000},  Ki = diag{20, 30000},  Kd = diag{10, 1000}
"""

from __future__ import annotations

import numpy as np

PAPER_KP = np.array([100.0, 100000.0])
PAPER_KI = np.array([20.0, 30000.0])
PAPER_KD = np.array([10.0, 1000.0])


class IncrementalPID:
    """Velocity-form PID with output saturation and anti-windup by clamping.

    ``Delta u_k = Kp (e_k - e_{k-1}) + Ki e_k + Kd (e_k - 2 e_{k-1} + e_{k-2})``
    """

    def __init__(self, kp: np.ndarray = PAPER_KP, ki: np.ndarray = PAPER_KI,
                 kd: np.ndarray = PAPER_KD,
                 u_min: np.ndarray | None = None,
                 u_max: np.ndarray | None = None,
                 u0: np.ndarray | None = None,
                 du_max: np.ndarray | None = None):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.n = self.kp.shape[0]
        self.u_min = np.zeros(self.n) if u_min is None else np.asarray(u_min, float)
        self.u_max = (np.full(self.n, np.inf) if u_max is None
                      else np.asarray(u_max, float))
        self.du_max = None if du_max is None else np.asarray(du_max, float)
        self.u0 = (0.5 * (self.u_min + self.u_max) if u0 is None
                   else np.asarray(u0, dtype=float))
        self.reset()

    def reset(self) -> None:
        self.u = self.u0.copy()
        self.e1 = np.zeros(self.n)
        self.e2 = np.zeros(self.n)

    def delta(self, error: np.ndarray) -> np.ndarray:
        """Return the PID increment for the set-point error ``e = r - s``."""
        e = np.asarray(error, dtype=float)
        du = (self.kp * (e - self.e1) + self.ki * e
              + self.kd * (e - 2.0 * self.e1 + self.e2))
        if self.du_max is not None:
            du = np.clip(du, -self.du_max, self.du_max)
        return du

    def update_history(self, error: np.ndarray) -> None:
        self.e2 = self.e1.copy()
        self.e1 = np.asarray(error, dtype=float).copy()

    def __call__(self, error: np.ndarray) -> np.ndarray:
        du = self.delta(error)
        self.u = np.clip(self.u + du, self.u_min, self.u_max)
        self.update_history(error)
        return self.u.copy()
