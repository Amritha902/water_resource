"""Fixed-set-point aeration baselines used by Du et al. (2023).

Both hold ``S_O,5`` at 2 mg/L by manipulating ``KLa_5``; they differ only in
how the increment is computed.  Their outputs are increments scaled to
[-1, 1] so they plug into :class:`~wwtp.rl.env.AerationEnv` exactly like a
learned policy.
"""

from __future__ import annotations

import numpy as np

#: PID gains reported by Du et al., Section 4.3
PAPER2_KP, PAPER2_KI, PAPER2_KD = 200.0, 15.0, 2.0
DO_SETPOINT = 2.0


class PIDAeration:
    """Incremental PID on ``S_O,5`` (the paper's PID comparator)."""

    name = "PID"

    def __init__(self, setpoint: float = DO_SETPOINT, kp: float = PAPER2_KP,
                 ki: float = PAPER2_KI, kd: float = PAPER2_KD,
                 da_max: float = 6.0, obs_scale_do: float = 4.0):
        self.setpoint, self.kp, self.ki, self.kd = setpoint, kp, ki, kd
        self.da_max = da_max
        self.obs_scale_do = obs_scale_do
        self.reset()

    def reset(self) -> None:
        self.e1 = 0.0
        self.e2 = 0.0

    def act(self, obs: np.ndarray, info: dict | None = None,
            explore: bool = False) -> float:
        s_o = float(obs[1]) * self.obs_scale_do   # undo the observation scaling
        e = self.setpoint - s_o
        du = (self.kp * (e - self.e1) + self.ki * e
              + self.kd * (e - 2.0 * self.e1 + self.e2))
        self.e2, self.e1 = self.e1, e
        return float(np.clip(du / self.da_max, -1.0, 1.0))


class FuzzyAeration:
    """Mamdani fuzzy DO controller (the paper's fuzzy comparator).

    Two inputs (error, error rate), five triangular membership functions
    each, the standard symmetric rule table for a set-point tracker, and
    centre-of-gravity defuzzification.
    """

    name = "Fuzzy"

    _CENTRES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    #: rule table over (error, error-rate) -> output level index
    _RULES = np.array([
        [-1.0, -1.0, -1.0, -0.5, 0.0],
        [-1.0, -1.0, -0.5, 0.0, 0.5],
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [-0.5, 0.0, 0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0, 1.0, 1.0],
    ])

    def __init__(self, setpoint: float = DO_SETPOINT, e_scale: float = 0.6,
                 de_scale: float = 0.08, gain: float = 1.0,
                 obs_scale_do: float = 4.0):
        self.setpoint = setpoint
        self.e_scale, self.de_scale, self.gain = e_scale, de_scale, gain
        self.obs_scale_do = obs_scale_do
        self.reset()

    def reset(self) -> None:
        self.e1 = 0.0

    @staticmethod
    def _membership(x: float) -> np.ndarray:
        """Triangular memberships on the five centres, width 0.5."""
        mu = np.clip(1.0 - np.abs(x - FuzzyAeration._CENTRES) / 0.5, 0.0, 1.0)
        total = mu.sum()
        return mu / total if total > 0 else mu

    def act(self, obs: np.ndarray, info: dict | None = None,
            explore: bool = False) -> float:
        s_o = float(obs[1]) * self.obs_scale_do
        e = np.clip((self.setpoint - s_o) / self.e_scale, -1.0, 1.0)
        de = np.clip((e - self.e1) / self.de_scale, -1.0, 1.0)
        self.e1 = e
        mu_e, mu_de = self._membership(e), self._membership(de)
        out = float(mu_e @ self._RULES @ mu_de)
        return float(np.clip(self.gain * out, -1.0, 1.0))


class ConstantAeration:
    """Hold ``KLa_5`` wherever it starts -- the open-loop BSM1 reference."""

    name = "OpenLoop"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict | None = None,
            explore: bool = False) -> float:
        return 0.0
