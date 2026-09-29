"""PID baseline wrapped in the :class:`~wwtp.envs.Controller` interface."""

from __future__ import annotations

import numpy as np

from .pid import IncrementalPID, PAPER_KP, PAPER_KI, PAPER_KD


class PIDController:
    """The expert prior strategy on its own (no learning)."""

    name = "PID"

    def __init__(self, kp=PAPER_KP, ki=PAPER_KI, kd=PAPER_KD,
                 u_min=(0.0, 0.0), u_max=(240.0, 92230.0),
                 u0=(84.0, 55338.0)):
        self._kwargs = dict(kp=kp, ki=ki, kd=kd, u_min=np.array(u_min),
                            u_max=np.array(u_max), u0=np.array(u0))
        self.pid = IncrementalPID(**self._kwargs)

    def reset(self) -> None:
        self.pid = IncrementalPID(**self._kwargs)

    def act(self, k: int, s: np.ndarray, r: np.ndarray, context: dict) -> np.ndarray:
        return self.pid(r - s)
