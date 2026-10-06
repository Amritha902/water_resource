"""BSM1 plant: five-reactor activated sludge cascade + ten-layer clarifier.

Layout (IWA Benchmark Simulation Model no. 1):

    influent -> [1 anoxic][2 anoxic][3 aerated][4 aerated][5 aerated] -> settler
                    ^                                         |            |-> effluent
                    |<------------- internal recycle Q_a -----|            |-> underflow
                    |<------------- return sludge Q_r ---------------------|
                                                                           |-> waste Q_w

Manipulated variables used by the controllers in this repository:

    u[0] = KLa_5  oxygen transfer coefficient in reactor 5  [1/d], 0..240
    u[1] = Q_a    internal recycle flow rate                [m3/d], 0..92230

Controlled variables:

    s[0] = S_O,5   dissolved oxygen in reactor 5   [g O2/m3]
    s[1] = S_NO,2  nitrate nitrogen in reactor 2   [g N/m3]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import asm1
from .settler import Settler, SettlerParameters

SEC_PER_DAY = 86400.0

#: default manipulated-variable bounds (BSM1)
KLA5_BOUNDS = (0.0, 240.0)
QA_BOUNDS = (0.0, 92230.0)

#: set-points used in the reference paper, r = [S_O,5, S_NO,2]
PAPER_SETPOINTS = (1.0, 2.0)
#: the conventional BSM1 default control set-points, kept for reference
BSM1_SETPOINTS = (2.0, 1.0)


@dataclass
class PlantParameters:
    volumes: np.ndarray = field(
        default_factory=lambda: np.array([1000.0, 1000.0, 1333.0, 1333.0, 1333.0]))
    kla: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 240.0, 240.0, 84.0]))
    q_r: float = 18446.0        # return sludge flow [m3/d]
    q_w: float = 385.0          # waste sludge flow  [m3/d]
    asm1_params: asm1.ASM1Parameters = asm1.DEFAULT_PARAMS
    settler_params: SettlerParameters = field(default_factory=SettlerParameters)


#: BSM1 open-loop steady state, used as the default initial condition.
STEADY_STATE_REACTORS = np.array([
    # S_I   S_S    X_I     X_S    X_BH    X_BA   X_P    S_O      S_NO   S_NH  S_ND  X_ND  S_ALK
    [30.0, 2.81, 1149.0, 82.13, 2552.0, 148.0, 449.0, 0.0043, 5.37, 7.92, 1.22, 5.28, 4.93],
    [30.0, 1.46, 1149.0, 76.39, 2553.0, 148.0, 450.0, 6.3e-5, 3.66, 8.34, 0.88, 5.03, 5.08],
    [30.0, 1.15, 1149.0, 64.85, 2557.0, 149.0, 450.0, 1.72, 6.54, 5.55, 0.83, 4.39, 4.67],
    [30.0, 0.995, 1149.0, 55.69, 2559.0, 150.0, 452.0, 2.43, 9.30, 2.97, 0.77, 3.88, 4.29],
    [30.0, 0.889, 1149.0, 49.31, 2559.0, 150.0, 453.0, 0.491, 10.42, 1.73, 0.69, 3.53, 4.13],
])

STEADY_STATE_SETTLER = np.array(
    [12.50, 18.11, 29.54, 68.98, 356.07, 356.07, 356.07, 356.07, 356.07, 6393.98])


class BSM1Plant:
    """Continuous-time BSM1 plant integrated with a fixed-step explicit scheme."""

    def __init__(self, params: PlantParameters | None = None,
                 substeps: int = 4):
        self.p = params or PlantParameters()
        self.substeps = substeps
        self.settler = Settler(self.p.settler_params)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.z = STEADY_STATE_REACTORS.copy()
        self.settler.x = STEADY_STATE_SETTLER.copy()
        self.z_return = self.z[4].copy()
        self.z_effluent = self.z[4].copy()
        self.t = 0.0

    # -- observation -------------------------------------------------------
    @property
    def s_o5(self) -> float:
        return float(self.z[4, asm1.S_O])

    @property
    def s_no2(self) -> float:
        return float(self.z[1, asm1.S_NO])

    def measurement(self) -> np.ndarray:
        """Controlled variables ``[S_O,5, S_NO,2]``."""
        return np.array([self.s_o5, self.s_no2])

    # -- dynamics ----------------------------------------------------------
    def _derivative(self, z: np.ndarray, z_in: np.ndarray, q_in: float,
                    q_a: float, kla: np.ndarray) -> np.ndarray:
        p = self.p
        q1 = q_in + q_a + p.q_r
        dz = np.empty_like(z)

        inflow = q_in * z_in + q_a * z[4] + p.q_r * self.z_return
        dz[0] = (inflow - q1 * z[0]) / p.volumes[0]
        for k in range(1, 5):
            dz[k] = q1 * (z[k - 1] - z[k]) / p.volumes[k]

        dz += asm1.reaction_rates(z, p.asm1_params)
        dz[:, asm1.S_O] += kla * (p.asm1_params.S_O_sat - z[:, asm1.S_O])
        return dz

    def step(self, dt_seconds: float, z_in: np.ndarray, q_in: float,
             u: np.ndarray) -> np.ndarray:
        """Advance the plant by ``dt_seconds`` under control input ``u``.

        Returns the controlled measurement after the step.
        """
        p = self.p
        kla = p.kla.copy()
        kla[4] = float(np.clip(u[0], *KLA5_BOUNDS))
        q_a = float(np.clip(u[1], *QA_BOUNDS))

        dt = dt_seconds / SEC_PER_DAY
        h = dt / self.substeps
        q_feed = q_in + p.q_r          # flow to the settler
        q_under = p.q_r + p.q_w

        for _ in range(self.substeps):
            self.z = np.maximum(
                self.z + h * self._derivative(self.z, z_in, q_in, q_a, kla), 0.0)
            z_feed = self.z[4]
            self.settler.step(h, float(asm1.tss(z_feed)), q_feed, q_under)
            self.z_effluent, self.z_return = self.settler.split(z_feed)

        self.t += dt
        return self.measurement()

    # -- performance -------------------------------------------------------
    def aeration_energy_rate(self, u: np.ndarray) -> float:
        """Instantaneous aeration energy [kWh/d].

        BSM1 definition, also eq. (20) of Du et al. (2023):
        ``AE = (S_O_sat / 1800) * sum_i V_i KLa_i``.
        """
        kla = self.p.kla.copy()
        kla[4] = float(np.clip(u[0], *KLA5_BOUNDS))
        return float(self.p.asm1_params.S_O_sat
                     * np.sum(self.p.volumes * kla) / 1800.0)

    def pumping_energy_rate(self, q_a: float, q_in: float) -> float:
        """Instantaneous pumping energy [kWh/d].

        BSM1 definition, also eq. (21) of Du et al. (2023):
        ``PE = 0.004 Q_a + 0.008 Q_r + 0.05 Q_w``.
        """
        q_a = float(np.clip(q_a, *QA_BOUNDS))
        return float(0.004 * q_a + 0.008 * self.p.q_r + 0.05 * self.p.q_w)
