"""Aeration-control environment, following Du et al. (2023).

The second base paper does *not* track a dissolved-oxygen set point.  The
agent writes increments directly onto the oxygen transfer coefficient,

    KLa_5,t = KLa_5,t-1 + a_t

and ``S_O,5`` is left to float wherever the energy / effluent-quality
trade-off puts it.  Observation and reward follow Sections 3.4 and 3.5 of
that paper:

    state  s_t = [S_S,5, S_O,5, S_NH,5]            (reactor-5 concentrations)
    r_EQ     = -( a1 max(S_NH,e - 4, 0) + a2 max(N_tot,e - 18, 0) )   eq. (19)
    r_AE,EQ  = -( b1 AE_hat        + b2 max(S_NH,e - 4, 0) )          eq. (23)

The nitrate loop is *not* part of the paper's problem.  We keep ``Q_a`` on
the incremental-PID nitrate controller of the first base paper for every
method compared here, so the aeration agents are judged on aeration alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baselines.pid import IncrementalPID
from ..bsm1 import asm1
from ..bsm1.influent import InfluentSeries
from ..bsm1.plant import (BSM1Plant, KLA5_BOUNDS, QA_BOUNDS, PlantParameters,
                          SEC_PER_DAY)

#: reactor index whose concentrations the agent observes (unit 5, 0-based)
OBS_REACTOR = 4
#: BSM1 discharge limits used in the reward
NH_LIMIT = 4.0
TN_LIMIT = 18.0

#: nitrate controller kept identical across every aeration method
NITRATE_KP, NITRATE_KI, NITRATE_KD = 100000.0, 30000.0, 1000.0
NITRATE_SETPOINT = 1.0


@dataclass
class AerationEnvConfig:
    dt_seconds: float = 45.0
    da_max: float = 6.0                 # max |increment| on KLa_5 per step
    kla0: float = 84.0
    warmup_days: float = 1.0
    nitrate_setpoint: float = NITRATE_SETPOINT
    # Du et al. control aeration only and leave Q_a at the BSM1 default, so
    # this is off for the reproduction.  The integrated two-layer experiments
    # switch it on, identically for every aeration method.
    control_nitrate: bool = False
    sensor_noise_std: np.ndarray | None = None
    actuator_noise_std: np.ndarray | None = None
    obs_scale: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 4.0, 5.0]))  # S_S, S_O, S_NH
    #: ``"paper"``   -> [S_S,5, S_O,5, S_NH,5] exactly as Du et al. specify;
    #: ``"augmented"`` appends the actuator position KLa_5/240.  With an
    #: incremental action the actuator is a hidden integrator, so the paper's
    #: observation is not Markov -- see docs/04_second_paper.md.
    obs_mode: str = "paper"
    #: ``"incremental"`` -> KLa_5 += a * da_max (the paper);
    #: ``"absolute"``    -> KLa_5 = (a + 1)/2 * 240
    action_mode: str = "incremental"


class AerationEnv:
    """Single-agent BSM1 aeration environment over one influent series."""

    def __init__(self, series: InfluentSeries,
                 cfg: AerationEnvConfig | None = None,
                 plant_params: PlantParameters | None = None,
                 start_day: float = 0.0, days: float | None = None,
                 seed: int = 0):
        self.series = series
        self.cfg = cfg or AerationEnvConfig()
        self.plant_params = plant_params
        dt = self.cfg.dt_seconds
        self.k0 = int(round(start_day * SEC_PER_DAY / dt))
        n_avail = series.n_steps - self.k0
        self.n_steps = n_avail if days is None else min(
            int(round(days * SEC_PER_DAY / dt)), n_avail)
        self.seed = seed
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        cfg = self.cfg
        self.rng = np.random.default_rng(self.seed)
        self.plant = BSM1Plant(self.plant_params)
        self.nitrate_pid = IncrementalPID(
            kp=np.array([NITRATE_KP]), ki=np.array([NITRATE_KI]),
            kd=np.array([NITRATE_KD]), u_min=np.array([QA_BOUNDS[0]]),
            u_max=np.array([QA_BOUNDS[1]]), u0=np.array([55338.0]))
        self.kla = float(cfg.kla0)
        self.q_a = 55338.0
        self.k = 0

        n_warm = int(round(cfg.warmup_days * SEC_PER_DAY / cfg.dt_seconds))
        for i in range(n_warm):
            j = min(self.k0 + i, self.series.n_steps - 1)
            self._advance(j, self.kla, self.q_a)
        return self.observation()

    # -- plant interface ---------------------------------------------------
    def _advance(self, j: int, kla: float, q_a: float) -> None:
        self.plant.step(self.cfg.dt_seconds, self.series.composition[j],
                        float(self.series.flow[j]), np.array([kla, q_a]))

    @property
    def n_obs(self) -> int:
        return 4 if self.cfg.obs_mode == "augmented" else 3

    def observation(self) -> np.ndarray:
        """``[S_S,5, S_O,5, S_NH,5]`` scaled, optionally plus ``KLa_5/240``."""
        z = self.plant.z[OBS_REACTOR]
        raw = np.array([z[asm1.S_S], z[asm1.S_O], z[asm1.S_NH]])
        if self.cfg.sensor_noise_std is not None:
            raw = raw + self.rng.normal(0.0, self.cfg.sensor_noise_std)
        obs = np.maximum(raw, 0.0) / self.cfg.obs_scale
        if self.cfg.obs_mode == "augmented":
            obs = np.append(obs, self.aeration_fraction)
        return obs

    def raw_observation(self) -> np.ndarray:
        z = self.plant.z[OBS_REACTOR]
        return np.array([z[asm1.S_S], z[asm1.S_O], z[asm1.S_NH]])

    # -- effluent and energy ----------------------------------------------
    def effluent(self) -> dict:
        ze = self.plant.z_effluent
        return {
            "S_NH": float(ze[asm1.S_NH]),
            "N_tot": float(asm1.total_nitrogen(ze)),
            "COD": float(asm1.total_cod(ze)),
            "BOD5": float(asm1.bod5(ze)),
            "TSS": float(asm1.tss(ze)),
            "S_NO": float(ze[asm1.S_NO]),
        }

    def aeration_energy(self) -> float:
        return self.plant.aeration_energy_rate(np.array([self.kla, self.q_a]))

    def pumping_energy(self) -> float:
        return self.plant.pumping_energy_rate(self.q_a, 0.0)

    @property
    def aeration_fraction(self) -> float:
        """``KLa_5`` normalised to [0, 1]; AE is affine in this quantity."""
        return float(self.kla / KLA5_BOUNDS[1])

    # -- MDP step ----------------------------------------------------------
    def step(self, action: float) -> tuple[np.ndarray, dict, bool]:
        """Apply an increment to ``KLa_5`` and advance one control period."""
        cfg = self.cfg
        a = float(np.clip(action, -1.0, 1.0))
        if cfg.action_mode == "absolute":
            self.kla = float(0.5 * (a + 1.0) * KLA5_BOUNDS[1])
        else:
            self.kla = float(np.clip(self.kla + a * cfg.da_max, *KLA5_BOUNDS))

        j = min(self.k0 + self.k, self.series.n_steps - 1)
        if cfg.control_nitrate:
            s_no2 = float(self.plant.z[1, asm1.S_NO])
            self.q_a = float(self.nitrate_pid(
                np.array([cfg.nitrate_setpoint - s_no2]))[0])

        kla_applied, qa_applied = self.kla, self.q_a
        if cfg.actuator_noise_std is not None:
            noise = self.rng.normal(0.0, cfg.actuator_noise_std)
            kla_applied = float(np.clip(self.kla + noise[0], *KLA5_BOUNDS))
            qa_applied = float(np.clip(self.q_a + noise[1], *QA_BOUNDS))

        self._advance(j, kla_applied, qa_applied)
        self.k += 1

        info = self.effluent()
        info.update(AE=self.aeration_energy(), PE=self.pumping_energy(),
                    S_O5=float(self.plant.z[OBS_REACTOR, asm1.S_O]),
                    S_NO2=float(self.plant.z[1, asm1.S_NO]),
                    KLa5=self.kla, Q_a=self.q_a,
                    aeration_fraction=self.aeration_fraction,
                    q_in=float(self.series.flow[j]),
                    inlet=np.array([self.series.flow[j],
                                    self.series.composition[j, asm1.S_NH],
                                    self.series.composition[j, asm1.S_S]]),
                    step=self.k, t_days=float(self.series.time_days[j]))
        return self.observation(), info, self.k >= self.n_steps
