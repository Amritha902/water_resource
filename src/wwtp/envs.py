"""Closed-loop simulation harness: BSM1 plant + influent series + controller.

Besides the idealised setting of the reference paper, the harness can impose
the instrumentation a real plant actually has:

``slew_rate``          maximum change of each actuator per control period.
                       Blowers and recycle pumps cannot step; BSM1 studies
                       that ignore this credit feedback with bandwidth no
                       plant has.
``sensor_tau_seconds`` first-order lag of each sensor (DO probes respond in
                       1-2 min).
``analyzer_period``    zero-order hold on a measurement.  Nitrate analysers
                       are sequential instruments with a cycle time of
                       several minutes, so ``S_NO,2`` is a *stale* reading
                       between cycles.

Rate limits and dead time are exactly the conditions under which feedback
cannot keep up and a forecast can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .bsm1 import asm1
from .bsm1.influent import InfluentSeries
from .bsm1.plant import (BSM1Plant, KLA5_BOUNDS, QA_BOUNDS, PAPER_SETPOINTS,
                         PlantParameters, SEC_PER_DAY)
from . import metrics

U_MIN = np.array([KLA5_BOUNDS[0], QA_BOUNDS[0]])
U_MAX = np.array([KLA5_BOUNDS[1], QA_BOUNDS[1]])


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, k: int, s: np.ndarray, r: np.ndarray,
            context: dict) -> np.ndarray: ...


@dataclass
class RunResult:
    """Everything recorded during one closed-loop run."""

    controller: str
    weather: str
    time_days: np.ndarray
    state: np.ndarray          # (T, 2)  [S_O,5, S_NO,2]
    setpoint: np.ndarray       # (T, 2)
    control: np.ndarray        # (T, 2)  [KLa_5, Q_a]
    influent_flow: np.ndarray  # (T,)
    effluent: np.ndarray       # (T, 13)
    aeration: np.ndarray       # (T,) kWh/d
    pumping: np.ndarray        # (T,) kWh/d
    dt_seconds: float
    extras: dict = field(default_factory=dict)

    @property
    def error(self) -> np.ndarray:
        return self.state - self.setpoint

    def summary(self, burn_in_days: float = 0.0) -> dict:
        burn = int(round(burn_in_days * SEC_PER_DAY / self.dt_seconds))
        dt_days = self.dt_seconds / SEC_PER_DAY
        e = self.error
        out: dict = {"controller": self.controller, "weather": self.weather}
        for i, label in enumerate(("SO5", "SNO2")):
            m = metrics.tracking_metrics(e[:, i], burn_in=burn)
            out[f"IAE_{label}"] = m.iae
            out[f"ISE_{label}"] = m.ise
            out[f"DEVmax_{label}"] = m.dev_max
        q_eff = np.maximum(self.influent_flow - 385.0, 0.0)
        out["EQ"] = metrics.effluent_quality_index(
            self.effluent[burn:], q_eff[burn:], dt_days)
        out["viol_NH"] = metrics.violation_fraction(self.effluent[burn:], "NH")
        out["viol_TN"] = metrics.violation_fraction(self.effluent[burn:], "TN")
        out.update(metrics.energy_indices(self.aeration[burn:],
                                          self.pumping[burn:], dt_days))
        return out


def run_closed_loop(controller: Controller, series: InfluentSeries,
                    setpoints: tuple[float, float] = PAPER_SETPOINTS,
                    dt_seconds: float = 45.0,
                    plant_params: PlantParameters | None = None,
                    actuator_noise_std: np.ndarray | None = None,
                    sensor_noise_std: np.ndarray | None = None,
                    seed: int = 0,
                    warmup_days: float = 1.0,
                    inlet_noise_rel: float = 0.03,
                    slew_rate: np.ndarray | None = None,
                    sensor_tau_seconds: np.ndarray | None = None,
                    analyzer_period_seconds: np.ndarray | None = None,
                    context_fn=None) -> RunResult:
    """Simulate ``controller`` on ``series`` and record the full trajectory.

    ``context_fn(k)`` optionally supplies per-step side information (e.g. an
    influent forecast) that predictive controllers consume.
    """
    rng = np.random.default_rng(seed)
    plant = BSM1Plant(plant_params)
    controller.reset()

    r = np.asarray(setpoints, dtype=float)
    n = series.n_steps

    # Warm the plant up on the first day of the series under the expert PID
    # prior, so every controller starts from the same plant state already at
    # the set point.  Otherwise the comparison is dominated by the start-up
    # transient rather than by disturbance rejection.
    n_warm = int(round(warmup_days * SEC_PER_DAY / dt_seconds))
    if n_warm > 0:
        from .baselines.pid_controller import PIDController
        warm = PIDController()
        s_warm = plant.measurement()
        for k in range(min(n_warm, n)):
            u_warm = np.clip(warm.act(k, s_warm, r, {}), U_MIN, U_MAX)
            s_warm = plant.step(dt_seconds, series.composition[k],
                                float(series.flow[k]), u_warm)

    state = np.empty((n, 2))
    control = np.empty((n, 2))
    effluent = np.empty((n, asm1.N_COMP))
    aeration = np.empty(n)
    pumping = np.empty(n)
    setpoint = np.tile(r, (n, 1))

    s = plant.measurement()
    s_filtered = s.copy()
    s_reported = s.copy()
    u_applied_prev = np.array([84.0, 55338.0])
    alpha = (None if sensor_tau_seconds is None
             else np.clip(dt_seconds / np.maximum(sensor_tau_seconds, 1e-9), 0.0, 1.0))
    hold = (None if analyzer_period_seconds is None
            else np.maximum(np.round(analyzer_period_seconds / dt_seconds), 1.0))

    for k in range(n):
        raw = s.copy()
        if sensor_noise_std is not None:
            raw = raw + rng.normal(0.0, sensor_noise_std)
        if alpha is None:
            s_filtered = raw
        else:
            s_filtered = s_filtered + alpha * (raw - s_filtered)
        if hold is None:
            s_reported = s_filtered
        else:
            refresh = (k % hold.astype(int)) == 0
            s_reported = np.where(refresh, s_filtered, s_reported)
        s_meas = s_reported.copy()

        ctx = {} if context_fn is None else context_fn(k)
        # inlet instrumentation available to predictive controllers
        inlet = np.array([series.flow[k],
                          series.composition[k, asm1.S_NH],
                          series.composition[k, asm1.S_S]])
        if inlet_noise_rel:
            inlet = inlet * np.exp(rng.normal(0.0, inlet_noise_rel, 3))
        ctx.setdefault("inlet", inlet)
        ctx.setdefault("influent_flow", float(series.flow[k]))
        u = np.asarray(controller.act(k, s_meas, r, ctx), dtype=float)
        u = np.clip(u, U_MIN, U_MAX)

        u_applied = u
        if slew_rate is not None:
            u_applied = np.clip(u_applied, u_applied_prev - slew_rate,
                                u_applied_prev + slew_rate)
        if actuator_noise_std is not None:
            u_applied = u_applied + rng.normal(0.0, actuator_noise_std)
        u_applied = np.clip(u_applied, U_MIN, U_MAX)
        u_applied_prev = u_applied

        s = plant.step(dt_seconds, series.composition[k], float(series.flow[k]),
                       u_applied)

        state[k] = s
        control[k] = u
        effluent[k] = plant.z_effluent
        aeration[k] = plant.aeration_energy_rate(u_applied)
        pumping[k] = plant.pumping_energy_rate(u_applied[1], float(series.flow[k]))

    return RunResult(controller=getattr(controller, "name", type(controller).__name__),
                     weather=series.weather,
                     time_days=series.time_days.copy(),
                     state=state, setpoint=setpoint, control=control,
                     influent_flow=series.flow.copy(), effluent=effluent,
                     aeration=aeration, pumping=pumping, dt_seconds=dt_seconds,
                     extras=dict(getattr(controller, "diagnostics", {})))
