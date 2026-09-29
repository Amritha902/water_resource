"""Corpus for the effluent-ammonium early-warning model.

Question the model answers, every 6 minutes:

    *Will effluent ammonium exceed the 4 g N/m3 discharge limit within the
    next 12, 30, 60, 90 or 120 minutes -- and how high will it go?*

Only signals a real plant already measures online are used as inputs:

=====================  ==================================================
inlet                  flow, ammonium, readily biodegradable COD (3 % noise)
reactor probes         DO in reactor 5, nitrate in reactor 2 (noisy)
controller outputs     KLa_5 and Q_a -- known exactly, the PLC set them
effluent analyser      ammonium, 10-min cycle (stale in between) + noise
context                water temperature, time of day
upstream forecast      the 6-D context of the trained influent forecaster
=====================  ==================================================

The target is the *true* effluent ammonium, which the analyser only sees late
and noisily.

Why the operating point is randomised
-------------------------------------
Under the reference paper's settings (aerated reactors 3-4 at KLa = 240 1/d,
15 degC) effluent ammonium never exceeds 2.4 g N/m3, so there is nothing to
warn about.  Real plants trim aeration to save energy and lose nitrification
capacity in cold water, and that is where violations come from.  Each rollout
therefore draws

* KLa in reactors 3-4 from 160-240 1/d (energy-trimmed aeration),
* water temperature from 11.5-17 degC, applied to the autotroph growth and
  decay rates with the BSM2 Arrhenius factors,
* the DO set point from 0.8-2.0 g/m3,

on top of the already randomised influent (load level, diurnal shape,
rain/storm events).  About a third of the resulting samples are in
violation; the rest ranges from never violated to episodically violated.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..baselines.pid_controller import PIDController
from ..bsm1 import asm1
from ..bsm1.influent import InfluentSeries, random_scenario
from ..bsm1.plant import PlantParameters, SEC_PER_DAY
from ..envs import run_closed_loop
from ..metrics import EFFLUENT_LIMITS

#: control period of the simulator
DT_SECONDS = 45.0
#: sampling period of the warning model: 8 control periods = 6 min, the
#: same coarse grid as the digital twin
STEP = 8
STEP_MINUTES = STEP * DT_SECONDS / 60.0            # 6 min
#: history fed to the model: 6 h of 6-min samples
LOOKBACK = 60
#: prediction horizons in model steps (12 ... 120 min)
HORIZONS = (2, 5, 10, 15, 20)
#: quantiles of the effluent-ammonium head
QUANTILES = (0.1, 0.5, 0.9)
#: discharge limit [g N/m3]
NH_LIMIT = EFFLUENT_LIMITS["NH"]
#: effluent analyser cycle time [model steps] (12 min)
ANALYSER_CYCLE = 2

#: input channel names, in model order
CHANNELS = (
    "Q_in", "NH_in", "COD_in",                       # inlet instruments
    "DO_5", "NO_2",                                  # reactor probes
    "KLa_5", "Q_a",                                  # controller outputs
    "NH_eff_analyser",                               # effluent analyser
    "temperature", "tod_sin", "tod_cos",             # context
    "fc_dQ_1h", "fc_dQ_2h", "fc_dNH_2h", "fc_dCOD_2h",
    "fc_uncertainty", "fc_p_wet",                    # upstream forecast
)

#: groups used for ablations and attribution
GROUPS = {
    "inlet": ("Q_in", "NH_in", "COD_in"),
    "reactor probes": ("DO_5", "NO_2"),
    "controller outputs": ("KLa_5", "Q_a"),
    "effluent analyser": ("NH_eff_analyser",),
    "temperature": ("temperature",),
    "time of day": ("tod_sin", "tod_cos"),
    "influent forecast": ("fc_dQ_1h", "fc_dQ_2h", "fc_dNH_2h", "fc_dCOD_2h",
                          "fc_uncertainty", "fc_p_wet"),
}


@dataclass
class OperatingPoint:
    kla34: float = 240.0
    temperature: float = 15.0
    do_setpoint: float = 1.0

    def plant_params(self) -> PlantParameters:
        dt = self.temperature - 15.0
        p = dataclasses.replace(asm1.DEFAULT_PARAMS,
                                mu_A=0.5 * np.exp(0.098 * dt),
                                b_A=0.05 * np.exp(0.069 * dt))
        pp = PlantParameters(asm1_params=p)
        pp.kla = np.array([0.0, 0.0, self.kla34, self.kla34, 84.0])
        return pp

    @staticmethod
    def sample(rng: np.random.Generator) -> "OperatingPoint":
        return OperatingPoint(kla34=float(rng.uniform(160.0, 240.0)),
                              temperature=float(rng.uniform(11.5, 17.0)),
                              do_setpoint=float(rng.uniform(0.8, 2.0)))


def simulate(series: InfluentSeries, op: OperatingPoint,
             rng: np.random.Generator, forecaster: str | Path | None) -> dict:
    """Run the PID-controlled plant and record every signal on the 6-min grid.

    Returns ``x`` (T, C) raw measured inputs, ``nh`` (T,) true effluent
    ammonium, plus the operating point.
    """
    res = run_closed_loop(PIDController(), series,
                          setpoints=(op.do_setpoint, 2.0),
                          plant_params=op.plant_params(),
                          sensor_noise_std=np.array([0.05, 0.10]),
                          seed=int(rng.integers(1 << 31)))
    idx = np.arange(0, series.n_steps, STEP)
    t = series.time_days[idx]
    n = len(idx)

    x = np.zeros((n, len(CHANNELS)))
    inlet = np.stack([series.flow, series.composition[:, asm1.S_NH],
                      series.composition[:, asm1.S_S]], axis=1)
    inlet_meas = inlet * np.exp(rng.normal(0.0, 0.03, inlet.shape))
    x[:, 0:3] = inlet_meas[idx]
    x[:, 3] = res.state[idx, 0] + rng.normal(0.0, 0.05, n)
    x[:, 4] = res.state[idx, 1] + rng.normal(0.0, 0.10, n)
    x[:, 5:7] = res.control[idx]

    nh_true = res.effluent[idx, asm1.S_NH]
    reading = nh_true + rng.normal(0.0, 0.1, n)
    held = reading.copy()
    for i in range(n):                       # analyser: refresh every cycle
        held[i] = reading[i] if i % ANALYSER_CYCLE == 0 else held[i - 1]
    # the reading reported at step i is the sample taken one cycle earlier
    x[:, 7] = np.concatenate([[held[0]] * ANALYSER_CYCLE, held[:-ANALYSER_CYCLE]])

    x[:, 8] = op.temperature
    x[:, 9] = np.sin(2 * np.pi * t)
    x[:, 10] = np.cos(2 * np.pi * t)

    if forecaster is not None:
        from ..forecast.predictor import InfluentPredictor
        pred = InfluentPredictor(forecaster, dt_seconds=DT_SECONDS)
        ctx = np.zeros(6)
        stride = pred.stride
        for i, k in enumerate(idx):
            # feed the predictor at its own 15-min cadence, causally
            for kk in range(max(0, k - STEP + 1), k + 1):
                if kk % stride == 0:
                    pred.update(kk, inlet_meas[kk])
                    ctx = pred.context()
            x[i, 11:17] = ctx

    return {"x": x.astype(np.float32), "nh": nh_true.astype(np.float32),
            "t": t, "weather": series.weather, "op": dataclasses.asdict(op)}


def collect(seed: int, days: float = 10.0,
            forecaster: str | Path | None = None,
            op: OperatingPoint | None = None,
            series: InfluentSeries | None = None) -> dict:
    """One randomised rollout (worker entry point)."""
    rng = np.random.default_rng(seed)
    op = op or OperatingPoint.sample(rng)
    series = series or random_scenario(rng, days=days, dt_seconds=DT_SECONDS)
    return simulate(series, op, rng, forecaster)


# -- supervised targets -------------------------------------------------------
def targets(nh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per time step: ammonium at each horizon, and 'violated within h'.

    ``level[i, j]``  = nh[i + h_j]
    ``event[i, j]``  = max(nh[i+1 : i+h_j+1]) > limit
    Steps without a full horizon ahead are marked NaN.
    """
    n = len(nh)
    hmax = max(HORIZONS)
    level = np.full((n, len(HORIZONS)), np.nan, np.float32)
    event = np.full((n, len(HORIZONS)), np.nan, np.float32)
    exceed = nh > NH_LIMIT
    # running "any exceedance in the next m steps" via cumulative sums
    c = np.concatenate([[0], np.cumsum(exceed)])
    for j, h in enumerate(HORIZONS):
        valid = np.arange(n) + hmax < n
        i = np.arange(n)[valid]
        level[valid, j] = nh[i + h]
        event[valid, j] = (c[i + h + 1] - c[i + 1]) > 0
    return level, event


@dataclass
class WarningCorpus:
    x: np.ndarray          # (N, LOOKBACK, C) normalised
    level: np.ndarray      # (N, H) true ammonium at horizons
    event: np.ndarray      # (N, H) violation within horizon
    current: np.ndarray    # (N,) analyser reading at prediction time (raw)
    nh_now: np.ndarray     # (N,) true ammonium at prediction time
    rollout: np.ndarray    # (N,) rollout id
    step: np.ndarray       # (N,) index into the rollout

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, sel) -> "WarningCorpus":
        return WarningCorpus(*[getattr(self, f.name)[sel]
                               for f in dataclasses.fields(self)])


class Normaliser:
    """Per-channel affine scaling; flows and loads are log-transformed."""

    LOG = (0, 1, 2, 6)

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def _pre(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for c in self.LOG:
            x[..., c] = np.log(np.maximum(x[..., c], 1.0))
        return x

    def fit(self, xs: list[np.ndarray]) -> "Normaliser":
        z = self._pre(np.concatenate(xs))
        self.mean = z.mean(0).astype(np.float32)
        self.std = (z.std(0) + 1e-3).astype(np.float32)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((self._pre(x) - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def window(rollouts: list[dict], norm: Normaliser, stride: int = 1,
           burn_in_steps: int = 240) -> WarningCorpus:
    """Cut rollouts into (history, future) pairs.

    The first ``burn_in_steps`` (24 h) are skipped: the influent forecaster
    needs a day of history before its context is meaningful.
    """
    parts = {k: [] for k in ("x", "level", "event", "current", "nh_now",
                             "rollout", "step")}
    for r_id, roll in enumerate(rollouts):
        z = norm(roll["x"])
        level, event = targets(roll["nh"])
        start = max(burn_in_steps, LOOKBACK - 1)
        ok = np.arange(len(z))
        ok = ok[(ok >= start) & ~np.isnan(event[:, 0])][::stride]
        if len(ok) == 0:
            continue
        parts["x"].append(np.stack([z[i - LOOKBACK + 1:i + 1] for i in ok]))
        parts["level"].append(level[ok])
        parts["event"].append(event[ok])
        parts["current"].append(roll["x"][ok, CHANNELS.index("NH_eff_analyser")])
        parts["nh_now"].append(roll["nh"][ok])
        parts["rollout"].append(np.full(len(ok), r_id))
        parts["step"].append(ok)
    return WarningCorpus(**{k: np.concatenate(v) for k, v in parts.items()})
