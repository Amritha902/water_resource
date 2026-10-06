"""Influent generator reproducing the BSM1 dry / rain / storm scenarios.

The official BSM1 distribution ships three 14-day influent files sampled at
15 min.  Those files are not redistributable here, so this module generates
statistically equivalent influent series from the documented BSM1 influent
characteristics: an average dry-weather flow of 18 446 m3/d with a two-peak
diurnal pattern, a weekday/weekend modulation, a sustained dilution event for
rain weather, and two short first-flush events for storm weather.

Two uses:

* ``canonical_scenario("dry"|"rain"|"storm")`` gives the deterministic
  14-day evaluation profiles used for every controller comparison.
* ``random_scenario(rng)`` gives randomised realisations (different diurnal
  amplitudes, load levels, rain/storm timing and intensity) used to build
  the training corpus for the forecasting and digital-twin models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import asm1

SEC_PER_DAY = 86400.0

#: BSM1 dry-weather average influent composition [g/m3] (S_ALK in mol/m3)
DRY_AVERAGE_COMPOSITION = np.array([
    30.0,    # S_I
    69.5,    # S_S
    51.2,    # X_I
    202.32,  # X_S
    28.17,   # X_BH
    0.0,     # X_BA
    0.0,     # X_P
    0.0,     # S_O
    0.0,     # S_NO
    31.56,   # S_NH
    6.95,    # S_ND
    10.59,   # X_ND
    7.0,     # S_ALK
])

DRY_AVERAGE_FLOW = 18446.0

#: components whose mass load follows the "pollutograph" rather than the flow
_LOAD_COMPONENTS = (asm1.S_I, asm1.S_S, asm1.X_I, asm1.X_S, asm1.X_BH,
                    asm1.S_NH, asm1.S_ND, asm1.X_ND)


@dataclass
class InfluentSeries:
    """A sampled influent trajectory."""

    time_days: np.ndarray        # (T,)
    flow: np.ndarray             # (T,)   [m3/d]
    composition: np.ndarray      # (T, 13) [g/m3]
    weather: str

    def __len__(self) -> int:
        return self.time_days.shape[0]

    @property
    def n_steps(self) -> int:
        return self.time_days.shape[0]

    def loads(self) -> np.ndarray:
        """Mass loading rates [g/d] for every component, shape (T, 13)."""
        return self.composition * self.flow[:, None]

    def cod_load(self) -> np.ndarray:
        return asm1.total_cod(self.composition) * self.flow

    def nitrogen_load(self) -> np.ndarray:
        return asm1.total_nitrogen(self.composition) * self.flow


def _diurnal(t: np.ndarray, amp: float, phase: float) -> np.ndarray:
    """Two-peak daily pattern (morning + evening), mean value 1."""
    tod = np.mod(t, 1.0)
    shape = (0.62 * np.sin(2 * np.pi * (tod - 0.30 + phase))
             + 0.38 * np.sin(4 * np.pi * (tod - 0.18 + phase)))
    return 1.0 + amp * shape


def _pollutograph(t: np.ndarray, amp: float, phase: float,
                  sharpness: float, floor: float) -> np.ndarray:
    """Sharpened daily load pattern, mean value 1.

    A municipal pollutograph is markedly peakier than the hydrograph -- the
    morning and evening load peaks are narrow and the night-time load does
    not fall to zero.  Raising the diurnal shape to a power above one
    sharpens the peaks, and the floor keeps the trough physical.  Matching
    this peak-to-mean ratio matters: with a flat pollutograph the BSM1
    discharge limits never bind and the energy/quality trade-off that
    Du et al. (2023) exploit disappears.
    """
    y = np.maximum(_diurnal(t, amp, phase), floor) ** sharpness
    return y / y.mean()


def _weekly(t: np.ndarray, weekend_drop: float) -> np.ndarray:
    """Weekday/weekend modulation; the series starts on a Monday."""
    day = np.floor(np.mod(t, 7.0)).astype(int)
    factor = np.ones_like(t)
    factor[np.isin(day, (5, 6))] = 1.0 - weekend_drop
    return factor


def _rain_event(t: np.ndarray, start: float, duration: float,
                intensity: float) -> np.ndarray:
    """Sustained rain hydrograph: fast rise, slow recession [m3/d]."""
    x = (t - start) / duration
    y = np.zeros_like(t)
    m = (x > 0) & (x < 1.6)
    y[m] = intensity * np.clip(1.0 - np.exp(-6.0 * x[m]), 0, None) * np.exp(-1.2 * x[m])
    return y


def _storm_event(t: np.ndarray, start: float, duration: float,
                 intensity: float) -> np.ndarray:
    """Short high-intensity storm hydrograph [m3/d]."""
    x = (t - start) / duration
    y = np.zeros_like(t)
    m = (x > 0) & (x < 3.0)
    y[m] = intensity * (x[m] ** 1.5) * np.exp(-2.2 * x[m]) * np.e ** 1.5 / 1.05
    return y


def _build(time_days: np.ndarray, weather: str, *,
           flow_amp: float, load_amp: float, weekend_drop: float,
           phase: float, load_scale: float, events: list[tuple],
           noise: float, rng: np.random.Generator | None,
           load_sharpness: float = 1.6) -> InfluentSeries:
    t = time_days
    base_flow = DRY_AVERAGE_FLOW * _diurnal(t, flow_amp, phase) * _weekly(t, weekend_drop)

    wet_flow = np.zeros_like(t)
    first_flush = np.zeros_like(t)
    for kind, start, duration, intensity in events:
        if kind == "rain":
            q = _rain_event(t, start, duration, intensity)
        else:
            q = _storm_event(t, start, duration, intensity)
            # first flush: particulates re-suspended from the sewer at the onset
            ff = np.zeros_like(t)
            m = (t > start) & (t < start + 0.55 * duration)
            ff[m] = np.sin(np.pi * (t[m] - start) / (0.55 * duration)) ** 2
            first_flush = np.maximum(first_flush, ff)
        wet_flow = wet_flow + q

    flow = base_flow + wet_flow

    # pollutant mass loads: sharper than the flow, plus first flush
    load_shape = (_pollutograph(t, load_amp, phase - 0.02, load_sharpness, 0.22)
                  * _weekly(t, 0.6 * weekend_drop))
    if rng is not None and noise > 0:
        n = len(t)
        # smooth multiplicative noise (AR(1)) so the series stays realistic
        w = rng.normal(0.0, 1.0, n)
        a = 0.995
        e = np.empty(n)
        e[0] = w[0]
        for i in range(1, n):
            e[i] = a * e[i - 1] + np.sqrt(1 - a * a) * w[i]
        load_shape = load_shape * np.exp(noise * e)
        flow = flow * np.exp(0.5 * noise * np.roll(e, 37))

    composition = np.tile(DRY_AVERAGE_COMPOSITION, (len(t), 1))
    base_load = DRY_AVERAGE_COMPOSITION * DRY_AVERAGE_FLOW * load_scale
    for idx in _LOAD_COMPONENTS:
        load = base_load[idx] * load_shape
        if idx in (asm1.X_S, asm1.X_I, asm1.X_ND, asm1.X_BH):
            load = load * (1.0 + 1.9 * first_flush)
        composition[:, idx] = load / np.maximum(flow, 1.0)

    composition[:, asm1.S_ALK] = 7.0
    composition[:, asm1.S_O] = 0.0
    composition[:, asm1.S_NO] = 0.0
    composition[:, asm1.X_BA] = 0.0
    composition[:, asm1.X_P] = 0.0
    return InfluentSeries(t, flow, np.maximum(composition, 0.0), weather)


def canonical_scenario(weather: str, days: float = 14.0,
                       dt_seconds: float = 45.0) -> InfluentSeries:
    """Deterministic 14-day evaluation influent for one weather condition."""
    weather = weather.lower()
    n = int(round(days * SEC_PER_DAY / dt_seconds))
    t = np.arange(n) * dt_seconds / SEC_PER_DAY

    if weather == "dry":
        events: list[tuple] = []
    elif weather == "rain":
        events = [("rain", 8.0, 1.6, 12000.0)]
    elif weather == "storm":
        events = [("storm", 8.6, 0.28, 26000.0), ("storm", 10.4, 0.22, 34000.0)]
    else:
        raise ValueError(f"unknown weather condition: {weather!r}")

    return _build(t, weather, flow_amp=0.30, load_amp=0.62, weekend_drop=0.18,
                  phase=0.0, load_scale=1.0, events=events, noise=0.0, rng=None,
                  load_sharpness=1.6)


def random_scenario(rng: np.random.Generator, days: float = 14.0,
                    dt_seconds: float = 45.0,
                    weather: str | None = None) -> InfluentSeries:
    """Randomised influent realisation used to build ML training corpora."""
    n = int(round(days * SEC_PER_DAY / dt_seconds))
    t = np.arange(n) * dt_seconds / SEC_PER_DAY
    if weather is None:
        weather = str(rng.choice(["dry", "rain", "storm"], p=[0.4, 0.3, 0.3]))

    # Event windows are clamped so short horizons stay valid -- the tests
    # build 1-2 day scenarios, and an unclamped window gives an inverted
    # uniform range there.
    def _start(margin: float) -> float:
        lo = min(0.4, 0.2 * days)
        hi = max(days - margin, lo + 1e-3)
        return float(rng.uniform(lo, hi))

    events: list[tuple] = []
    if weather == "rain":
        for _ in range(int(rng.integers(1, 3))):
            duration = float(np.clip(rng.uniform(0.9, 2.4), 0.1, 0.5 * days))
            events.append(("rain", _start(1.5 * duration), duration,
                           float(rng.uniform(6000.0, 17000.0))))
    elif weather == "storm":
        for _ in range(int(rng.integers(1, 4))):
            duration = float(np.clip(rng.uniform(0.12, 0.40), 0.05, 0.3 * days))
            events.append(("storm", _start(2.0 * duration), duration,
                           float(rng.uniform(15000.0, 42000.0))))

    return _build(t, weather,
                  flow_amp=float(rng.uniform(0.20, 0.40)),
                  load_amp=float(rng.uniform(0.45, 0.80)),
                  weekend_drop=float(rng.uniform(0.08, 0.28)),
                  phase=float(rng.uniform(-0.06, 0.06)),
                  load_scale=float(rng.uniform(0.80, 1.25)),
                  events=events,
                  noise=float(rng.uniform(0.04, 0.12)),
                  rng=rng,
                  load_sharpness=float(rng.uniform(1.3, 2.0)))
