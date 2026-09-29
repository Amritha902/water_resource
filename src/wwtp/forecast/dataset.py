"""Corpus construction for the influent forecaster.

The forecaster sees what a real plant can actually measure at the inlet --
flow rate, ammonium and readily biodegradable COD -- sampled every 15 min,
and must predict the next two hours.  Training scenarios are randomised
realisations from :mod:`wwtp.bsm1.influent`; the three canonical BSM1
profiles are never used for training so that every reported control result
is an out-of-sample evaluation of the forecaster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bsm1 import asm1
from ..bsm1.influent import InfluentSeries, random_scenario

#: observable inlet channels, in the order the models consume them
CHANNELS = ("Q_in", "S_NH_in", "S_S_in")
CHANNEL_INDEX = (None, asm1.S_NH, asm1.S_S)

#: sensor sampling period of the inlet instruments
FORECAST_DT_SECONDS = 900.0
#: samples of history fed to the encoder (24 h)
LOOKBACK = 96
#: samples predicted ahead (2 h)
HORIZON = 8
#: quantile levels of the probabilistic head
QUANTILES = (0.1, 0.5, 0.9)
#: weather regimes used by the auxiliary classification head
REGIMES = ("dry", "rain", "storm")


def observable(series: InfluentSeries, dt_seconds: float = FORECAST_DT_SECONDS,
               source_dt_seconds: float = 45.0) -> np.ndarray:
    """Down-sample a series to the inlet-sensor grid, shape (T, 3)."""
    stride = max(1, int(round(dt_seconds / source_dt_seconds)))
    q = series.flow[::stride]
    out = np.empty((q.shape[0], len(CHANNELS)))
    out[:, 0] = q
    for j, idx in enumerate(CHANNEL_INDEX[1:], start=1):
        out[:, j] = series.composition[::stride, idx]
    return out


def add_sensor_noise(x: np.ndarray, rng: np.random.Generator,
                     rel_std: float = 0.03) -> np.ndarray:
    return x * np.exp(rng.normal(0.0, rel_std, size=x.shape))


@dataclass
class ForecastCorpus:
    """Windowed supervised corpus for multi-horizon forecasting."""

    x: np.ndarray          # (N, LOOKBACK, C) normalised inputs
    y: np.ndarray          # (N, HORIZON, C) normalised targets
    regime: np.ndarray     # (N,) int labels
    mean: np.ndarray       # (C,) log-space normalisation
    std: np.ndarray        # (C,)

    def __len__(self) -> int:
        return self.x.shape[0]


def _windows(obs: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    n = obs.shape[0] - LOOKBACK - HORIZON
    starts = np.arange(0, max(n, 0), stride)
    xs = np.stack([obs[s:s + LOOKBACK] for s in starts]) if len(starts) else np.zeros((0, LOOKBACK, obs.shape[1]))
    ys = np.stack([obs[s + LOOKBACK:s + LOOKBACK + HORIZON] for s in starts]) if len(starts) else np.zeros((0, HORIZON, obs.shape[1]))
    return xs, ys


def build_corpus(n_scenarios: int = 160, seed: int = 0, stride: int = 2,
                 days: float = 14.0, noisy: bool = True,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None) -> ForecastCorpus:
    """Simulate ``n_scenarios`` random influent series and window them."""
    rng = np.random.default_rng(seed)
    xs, ys, regimes = [], [], []
    for _ in range(n_scenarios):
        series = random_scenario(rng, days=days)
        obs = observable(series)
        if noisy:
            obs = add_sensor_noise(obs, rng)
        xw, yw = _windows(np.log(np.maximum(obs, 1e-3)), stride)
        if len(xw) == 0:
            continue
        xs.append(xw)
        ys.append(yw)
        regimes.append(np.full(len(xw), REGIMES.index(series.weather)))

    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    regime = np.concatenate(regimes).astype(np.int64)

    if mean is None or std is None:
        mean = x.reshape(-1, x.shape[-1]).mean(0)
        std = x.reshape(-1, x.shape[-1]).std(0) + 1e-6
    x = (x - mean) / std
    y = (y - mean) / std
    return ForecastCorpus(x, y, regime, np.asarray(mean, np.float32),
                          np.asarray(std, np.float32))
