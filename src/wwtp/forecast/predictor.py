"""Online wrapper turning the trained forecaster into a control signal.

The controller calls :meth:`InfluentPredictor.update` once per control step
with the newest inlet measurement.  Internally the predictor keeps a 24 h
ring buffer on the 15-minute sensor grid and re-runs the network whenever a
new sensor sample completes, which is far cheaper than running it every
45 s control period and matches how a plant would deploy it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import (CHANNELS, FORECAST_DT_SECONDS, HORIZON, LOOKBACK,
                      QUANTILES, REGIMES)
from .models import InfluentForecaster


class InfluentPredictor:
    """Rolling probabilistic forecast of the observable inlet channels."""

    def __init__(self, checkpoint: str | Path, dt_seconds: float = 45.0,
                 device: str = "cpu"):
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = InfluentForecaster().to(device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.mean = np.asarray(blob["mean"], dtype=np.float32)
        self.std = np.asarray(blob["std"], dtype=np.float32)
        self.device = device
        self.stride = max(1, int(round(FORECAST_DT_SECONDS / dt_seconds)))
        self.reset()

    def reset(self) -> None:
        self.buffer: list[np.ndarray] = []
        self._median = np.zeros((HORIZON, len(CHANNELS)))
        self._spread = np.zeros((HORIZON, len(CHANNELS)))
        self._regime = np.zeros(len(REGIMES))
        self._last = np.zeros(len(CHANNELS))
        self._ready = False

    # -- online interface --------------------------------------------------
    def update(self, k: int, measurement: np.ndarray) -> None:
        """Feed one inlet measurement ``[Q_in, S_NH_in, S_S_in]``."""
        m = np.asarray(measurement, dtype=float)
        self._last = m
        if k % self.stride:
            return
        self.buffer.append(np.log(np.maximum(m, 1e-3)))
        if len(self.buffer) > LOOKBACK:
            self.buffer.pop(0)
        if len(self.buffer) < LOOKBACK:
            return
        self._infer()

    @torch.no_grad()
    def _infer(self) -> None:
        x = (np.stack(self.buffer).astype(np.float32) - self.mean) / self.std
        q, logits, _ = self.model(torch.from_numpy(x[None]).to(self.device))
        q = q[0].cpu().numpy() * self.std[None, :, None] + self.mean[None, :, None]
        q = np.exp(q)                                   # back to physical units
        self._median = q[..., QUANTILES.index(0.5)]
        self._spread = q[..., -1] - q[..., 0]
        self._regime = torch.softmax(logits[0], -1).cpu().numpy()
        self._ready = True

    # -- signals consumed by the controller --------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def median(self) -> np.ndarray:
        """Median forecast, shape (HORIZON, 3)."""
        return self._median

    @property
    def spread(self) -> np.ndarray:
        """q90 - q10, shape (HORIZON, 3)."""
        return self._spread

    @property
    def regime(self) -> np.ndarray:
        """Posterior over (dry, rain, storm)."""
        return self._regime

    def relative_change(self, horizon: int | None = None) -> np.ndarray:
        """Predicted relative change of each channel versus the latest sample."""
        if not self._ready:
            return np.zeros(len(CHANNELS))
        h = self._median.shape[0] - 1 if horizon is None else horizon
        return self._median[h] / np.maximum(self._last, 1e-6) - 1.0

    def relative_uncertainty(self, horizon: int | None = None) -> np.ndarray:
        if not self._ready:
            return np.zeros(len(CHANNELS))
        h = self._spread.shape[0] - 1 if horizon is None else horizon
        return self._spread[h] / np.maximum(self._median[h], 1e-6)

    def context(self) -> np.ndarray:
        """Compact feature vector handed to the adaptive-critic agents.

        ``[dQ/Q at 1h, dQ/Q at 2h, dNH/NH at 2h, dCOD/COD at 2h,
          relative flow uncertainty, P(storm) + P(rain)]``
        """
        if not self._ready:
            return np.zeros(6, dtype=float)
        mid = self._median.shape[0] // 2
        r_mid = self.relative_change(mid)
        r_end = self.relative_change(None)
        return np.array([
            np.clip(r_mid[0], -1.0, 2.0),
            np.clip(r_end[0], -1.0, 2.0),
            np.clip(r_end[1], -1.0, 2.0),
            np.clip(r_end[2], -1.0, 2.0),
            np.clip(self.relative_uncertainty(None)[0], 0.0, 1.0),
            float(self._regime[REGIMES.index("rain")]
                  + self._regime[REGIMES.index("storm")]),
        ], dtype=float)
