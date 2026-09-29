"""Ten-layer Takacs secondary clarifier as specified by BSM1.

Layer 1 is the top (effluent) layer and layer 10 the bottom (underflow)
layer; the reactor cascade feeds layer 6.  Solids are transported by the
double-exponential settling velocity function of Takacs et al. (1991);
soluble components are treated as non-reactive and simply follow the flow,
which is the standard BSM1 simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import asm1


@dataclass(frozen=True)
class SettlerParameters:
    n_layers: int = 10
    feed_layer: int = 6        # 1-based, counted from the top
    area: float = 1500.0       # [m2]
    height: float = 4.0        # [m]
    v0_max: float = 250.0      # max practical settling velocity [m/d]
    v0: float = 474.0          # max theoretical settling velocity [m/d]
    r_h: float = 5.76e-4       # hindered zone parameter [m3/g SS]
    r_p: float = 2.86e-3       # flocculant zone parameter [m3/g SS]
    f_ns: float = 2.28e-3      # non-settleable fraction [-]


DEFAULT_SETTLER = SettlerParameters()


def settling_velocity(x: np.ndarray, x_feed: float,
                      p: SettlerParameters = DEFAULT_SETTLER) -> np.ndarray:
    """Takacs double-exponential settling velocity [m/d]."""
    x_min = p.f_ns * x_feed
    xj = np.maximum(x - x_min, 0.0)
    v = p.v0 * (np.exp(-p.r_h * xj) - np.exp(-p.r_p * xj))
    return np.clip(v, 0.0, p.v0_max)


class Settler:
    """Stateful ten-layer clarifier.

    The state is the TSS concentration in each layer, ``x`` of shape
    ``(n_layers,)`` in g SS/m3, ordered top -> bottom.
    """

    def __init__(self, p: SettlerParameters = DEFAULT_SETTLER):
        self.p = p
        self.layer_height = p.height / p.n_layers
        self.x = np.full(p.n_layers, 100.0)

    def derivative(self, x: np.ndarray, x_feed: float, q_feed: float,
                   q_under: float) -> np.ndarray:
        """dTSS/dt for every layer [g SS/(m3 d)]."""
        p = self.p
        n = p.n_layers
        fl = p.feed_layer - 1                       # 0-based feed layer
        dz = self.layer_height
        q_over = q_feed - q_under

        v_up = q_over / p.area                      # upward bulk velocity [m/d]
        v_dn = q_under / p.area                     # downward bulk velocity [m/d]

        xc = np.maximum(x, 0.0)
        vs = settling_velocity(xc, x_feed, p)
        j_s = vs * xc                               # gravity flux leaving each layer downward

        # Gravity flux across each interface j|j+1 (n-1 interfaces).
        # Takacs: below the feed layer the flux is additionally limited by the
        # settling capacity of the receiving layer.
        j = np.zeros(n + 1)                         # j[k] = flux from layer k-1 into layer k
        for k in range(n - 1):
            if k < fl:
                j[k + 1] = j_s[k]
            else:
                j[k + 1] = min(j_s[k], j_s[k + 1])
        j[0] = 0.0                                  # no gravity flux through the surface
        j[n] = 0.0                                  # bottom handled by the underflow term

        dx = np.zeros(n)
        for k in range(n):
            adv = 0.0
            if k < fl:                              # clarification zone: bulk flow upward
                adv = v_up * (xc[k + 1] - xc[k]) / dz if k + 1 < n else 0.0
            elif k > fl:                            # thickening zone: bulk flow downward
                adv = v_dn * (xc[k - 1] - xc[k]) / dz
            else:                                   # feed layer
                adv = (q_feed * x_feed / p.area
                       - v_up * xc[k] - v_dn * xc[k]) / dz
            dx[k] = adv + (j[k] - j[k + 1]) / dz
        return dx

    def step(self, dt: float, x_feed: float, q_feed: float, q_under: float,
             substeps: int = 1) -> None:
        h = dt / substeps
        for _ in range(substeps):
            self.x = np.maximum(self.x + h * self.derivative(
                self.x, x_feed, q_feed, q_under), 0.0)

    # -- interface ---------------------------------------------------------
    @property
    def x_effluent(self) -> float:
        return float(self.x[0])

    @property
    def x_underflow(self) -> float:
        return float(self.x[-1])

    def split(self, z_feed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split the settler feed into (effluent, underflow) ASM1 vectors.

        Soluble components pass through unchanged; particulate components are
        scaled by the ratio of layer TSS to feed TSS so that the particulate
        composition of the feed is preserved.
        """
        x_feed = float(asm1.tss(z_feed))
        ratio_e = self.x_effluent / x_feed if x_feed > 1e-9 else 0.0
        ratio_u = self.x_underflow / x_feed if x_feed > 1e-9 else 0.0

        z_e = z_feed.copy()
        z_u = z_feed.copy()
        for idx in asm1.PARTICULATE:
            z_e[idx] = z_feed[idx] * ratio_e
            z_u[idx] = z_feed[idx] * ratio_u
        return z_e, z_u
