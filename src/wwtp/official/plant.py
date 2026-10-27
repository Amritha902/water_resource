"""Official BSM1 plant for the exact paper-1 replication.

``bsm2-python`` (FAU-EVT, BSD-3) ports the IWA benchmark code and ships the
official dry and rain influent files.  This wrapper exposes what paper 1
controls: inputs ``[KLa_5, Q_a]``, outputs ``[S_O,5, S_NO,2]``.
"""

from __future__ import annotations

import os

import numpy as np
from bsm2_python.bsm1_base import BSM1Base

import bsm2_python

DATA = os.path.join(os.path.dirname(bsm2_python.__file__), "data")
DT_DAYS = 45.0 / 86400.0
U_NOMINAL = np.array([84.0, 55338.0])

#: the IWA BSM1 influent files (Inf_dry/rain/strm_2006), as spreadsheets in
#: github.com/davidfernand/BSM1_in_Python (GPL-3).  Not redistributed here:
#: downloaded on first use into data/official/.
_SRC = "https://raw.githubusercontent.com/davidfernand/BSM1_in_Python/{rev}/code/Influent%20files/{name}"
_REV = "5dadb209a5a34ecb0e2f83a2e79c060bf3470582"
_NAMES = {"dry": "Inf_dry_2006.xlsx", "rain": "Inf_rain_2006.xlsx", "storm": "Inf_strm_2006.xlsx"}
_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "official")


def _official_table(weather: str) -> np.ndarray:
    """(1345, 15) array: t [d], 13 ASM1 concentrations, Q [m3/d]."""
    cache = os.path.abspath(_CACHE)
    npy = os.path.join(cache, weather + ".npy")
    if os.path.exists(npy):
        return np.load(npy)
    os.makedirs(cache, exist_ok=True)
    xlsx = os.path.join(cache, _NAMES[weather])
    if not os.path.exists(xlsx):
        import urllib.request
        urllib.request.urlretrieve(_SRC.format(rev=_REV, name=_NAMES[weather]), xlsx)
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True, data_only=True).worksheets[0]
    rows = [r[:15] for r in ws.iter_rows(values_only=True)
            if r and isinstance(r[0], (int, float))]
    arr = np.array(rows, dtype=float)
    np.save(npy, arr)
    return arr


def load_influent(weather: str) -> np.ndarray:
    """Official influent as a (1345, 22) array: 14 days at 15 min.

    Columns as ``bsm2-python`` expects: t, 13 ASM1 components, TSS, Q, T and
    five dummy states.  One correction: the spreadsheets keep the inert
    soluble COD ``S_I`` at 30 g/m3 in every row, whereas in the official rain
    data it is diluted by the rain water, ``S_I = 30 Q_dry / Q`` (checked
    against the ``bsm2-python`` rain file to 0.005 g/m3).  The same rule is
    applied to rain and storm; dry is unchanged.
    """
    d = _official_table(weather)
    dry = _official_table("dry")
    out = np.zeros((len(d), 22))
    out[:, 0] = d[:, 0]
    out[:, 1:14] = d[:, 1:14]
    out[:, 1] = 30.0 * dry[:, 14] / d[:, 14]
    out[:, 14] = 0.75 * d[:, [3, 4, 5, 6, 7]].sum(axis=1)
    out[:, 15] = d[:, 14]
    out[:, 16] = 15.0
    return out


class OfficialBSM1:
    def __init__(self, weather: str = "dry", days: float | None = None):
        self.sim = BSM1Base(data_in=load_influent(weather), timestep=DT_DAYS,
                            endtime=days)
        self.n = len(self.sim.simtime)
        self.k = 0

    _STATE = ("y_in1", "y_out1", "y_out2", "y_out3", "y_out4", "y_out5",
              "y_out5_r", "ys_in", "ys_out", "ys_eff")

    def get_state(self) -> dict:
        """Plant state (reactor ODE states, settler layers, recycle streams)."""
        sim = self.sim
        st = {a: np.copy(getattr(sim, a)) for a in self._STATE}
        st["reactors"] = [np.copy(getattr(sim, f"reactor{i}").y0) for i in range(1, 6)]
        st["settler"] = np.copy(sim.settler.ys0)
        return st

    def set_state(self, st: dict) -> None:
        sim = self.sim
        for a in self._STATE:
            setattr(sim, a, np.copy(st[a]))
        for i, y in enumerate(st["reactors"], 1):
            getattr(sim, f"reactor{i}").y0 = np.copy(y)
        sim.settler.ys0 = np.copy(st["settler"])

    def measurement(self) -> np.ndarray:
        return np.array([self.sim.y_out5[7], self.sim.y_out2[8]])

    def step(self, u: np.ndarray) -> np.ndarray:
        """Apply ``[KLa_5, Q_a]`` for one 45 s period; return ``[S_O,5, S_NO,2]``."""
        self.sim.klas[4] = float(u[0])
        self.sim.qintr = float(u[1])
        self.sim.step(self.k)
        self.k += 1
        return self.measurement()


# -- adapter for the aeration environment of paper 2 ------------------------

from bsm2_python.bsm2.init import asm1init_bsm1 as _asm1init  # noqa: E402

from ..bsm1 import asm1 as _asm1  # noqa: E402
from ..bsm1.influent import InfluentSeries  # noqa: E402

_VOL = np.array([1000.0, 1000.0, 1333.0, 1333.0, 1333.0])


def official_series(weather: str, dt_seconds: float = 45.0) -> InfluentSeries:
    """Official influent resampled (zero-order hold) to the control period."""
    d = load_influent(weather)
    n = int(round(d[-1, 0] * 86400.0 / dt_seconds))
    t = np.arange(n) * dt_seconds / 86400.0
    idx = np.minimum(np.searchsorted(d[:, 0], t, side="right") - 1, len(d) - 1)
    return InfluentSeries(time_days=t, flow=d[idx, 15].copy(),
                          composition=d[idx, 1:14].copy(), weather=weather)


class _Params:
    q_w = float(_asm1init.QW)


class OfficialPlantAdapter:
    """The official BSM1 behind the interface :class:`~wwtp.rl.env.AerationEnv`
    uses (``step``, ``z``, ``z_effluent``, energy rates).  The influent row
    is the one the environment passes in, so time indexing stays the
    environment's."""

    p = _Params()

    def __init__(self, state: dict | None = None):
        self.plant = OfficialBSM1("dry")
        if state is not None:
            self.plant.set_state(state)
        self.k = 0

    def step(self, dt_seconds: float, composition: np.ndarray, flow: float,
             u: np.ndarray) -> None:
        sim = self.plant.sim
        sim.klas[4] = float(u[0])
        sim.qintr = float(u[1])
        y = np.zeros(21)
        y[:13] = composition
        y[13] = 0.75 * (composition[2] + composition[3] + composition[4]
                        + composition[5] + composition[6])
        y[14], y[15] = flow, 15.0
        t, h = self.k * DT_DAYS, DT_DAYS
        # the plant part of BSM1Base.step, with this influent row
        for i, r in enumerate((sim.reactor1, sim.reactor2, sim.reactor3,
                               sim.reactor4, sim.reactor5)):
            r.kla = sim.klas[i]
        sim.y_in1 = sim.combiner.output(y, sim.ys_out, sim.y_out5_r)
        sim.y_out1 = sim.reactor1.output(h, t, sim.y_in1)
        sim.y_out2 = sim.reactor2.output(h, t, sim.y_out1)
        sim.y_out3 = sim.reactor3.output(h, t, sim.y_out2)
        sim.y_out4 = sim.reactor4.output(h, t, sim.y_out3)
        sim.y_out5 = sim.reactor5.output(h, t, sim.y_out4)
        sim.ys_in, sim.y_out5_r = sim.splitter.output(
            sim.y_out5, (max(sim.y_out5[14] - sim.qintr, 0.0), float(sim.qintr)))
        sim.ys_out, _, sim.ys_eff, sim.sludge_height, sim.ys_tss_internal = \
            sim.settler.output(h, t, sim.ys_in)
        self.k += 1

    @property
    def z(self) -> np.ndarray:
        s = self.plant.sim
        return np.array([s.y_out1[:13], s.y_out2[:13], s.y_out3[:13],
                         s.y_out4[:13], s.y_out5[:13]])

    @property
    def z_effluent(self) -> np.ndarray:
        return np.array(self.plant.sim.ys_eff[:13])

    def aeration_energy_rate(self, u: np.ndarray) -> float:
        kla = np.array([0.0, 0.0, 240.0, 240.0, float(u[0])])
        return float(8.0 / 1800.0 * np.sum(_VOL * kla))

    def pumping_energy_rate(self, q_a: float, _q_other: float = 0.0) -> float:
        return float(0.004 * q_a + 0.008 * _asm1init.QR + 0.05 * _asm1init.QW)


def paper2_warm_state() -> dict:
    """14 days of official dry weather under paper 2's PID (DO = 2 mg/L via
    KLa_5, ``Q_a`` at the BSM1 default), from the benchmark steady state."""
    from ..rl.baselines import PAPER2_KD, PAPER2_KI, PAPER2_KP
    p = OfficialBSM1("dry")
    kla, e1, e2 = 84.0, 0.0, 0.0
    for _ in range(p.n - 1):
        e = 2.0 - p.sim.y_out5[7]
        du = PAPER2_KP * (e - e1) + PAPER2_KI * e + PAPER2_KD * (e - 2 * e1 + e2)
        e2, e1 = e1, e
        kla = float(np.clip(kla + np.clip(du, -6.0, 6.0), 0.0, 240.0))
        p.step(np.array([kla, 55338.0]))
    return p.get_state()
