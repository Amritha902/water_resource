"""Control-performance and plant-performance indicators.

``IAE``, ``ISE`` and ``DEVmax`` follow the definitions given in the reference
paper (eq. 36).  ``EQ`` (effluent quality index) and ``OCI`` (overall cost
index) follow the BSM1 evaluation protocol and are used to show that the
prediction-driven controller does not buy accuracy with energy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .bsm1 import asm1

#: BSM1 effluent quality weighting factors [g pollution unit / g]
EQ_WEIGHTS = {"TSS": 2.0, "COD": 1.0, "BOD5": 2.0, "TKN": 30.0, "NO": 10.0}

#: discharge limits used for the violation statistics
EFFLUENT_LIMITS = {"TN": 18.0, "NH": 4.0, "COD": 100.0, "BOD5": 10.0, "TSS": 30.0}


@dataclass
class TrackingMetrics:
    iae: float
    ise: float
    dev_max: float

    def as_dict(self) -> dict:
        return asdict(self)


def tracking_metrics(error: np.ndarray, burn_in: int = 0) -> TrackingMetrics:
    """IAE / ISE / DEVmax of a scalar tracking-error trajectory.

    ``IAE`` and ``ISE`` are sample means (as in eq. 36 of the paper), so they
    are independent of the sampling interval and directly comparable with the
    published numbers.
    """
    e = np.asarray(error, dtype=float)[burn_in:]
    return TrackingMetrics(iae=float(np.mean(np.abs(e))),
                           ise=float(np.mean(e ** 2)),
                           dev_max=float(np.max(np.abs(e))))


def effluent_quality_index(z_eff: np.ndarray, q_eff: np.ndarray,
                           dt_days: float) -> float:
    """BSM1 effluent quality index [kg poll. units/d]."""
    tss = asm1.tss(z_eff)
    cod = asm1.total_cod(z_eff)
    bod = asm1.bod5(z_eff)
    tn = asm1.total_nitrogen(z_eff)
    tkn = tn - z_eff[:, asm1.S_NO]
    load = (EQ_WEIGHTS["TSS"] * tss + EQ_WEIGHTS["COD"] * cod
            + EQ_WEIGHTS["BOD5"] * bod + EQ_WEIGHTS["TKN"] * tkn
            + EQ_WEIGHTS["NO"] * z_eff[:, asm1.S_NO]) * q_eff
    total_days = dt_days * len(q_eff)
    return float(np.sum(load) * dt_days / (1000.0 * total_days))


def violation_fraction(z_eff: np.ndarray, key: str = "NH") -> float:
    """Fraction of samples exceeding a discharge limit."""
    if key == "NH":
        value = z_eff[:, asm1.S_NH]
    elif key == "TN":
        value = asm1.total_nitrogen(z_eff)
    elif key == "COD":
        value = asm1.total_cod(z_eff)
    elif key == "BOD5":
        value = asm1.bod5(z_eff)
    elif key == "TSS":
        value = asm1.tss(z_eff)
    else:
        raise ValueError(f"unknown effluent key {key!r}")
    return float(np.mean(value > EFFLUENT_LIMITS[key]))


def energy_indices(aeration_kwh_d: np.ndarray, pumping_kwh_d: np.ndarray,
                   dt_days: float) -> dict:
    """Average aeration / pumping energy and the BSM1 overall cost index."""
    ae = float(np.mean(aeration_kwh_d))
    pe = float(np.mean(pumping_kwh_d))
    return {"AE": ae, "PE": pe, "OCI": ae + pe}
