"""Activated Sludge Model No. 1 (ASM1) biokinetics.

Implements the 13-component / 8-process ASM1 matrix used by the IWA
Benchmark Simulation Model no. 1 (BSM1) at 15 degrees Celsius.

Component ordering (BSM1 convention), index -> symbol [unit]:

    0  S_I    soluble inert organic matter            [g COD/m3]
    1  S_S    readily biodegradable substrate         [g COD/m3]
    2  X_I    particulate inert organic matter        [g COD/m3]
    3  X_S    slowly biodegradable substrate          [g COD/m3]
    4  X_BH   active heterotrophic biomass            [g COD/m3]
    5  X_BA   active autotrophic biomass              [g COD/m3]
    6  X_P    particulate products from decay         [g COD/m3]
    7  S_O    dissolved oxygen                        [g -COD/m3]
    8  S_NO   nitrate and nitrite nitrogen            [g N/m3]
    9  S_NH   ammonium + ammonia nitrogen             [g N/m3]
    10 S_ND   soluble biodegradable organic nitrogen  [g N/m3]
    11 X_ND   particulate biodegradable organic N     [g N/m3]
    12 S_ALK  alkalinity                              [mol HCO3-/m3]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_COMP = 13

(S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK) = range(N_COMP)

COMPONENT_NAMES = (
    "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
    "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK",
)

#: indices of the particulate (settleable) components
PARTICULATE = (X_I, X_S, X_BH, X_BA, X_P, X_ND)
#: indices of the soluble components
SOLUBLE = (S_I, S_S, S_O, S_NO, S_NH, S_ND, S_ALK)

#: COD -> TSS conversion factor used throughout BSM1
TSS_FACTOR = 0.75


@dataclass(frozen=True)
class ASM1Parameters:
    """ASM1 stoichiometric and kinetic parameters (BSM1 defaults, 15 degC)."""

    # stoichiometry
    Y_H: float = 0.67      # heterotrophic yield        [g COD/g COD]
    Y_A: float = 0.24      # autotrophic yield          [g COD/g N]
    f_P: float = 0.08      # inert fraction of biomass  [-]
    i_XB: float = 0.08     # N content of biomass       [g N/g COD]
    i_XP: float = 0.06     # N content of inert product [g N/g COD]

    # kinetics
    mu_H: float = 4.0      # max heterotrophic growth rate   [1/d]
    K_S: float = 10.0      # half-sat, substrate             [g COD/m3]
    K_OH: float = 0.2      # half-sat, oxygen (heterotrophs) [g O2/m3]
    K_NO: float = 0.5      # half-sat, nitrate               [g N/m3]
    b_H: float = 0.3       # heterotrophic decay             [1/d]
    eta_g: float = 0.8     # anoxic growth correction        [-]
    eta_h: float = 0.8     # anoxic hydrolysis correction    [-]
    k_h: float = 3.0       # max specific hydrolysis rate    [g COD/(g COD d)]
    K_X: float = 0.1       # half-sat, hydrolysis            [g COD/g COD]
    mu_A: float = 0.5      # max autotrophic growth rate     [1/d]
    K_NH: float = 1.0      # half-sat, ammonium              [g N/m3]
    b_A: float = 0.05      # autotrophic decay               [1/d]
    K_OA: float = 0.4      # half-sat, oxygen (autotrophs)   [g O2/m3]
    k_a: float = 0.05      # ammonification rate             [m3/(g COD d)]

    # oxygen transfer
    S_O_sat: float = 8.0   # oxygen saturation concentration [g O2/m3]


DEFAULT_PARAMS = ASM1Parameters()


def process_rates(z: np.ndarray, p: ASM1Parameters = DEFAULT_PARAMS) -> np.ndarray:
    """Return the eight ASM1 process rates for each reactor.

    Parameters
    ----------
    z : array of shape (..., 13)
        Component concentrations. The leading axes are broadcast, so a whole
        reactor cascade of shape ``(5, 13)`` can be evaluated at once.
    """
    eps = 1e-10
    S_s = np.maximum(z[..., S_S], 0.0)
    X_s = np.maximum(z[..., X_S], 0.0)
    X_bh = np.maximum(z[..., X_BH], 0.0)
    X_ba = np.maximum(z[..., X_BA], 0.0)
    S_o = np.maximum(z[..., S_O], 0.0)
    S_no = np.maximum(z[..., S_NO], 0.0)
    S_nh = np.maximum(z[..., S_NH], 0.0)
    S_nd = np.maximum(z[..., S_ND], 0.0)
    X_nd = np.maximum(z[..., X_ND], 0.0)

    mon_s = S_s / (p.K_S + S_s + eps)
    mon_oh = S_o / (p.K_OH + S_o + eps)
    inh_oh = p.K_OH / (p.K_OH + S_o + eps)
    mon_no = S_no / (p.K_NO + S_no + eps)

    rho = np.empty(z.shape[:-1] + (8,), dtype=float)
    # 1: aerobic growth of heterotrophs
    rho[..., 0] = p.mu_H * mon_s * mon_oh * X_bh
    # 2: anoxic growth of heterotrophs (denitrification)
    rho[..., 1] = p.mu_H * mon_s * inh_oh * mon_no * p.eta_g * X_bh
    # 3: aerobic growth of autotrophs (nitrification)
    rho[..., 2] = (p.mu_A * S_nh / (p.K_NH + S_nh + eps)
                   * S_o / (p.K_OA + S_o + eps) * X_ba)
    # 4: decay of heterotrophs
    rho[..., 3] = p.b_H * X_bh
    # 5: decay of autotrophs
    rho[..., 4] = p.b_A * X_ba
    # 6: ammonification of soluble organic nitrogen
    rho[..., 5] = p.k_a * S_nd * X_bh
    # 7: hydrolysis of entrapped organics
    ratio = X_s / (X_bh + eps)
    rho[..., 6] = (p.k_h * ratio / (p.K_X + ratio + eps)
                   * (mon_oh + p.eta_h * inh_oh * mon_no) * X_bh)
    # 8: hydrolysis of entrapped organic nitrogen
    rho[..., 7] = rho[..., 6] * X_nd / (X_s + eps)
    return rho


def reaction_rates(z: np.ndarray, p: ASM1Parameters = DEFAULT_PARAMS) -> np.ndarray:
    """Return dC/dt contributions from biological conversion, shape (..., 13)."""
    rho = process_rates(z, p)
    r1, r2, r3, r4, r5, r6, r7, r8 = (rho[..., i] for i in range(8))

    r = np.zeros_like(z)
    r[..., S_I] = 0.0
    r[..., S_S] = -(r1 + r2) / p.Y_H + r7
    r[..., X_I] = 0.0
    r[..., X_S] = (1.0 - p.f_P) * (r4 + r5) - r7
    r[..., X_BH] = r1 + r2 - r4
    r[..., X_BA] = r3 - r5
    r[..., X_P] = p.f_P * (r4 + r5)
    r[..., S_O] = -((1.0 - p.Y_H) / p.Y_H) * r1 - ((4.57 - p.Y_A) / p.Y_A) * r3
    r[..., S_NO] = -((1.0 - p.Y_H) / (2.86 * p.Y_H)) * r2 + r3 / p.Y_A
    r[..., S_NH] = -p.i_XB * (r1 + r2) - (p.i_XB + 1.0 / p.Y_A) * r3 + r6
    r[..., S_ND] = -r6 + r8
    r[..., X_ND] = (p.i_XB - p.f_P * p.i_XP) * (r4 + r5) - r8
    r[..., S_ALK] = (-p.i_XB / 14.0 * r1
                     + ((1.0 - p.Y_H) / (14.0 * 2.86 * p.Y_H) - p.i_XB / 14.0) * r2
                     - (p.i_XB / 14.0 + 1.0 / (7.0 * p.Y_A)) * r3
                     + r6 / 14.0)
    return r


def tss(z: np.ndarray) -> np.ndarray:
    """Total suspended solids [g SS/m3] from the particulate COD fractions."""
    return TSS_FACTOR * (z[..., X_I] + z[..., X_S] + z[..., X_BH]
                         + z[..., X_BA] + z[..., X_P])


def total_nitrogen(z: np.ndarray, p: ASM1Parameters = DEFAULT_PARAMS) -> np.ndarray:
    """Total nitrogen [g N/m3] as defined for the BSM1 effluent quality index."""
    kjeldahl = (z[..., S_NH] + z[..., S_ND] + z[..., X_ND]
                + p.i_XB * (z[..., X_BH] + z[..., X_BA])
                + p.i_XP * (z[..., X_P] + z[..., X_I]))
    return kjeldahl + z[..., S_NO]


def total_cod(z: np.ndarray) -> np.ndarray:
    """Total chemical oxygen demand [g COD/m3]."""
    return (z[..., S_I] + z[..., S_S] + z[..., X_I] + z[..., X_S]
            + z[..., X_BH] + z[..., X_BA] + z[..., X_P])


def bod5(z: np.ndarray, p: ASM1Parameters = DEFAULT_PARAMS) -> np.ndarray:
    """Five-day biochemical oxygen demand [g/m3], BSM1 definition."""
    return 0.25 * (z[..., S_S] + z[..., X_S]
                   + (1.0 - p.f_P) * (z[..., X_BH] + z[..., X_BA]))
