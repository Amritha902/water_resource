"""Reward functions of Du et al. (2023), Section 3.5.

``r_EQ`` (eq. 19) targets effluent quality only; ``r_AE_EQ`` (eq. 23) adds
aeration energy and is the paper's energy-saving variant.  Coefficients are
the published ones: ``a1 = 1, a2 = 0.38`` and ``b1 = 1, b2 = 0.42``.

One scaling decision has to be made.  Equation (23) writes ``b1 * AE`` with
``AE`` in kWh/d, which is of order 3500 and would swamp the ammonium term
entirely at ``b2 = 0.42``; the paper does not state the units it feeds the
reward.  ``AE`` is affine in ``KLa_5`` (the other four transfer coefficients
are constant in BSM1), so we use the normalised aeration level
``KLa_5 / 240`` in [0, 1].  That is the same quantity up to an affine
transform and makes the published coefficients meaningful: a full-aeration
penalty of 1.0 against 0.42 per mg/L of ammonium over the limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .env import NH_LIMIT, TN_LIMIT

#: published coefficients of eq. (19)
ALPHA_1, ALPHA_2 = 1.0, 0.38
#: published coefficients of eq. (23)
BETA_1, BETA_2 = 1.0, 0.42


def _excess(value: float, limit: float) -> float:
    return max(value - limit, 0.0)


def reward_eq(info: dict, a1: float = ALPHA_1, a2: float = ALPHA_2) -> float:
    """Equation (19): effluent-quality reward (the paper's DDPG-A)."""
    return -(a1 * _excess(info["S_NH"], NH_LIMIT)
             + a2 * _excess(info["N_tot"], TN_LIMIT))


def reward_ae_eq(info: dict, b1: float = BETA_1, b2: float = BETA_2) -> float:
    """Equation (23): energy + ammonium reward (the paper's DDPG-B)."""
    return -(b1 * info["aeration_fraction"]
             + b2 * _excess(info["S_NH"], NH_LIMIT))


@dataclass
class RewardSpec:
    name: str
    fn: callable

    def __call__(self, info: dict) -> float:
        return self.fn(info)


REWARDS = {
    "EQ": RewardSpec("EQ", reward_eq),            # DDPG-A
    "AE_EQ": RewardSpec("AE_EQ", reward_ae_eq),   # DDPG-B
}
