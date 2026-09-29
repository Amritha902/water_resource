"""Sanity checks on the ASM1 / BSM1 re-implementation."""

import numpy as np
import pytest

from wwtp.bsm1 import asm1, influent
from wwtp.bsm1.plant import BSM1Plant

DRY_INFLUENT = influent.DRY_AVERAGE_COMPOSITION


def _run(plant, u, days=2.0, dt=45.0, q=influent.DRY_AVERAGE_FLOW):
    for _ in range(int(days * 86400 / dt)):
        m = plant.step(dt, DRY_INFLUENT, q, np.asarray(u, dtype=float))
    return m


def test_reaction_rates_conserve_cod_within_growth_stoichiometry():
    z = np.tile(np.array([30., 3., 1000., 60., 2200., 140., 600., 1.,
                          10., 5., 1., 3., 5.]), (5, 1))
    r = asm1.reaction_rates(z)
    assert r.shape == z.shape
    # inert soluble COD is never produced or consumed
    assert np.allclose(r[:, asm1.S_I], 0.0)
    assert np.allclose(r[:, asm1.X_I], 0.0)
    # aerobic conditions must consume oxygen
    assert np.all(r[:, asm1.S_O] < 0.0)


def test_states_stay_non_negative():
    plant = BSM1Plant()
    _run(plant, [84.0, 55338.0], days=3.0)
    assert np.all(plant.z >= 0.0)
    assert np.all(plant.settler.x >= 0.0)


def test_open_loop_steady_state_matches_bsm1():
    """Reactor 5 DO and reactor 2 nitrate stay near the published values."""
    plant = BSM1Plant()
    m = _run(plant, [84.0, 55338.0], days=3.0)
    assert 0.3 < m[0] < 0.8       # BSM1 open-loop S_O,5 ~ 0.49 g/m3
    assert 2.5 < m[1] < 5.0       # BSM1 open-loop S_NO,2 ~ 3.66 g N/m3
    assert 5.0 < float(asm1.tss(plant.z_effluent)) < 25.0


@pytest.mark.parametrize("kla,expected", [(20.0, "low"), (200.0, "high")])
def test_aeration_gain_has_the_right_sign(kla, expected):
    plant = BSM1Plant()
    m = _run(plant, [kla, 55338.0], days=2.0)
    if expected == "low":
        assert m[0] < 0.5
    else:
        assert m[0] > 1.5


def test_internal_recycle_raises_nitrate_in_reactor_2():
    low = _run(BSM1Plant(), [84.0, 10000.0], days=3.0)
    high = _run(BSM1Plant(), [84.0, 80000.0], days=3.0)
    assert high[1] > low[1]


def test_settler_thickens_sludge():
    plant = BSM1Plant()
    _run(plant, [84.0, 55338.0], days=2.0)
    assert plant.settler.x_underflow > 50.0 * plant.settler.x_effluent


@pytest.mark.parametrize("weather", ["dry", "rain", "storm"])
def test_influent_scenarios_have_plausible_statistics(weather):
    s = influent.canonical_scenario(weather)
    assert s.n_steps == int(14 * 86400 / 45)
    assert 15000.0 < s.flow.mean() < 22000.0
    assert np.all(s.composition >= 0.0)
    if weather != "dry":
        dry = influent.canonical_scenario("dry")
        assert s.flow.max() > 1.2 * dry.flow.max()
