"""Closed-loop behaviour of the controllers on short BSM1 runs."""

import numpy as np
import pytest

from wwtp.baselines.pid_controller import PIDController
from wwtp.bsm1 import influent
from wwtp.bsm1.plant import PAPER_SETPOINTS
from wwtp.envs import run_closed_loop
from wwtp.maacc.agent import MAACCController
from wwtp.metrics import tracking_metrics

SHORT = dict(days=3.0)


def _series(weather="dry"):
    return influent.canonical_scenario(weather, **SHORT)


def test_pid_prior_tracks_both_set_points():
    res = run_closed_loop(PIDController(), _series())
    summary = res.summary(burn_in_days=1.0)
    assert summary["IAE_SO5"] < 0.05
    assert summary["IAE_SNO2"] < 0.10


def test_control_inputs_respect_bsm1_bounds():
    res = run_closed_loop(PIDController(), _series("storm"))
    assert res.control[:, 0].min() >= 0.0 and res.control[:, 0].max() <= 240.0
    assert res.control[:, 1].min() >= 0.0 and res.control[:, 1].max() <= 92230.0


def test_maacc_starts_from_the_expert_prior():
    """Zero-initialised action-network outer weights => u_0 == PID output."""
    ctrl = MAACCController(seed=0)
    ctrl.reset()
    s = np.array(PAPER_SETPOINTS, dtype=float)
    u = ctrl.act(0, s, s, {})
    assert np.allclose([a.a for a in ctrl.agents], 0.0)
    assert np.allclose(u, [84.0, 55338.0], rtol=1e-6)


def test_maacc_learned_correction_stays_bounded():
    ctrl = MAACCController(seed=0)
    run_closed_loop(ctrl, _series("storm"),
                    sensor_noise_std=np.array([0.05, 0.10]))
    for agent in ctrl.agents:
        assert abs(agent.a) <= agent.cfg.a_max + 1e-9


@pytest.mark.parametrize("weather", ["dry", "storm"])
def test_metrics_are_finite_and_ordered(weather):
    res = run_closed_loop(PIDController(), _series(weather))
    m = tracking_metrics(res.error[:, 0], burn_in=100)
    assert np.isfinite([m.iae, m.ise, m.dev_max]).all()
    assert m.dev_max >= m.iae >= 0.0
    summary = res.summary(burn_in_days=1.0)
    assert summary["EQ"] > 0.0 and summary["OCI"] > 0.0
