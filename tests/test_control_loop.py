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


def test_incremental_policy_is_zero_at_zero_error():
    """Eq. (8): with no tracking error the actor emits no increment, so the
    learned term cannot drift into its clip on its own.  A bias input on the
    action network breaks this (pi(0) != 0), which is why it has none."""
    ctrl = MAACCController(seed=3)
    rng = np.random.default_rng(0)
    for agent in ctrl.agents:            # arbitrary, non-zero trained weights
        agent.actor.beta = rng.normal(0.0, 1.0, agent.actor.beta.shape)
    s = np.array(PAPER_SETPOINTS, dtype=float)
    for k in range(50):
        ctrl.act(k, s, s, {})
    assert np.allclose([a.a for a in ctrl.agents], 0.0, atol=1e-12)


def test_fallback_hands_the_correction_to_the_pid_without_a_bump():
    """When the safety fallback fires, u stays continuous and a returns to 0."""
    from wwtp.maacc.agent import tuned_agent_configs
    ctrl = MAACCController(configs=tuned_agent_configs(), seed=0)
    ctrl.reset()
    s = np.array(PAPER_SETPOINTS, dtype=float)
    ctrl.act(0, s, s, {})
    agent = ctrl.agents[0]
    agent.a = 20.0                       # a learned correction is in place ...
    agent.err_avg = 10.0                 # ... and the error has blown up
    u_before = ctrl.pid.u[0] + agent.a
    u = ctrl.act(1, s, s, {})
    assert agent.n_resets == 1 and agent.a == 0.0
    assert np.allclose(agent.actor.beta, 0.0)
    assert abs(u[0] - u_before) < 1.0    # PID took over the correction


def test_tuned_maacc_beats_pid_on_a_short_run():
    from wwtp.maacc.agent import tuned_agent_configs
    pid = run_closed_loop(PIDController(), _series()).summary(1.0)
    m = run_closed_loop(MAACCController(configs=tuned_agent_configs(), seed=0),
                        _series()).summary(1.0)
    assert m["IAE_SO5"] < pid["IAE_SO5"]
    assert m["IAE_SNO2"] < pid["IAE_SNO2"]
