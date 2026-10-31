"""Tests for the held-out conditions of paper 1: official sensor models and BSM2 influent windows."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("bsm2_python")

from wwtp.official.plant import bsm2_window, load_influent
from wwtp.sacil.sacil import OfficialSensors


def test_sensor_dynamics_follow_a_step_without_noise():
    sens = OfficialSensors(np.array([2.0, 1.0]), noise=False)
    out = [sens(np.array([3.0, 2.0])) for _ in range(40)]       # 30 min
    assert np.allclose(out[-1], [3.0, 2.0], atol=1e-3)
    assert out[0][0] > 2.8 and out[0][1] < 1.05                  # oxygen fast (1 min), nitrate slow (10 min)


def test_sensor_noise_has_the_official_size():
    sens = OfficialSensors(np.array([5.0, 5.0]), seed=1)
    out = np.array([sens(np.array([5.0, 5.0])) for _ in range(4000)])
    assert np.allclose(out.std(0), 0.25, rtol=0.1)


def test_bsm2_window_is_14_days_with_scaled_flow():
    w = bsm2_window(245)
    assert w.shape[1] == 22 and w[0, 0] == 0.0 and abs(w[-1, 0] - 14.0) < 1e-6
    assert np.all(w[:, 16] == 15.0)
    assert np.allclose(load_influent("bsm2:245"), w)
