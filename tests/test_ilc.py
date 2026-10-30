"""Tests for the daily ILC and its bumpless transfer in the paper 1 loop."""

import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("bsm2_python")

from wwtp.sacil.ilc import PERIOD, DailyILC
from wwtp.sacil.sacil import Paper1Env

WARM = Path(__file__).resolve().parents[1] / "demo" / "cache" / "warm_paper1.pkl"


def test_ilc_learns_the_daily_deviation_ahead_of_time():
    ilc = DailyILC(beta=[1.0], lead=[2], smooth=0)
    u = np.sin(2 * np.pi * np.arange(PERIOD) / PERIOD)[:, None]
    for k in range(PERIOD):
        assert np.all(ilc.ff(k) == 0.0)
        ilc.record(k, np.zeros(1), u[k])
    assert np.allclose(ilc.ff(PERIOD), u[2] - u.mean())


def midnight_jump(bumpless):
    st, u0 = pickle.loads(WARM.read_bytes())
    ilc = DailyILC(beta=[0.75, 0.0], lead=[1, 0], smooth=5, bumpless=bumpless)
    env = Paper1Env("dry", st, u0, days=(PERIOD + 3) * 45 / 86400, base_gain=(3, 3), act_mask=(0, 0), ilc=ilc)
    env.reset()
    us, done = [], False
    while not done:
        _, _, done, info = env.step(np.zeros(2))
        us.append(info["u"][0])
    steps = np.abs(np.diff(us))
    return steps[PERIOD - 1], np.median(steps)


@pytest.mark.skipif(not WARM.exists(), reason="needs the cached warm state")
def test_bumpless_transfer_removes_the_midnight_jump():
    jump, typical = midnight_jump(bumpless=False)
    assert jump > 20 * typical
    jump, typical = midnight_jump(bumpless=True)
    assert jump < 5 * typical
