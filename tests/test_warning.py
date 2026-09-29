"""Checks on the effluent-ammonium early warning.

The label and metric tests guard against the two classic ways an early-warning
result is inflated: labels that peek at the present (so "prediction" is really
detection), and lead times that count alarms raised after the onset.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wwtp.warning import evaluate as ev
from wwtp.warning.dataset import (CHANNELS, HORIZONS, LOOKBACK, NH_LIMIT,
                                  STEP_MINUTES, Normaliser, targets, window)
from wwtp.warning.models import EarlyWarningNet

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_targets_look_strictly_ahead():
    nh = np.zeros(100)
    nh[50] = 10.0
    level, event = targets(nh)
    h0 = HORIZONS[0]
    # at the violating step itself, the label only asks about the future
    assert event[50, 0] == 0.0
    # one step before, every horizon sees it
    assert np.all(event[49] == 1.0)
    # just outside the shortest horizon, only longer horizons see it
    assert event[50 - h0 - 1, 0] == 0.0 and event[50 - h0 - 1, -1] == 1.0
    assert level[50 - h0, 0] == 10.0
    # no labels where the longest horizon runs off the end
    assert np.isnan(event[-max(HORIZONS):]).all()


def test_hazard_head_is_monotone_in_horizon():
    model = EarlyWarningNet()
    levels, p = model(torch.randn(16, LOOKBACK, len(CHANNELS)))
    assert levels.shape[1:] == (len(HORIZONS), 3)
    assert bool((p[:, 1:] >= p[:, :-1]).all())
    assert bool((levels[..., 2] >= levels[..., 1]).all())
    assert bool((levels[..., 1] >= levels[..., 0]).all())


def test_windows_are_causal():
    """The last row of each window is the prediction step, never later."""
    n = 400
    x = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, len(CHANNELS))) + 10
    roll = {"x": x, "nh": np.zeros(n, np.float32)}
    norm = Normaliser(np.zeros(len(CHANNELS), np.float32),
                      np.ones(len(CHANNELS), np.float32))
    c = window([roll], norm, burn_in_steps=0)
    raw_last = c.x[:, -1, CHANNELS.index("DO_5")] - 10
    assert np.array_equal(raw_last, c.step)


def test_lead_time_accounting():
    nh = np.zeros(200)
    nh[100:120] = 6.0                      # one episode, onset at step 100
    early = np.zeros(200); early[90:120] = 1.0
    late = np.zeros(200); late[103:120] = 1.0
    th = 0.5
    o = ev.operational([early], [nh], th)
    assert o["n_episodes"] == 1
    assert o["median_lead_min"] == pytest.approx(10 * STEP_MINUTES)
    assert o["false_alarms_per_day"] == 0.0
    o = ev.operational([late], [nh], th)
    assert o["median_lead_min"] == pytest.approx(-3 * STEP_MINUTES)
    assert o["frac_warned_before"] == 0.0


def test_isolated_alarm_counts_as_false():
    nh = np.zeros(300)
    s = np.zeros(300); s[50:55] = 1.0
    o = ev.operational([s], [nh], 0.5)
    assert o["n_episodes"] == 0
    assert o["false_alarms_per_day"] > 0


@pytest.mark.skipif(not (ARTIFACTS / "warning.pt").exists(),
                    reason="early-warning network not trained yet")
def test_trained_warning_responds_to_the_analyser():
    """Raising the recent analyser readings must raise the violation risk."""
    from wwtp.warning.train import load_warning
    model, norm, drop = load_warning(ARTIFACTS / "warning.pt")
    base = np.tile(norm.mean, (LOOKBACK, 1)).astype(np.float32)
    raw = base.copy()
    for c in norm.LOG:
        raw[:, c] = np.exp(raw[:, c])
    lo, hi = raw.copy(), raw.copy()
    a = CHANNELS.index("NH_eff_analyser")
    lo[:, a] = 1.0
    hi[:, a] = np.linspace(1.0, NH_LIMIT - 0.3, LOOKBACK)   # rising, still compliant
    with torch.no_grad():
        p_lo = model(torch.from_numpy(norm(lo)[None]))[1][0]
        p_hi = model(torch.from_numpy(norm(hi)[None]))[1][0]
    assert float(p_hi[-1]) > float(p_lo[-1])
