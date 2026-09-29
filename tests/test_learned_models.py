"""Checks on the learned components.

The tests that need trained weights are skipped when ``artifacts/`` has not
been built yet, so the suite still runs on a fresh clone.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wwtp.forecast.dataset import HORIZON, LOOKBACK, build_corpus
from wwtp.forecast.models import InfluentForecaster, quantile_loss
from wwtp.twin.dataset import D_SCALE, S_SCALE, U_SCALE
from wwtp.twin.models import DigitalTwin

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FORECASTER = ARTIFACTS / "forecaster.pt"
TWIN = ARTIFACTS / "twin.pt"


def test_quantile_head_is_monotone_by_construction():
    model = InfluentForecaster()
    q, logits, ctx = model(torch.randn(8, LOOKBACK, 3))
    assert q.shape[1:3] == (HORIZON, 3)
    assert bool((q[..., 1] >= q[..., 0]).all())
    assert bool((q[..., 2] >= q[..., 1]).all())
    assert logits.shape[1] == 3


def test_pinball_loss_is_minimised_at_the_true_quantile():
    target = torch.zeros(1, 1, 1)
    worse = quantile_loss(torch.full((1, 1, 1, 3), 0.5), target)
    better = quantile_loss(torch.zeros(1, 1, 1, 3), target)
    assert float(better) < float(worse)


def test_corpus_windows_are_causal_and_normalised():
    corpus = build_corpus(n_scenarios=3, seed=1)
    assert corpus.x.shape[1:] == (LOOKBACK, 3)
    assert corpus.y.shape[1:] == (HORIZON, 3)
    assert abs(float(corpus.x.mean())) < 0.5
    assert set(np.unique(corpus.regime)).issubset({0, 1, 2})


def test_twin_rollout_is_differentiable_in_the_control():
    twin = DigitalTwin()
    s0 = torch.tensor([[0.5, 0.6]])
    u = torch.rand(1, 6, 2, requires_grad=True)
    d = torch.rand(1, 6, 3)
    states, nh, _ = twin.rollout(s0, u, d, twin.init_hidden(1))
    states.sum().backward()
    assert u.grad is not None and float(u.grad.abs().sum()) > 0.0
    assert states.shape == (1, 6, 2) and nh.shape == (1, 6, 1)


@pytest.mark.skipif(not TWIN.exists(), reason="run experiments/02_train_twin.py")
def test_trained_twin_has_the_right_control_gains():
    """Closed-loop identification bias would flip or flatten these gains."""
    blob = torch.load(TWIN, map_location="cpu", weights_only=False)
    twin = DigitalTwin()
    twin.load_state_dict(blob["state_dict"])
    twin.eval()

    s0 = torch.tensor([[1.0, 2.0]], dtype=torch.float32) / torch.tensor(
        S_SCALE, dtype=torch.float32)
    d = torch.tensor([[18446.0, 31.6, 69.5]], dtype=torch.float32) / torch.tensor(
        D_SCALE, dtype=torch.float32)

    def settle(kla, qa, steps=10):
        u = torch.tensor([[kla, qa]], dtype=torch.float32) / torch.tensor(
            U_SCALE, dtype=torch.float32)
        s, h = s0, twin.init_hidden(1)
        with torch.no_grad():
            for _ in range(steps):
                s, _, h = twin.step(s, u, d, h)
        return (s[0] * torch.tensor(S_SCALE, dtype=torch.float32)).numpy()

    low_air, high_air = settle(40.0, 55338.0), settle(200.0, 55338.0)
    assert high_air[0] - low_air[0] > 0.5, "aeration gain too small or inverted"

    low_rec, high_rec = settle(84.0, 15000.0), settle(84.0, 85000.0)
    assert high_rec[1] > low_rec[1], "internal-recycle gain has the wrong sign"


@pytest.mark.skipif(not FORECASTER.exists(),
                    reason="run experiments/01_train_forecaster.py")
def test_predictor_produces_a_bounded_context_vector():
    from wwtp.bsm1 import influent
    from wwtp.forecast.dataset import observable
    from wwtp.forecast.predictor import InfluentPredictor

    predictor = InfluentPredictor(FORECASTER)
    series = influent.canonical_scenario("storm")
    obs = observable(series)
    assert not predictor.ready
    for k in range(0, 20 * LOOKBACK * 20, 20):
        predictor.update(k, obs[(k // 20) % len(obs)])
    assert predictor.ready
    ctx = predictor.context()
    assert ctx.shape == (6,) and np.all(np.isfinite(ctx))
    assert np.all(predictor.spread >= 0.0)
    assert abs(float(predictor.regime.sum()) - 1.0) < 1e-4
