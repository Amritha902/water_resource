"""Reference predictors the early-warning network has to beat.

``persistence``  what a plant does today: alarm when the effluent analyser
                 reads above the limit.  As a score, the current reading.
``trend``        the analyser reading extrapolated along its last 30 min.
``gbm``          gradient-boosted trees on hand-engineered features of the
                 *same* signals (current values, lags, 1-h changes, rolling
                 load).  A strong tabular baseline: if the network does not
                 beat this, the deep model is not earning its keep.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .dataset import CHANNELS, HORIZONS, Normaliser, WarningCorpus

A = CHANNELS.index("NH_eff_analyser")


def analyser_history(c: WarningCorpus, norm: Normaliser) -> np.ndarray:
    """Raw analyser readings over the lookback window, (N, T)."""
    return c.x[..., A] * norm.std[A] + norm.mean[A]


def persistence_score(c: WarningCorpus, norm: Normaliser) -> np.ndarray:
    """(N, H): the current reading, identical for every horizon."""
    cur = analyser_history(c, norm)[:, -1]
    return np.repeat(cur[:, None], len(HORIZONS), 1)


def trend_score(c: WarningCorpus, norm: Normaliser, span: int = 5) -> np.ndarray:
    """(N, H): linear extrapolation of the last ``span`` steps (30 min)."""
    hist = analyser_history(c, norm)[:, -span:]
    t = np.arange(span) - (span - 1)
    slope = (hist * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
    # the reading is one analyser cycle old, so extrapolate a little further
    lead = np.asarray(HORIZONS, float) + 2.0
    return hist[:, -1:] + slope[:, None] * lead[None]


def features(c: WarningCorpus) -> np.ndarray:
    """Hand-engineered tabular features from the normalised windows."""
    x = c.x
    q, nh_in = CHANNELS.index("Q_in"), CHANNELS.index("NH_in")
    load = x[..., q] + x[..., nh_in]              # log-load proxy
    f = [x[:, -1, :],                             # current values
         x[:, -1, :] - x[:, -11, :],              # 1-h change
         x[:, -1, :] - x[:, -31, :],              # 3-h change
         x[:, -10:, :].mean(1),                   # 1-h mean
         x[:, [-3, -6, -11, -21], A],             # analyser lags
         np.stack([load[:, -10:].mean(1), load[:, -30:].mean(1),
                   load.mean(1), x[:, -10:, CHANNELS.index("KLa_5")].max(1)], 1)]
    return np.concatenate(f, 1)


class GBMWarning:
    """One boosted classifier per horizon."""

    def __init__(self, seed: int = 0):
        self.models = [HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed) for _ in HORIZONS]

    def fit(self, c: WarningCorpus) -> "GBMWarning":
        f = features(c)
        for j, m in enumerate(self.models):
            m.fit(f, c.event[:, j].astype(int))
        return self

    def predict(self, c: WarningCorpus) -> np.ndarray:
        f = features(c)
        p = np.stack([m.predict_proba(f)[:, 1] for m in self.models], 1)
        return np.maximum.accumulate(p, axis=1)   # same monotonicity as the net
