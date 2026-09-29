"""Evaluation of violation early warnings.

Two views, both needed:

*Sample view* -- AUROC, AUPRC, Brier score and calibration per horizon.
Reported on all samples and on the **onset subset**: samples at which the
plant is compliant right now (true ammonium and analyser reading both below
the limit).  On ongoing violations persistence is trivially right, so the
onset subset is where prediction actually has to happen.

*Operational view* -- what an operator experiences.  An alarm is raised
while a score exceeds a threshold.  For every violation episode (exceedance
after at least one compliant hour) we measure the **lead time**: how long
before the onset the alarm had been continuously on (negative = the alarm
only came after the plant was already violating).  Thresholds are chosen on
the validation rollouts for a fixed budget of false alarms per compliant
day, so methods are compared at equal nuisance to the operator.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import HORIZONS, NH_LIMIT, STEP_MINUTES, WarningCorpus


def onset_mask(c: WarningCorpus) -> np.ndarray:
    return (c.current <= NH_LIMIT) & (c.nh_now <= NH_LIMIT)


def sample_metrics(score: np.ndarray, c: WarningCorpus,
                   probabilistic: bool) -> dict:
    out = {}
    for subset, m in (("all", np.ones(len(c), bool)), ("onset", onset_mask(c))):
        for j, h in enumerate(HORIZONS):
            y, s = c.event[m, j], score[m, j]
            key = f"{subset}@{int(h * STEP_MINUTES)}min"
            if y.min() == y.max():
                continue
            row = {"auroc": float(roc_auc_score(y, s)),
                   "auprc": float(average_precision_score(y, s)),
                   "prevalence": float(y.mean())}
            if probabilistic:
                row["brier"] = float(np.mean((s - y) ** 2))
                row["ece"] = expected_calibration_error(s, y)
            out[key] = row
    return out


def expected_calibration_error(p: np.ndarray, y: np.ndarray,
                               bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, freq, count = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            conf.append(p[m].mean()); freq.append(y[m].mean()); count.append(m.sum())
    return np.array(conf), np.array(freq), np.array(count)


# -- operational view ----------------------------------------------------------
def _runs(alarm: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of every run of True."""
    a = np.concatenate([[False], alarm, [False]]).astype(int)
    d = np.diff(a)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def episodes(nh: np.ndarray, quiet: int = 10) -> list[int]:
    """Violation onsets preceded by at least ``quiet`` compliant steps."""
    exceed = nh > NH_LIMIT
    return [i for i in range(quiet, len(nh))
            if exceed[i] and not exceed[i - quiet:i].any()]


def operational(score_seq: list[np.ndarray], nh_seq: list[np.ndarray],
                threshold: float, window: int = max(HORIZONS)) -> dict:
    """Lead times and false alarms over whole rollouts.

    ``score_seq[r]`` is the alarm score at every step of rollout ``r``.
    """
    leads, false_alarms, compliant_days = [], 0, 0.0
    for s, nh in zip(score_seq, nh_seq):
        alarm = s > threshold
        exceed = nh > NH_LIMIT
        compliant_days += (~exceed).sum() * STEP_MINUTES / 1440.0
        for i in episodes(nh):
            if alarm[i - 1]:
                j = i - 1
                while j > 0 and alarm[j - 1] and not exceed[j - 1]:
                    j -= 1
                leads.append((i - j) * STEP_MINUTES)
            else:
                later = np.flatnonzero(alarm[i:])
                leads.append(-later[0] * STEP_MINUTES if len(later)
                             else -np.inf)
        for a, b in _runs(alarm & ~exceed):
            # a run is a false alarm if no exceedance follows within the window
            if not exceed[a:min(len(nh), b + window)].any():
                false_alarms += 1
    leads = np.array(leads)
    finite = leads[np.isfinite(leads)]
    return {"n_episodes": int(len(leads)),
            "median_lead_min": float(np.median(finite)) if len(finite) else np.nan,
            "frac_warned_30min": float(np.mean(leads >= 30)) if len(leads) else np.nan,
            "frac_warned_before": float(np.mean(leads > 0)) if len(leads) else np.nan,
            "missed": int(np.sum(~np.isfinite(leads))),
            "false_alarms_per_day": false_alarms / max(compliant_days, 1e-9),
            "threshold": float(threshold)}


def threshold_for_budget(score_seq, nh_seq, budget: float,
                         candidates: np.ndarray) -> float:
    """Smallest threshold whose false-alarm rate stays within ``budget``/day."""
    for th in np.sort(candidates):
        if operational(score_seq, nh_seq, th)["false_alarms_per_day"] <= budget:
            return float(th)
    return float(np.max(candidates))
