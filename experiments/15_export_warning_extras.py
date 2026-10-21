"""Calibration curves and one example episode for the early-warning web page.

Run after ``05_warning_benchmark.py``.  Merges ``calibration`` into
``results/warning/metrics.json`` and writes ``results/warning/episode.json``.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.warning import baselines, evaluate as ev
from wwtp.warning.dataset import HORIZONS, STEP_MINUTES, Normaliser, window
from wwtp.warning.train import load_warning, predict

ROOT = Path(__file__).resolve().parents[1]
H60 = HORIZONS.index(10)
H120 = HORIZONS.index(20)


def main() -> None:
    data = pickle.load(open(ROOT / "data/warning_rollouts.pkl", "rb"))
    metrics_path = ROOT / "results/warning/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    norm = Normaliser().fit([r["x"] for r in data["train"]])
    test = window(data["test"], norm)
    net, _, drop = load_warning(ROOT / "artifacts/warning.pt")

    # calibration of the 60-min probability on the test plants
    p_net = predict(net, test.x, drop=drop)[1][:, H60]
    print("fitting gradient boosting for its calibration curve", flush=True)
    gbm = baselines.GBMWarning().fit(window(data["train"], norm, stride=2))
    p_gbm = gbm.predict(test)[:, H60]
    cal = {}
    for name, p in (("Early-warning net", p_net), ("Gradient boosting", p_gbm)):
        conf, freq, count = ev.reliability(p, test.event[:, H60])
        cal[name] = {"conf": conf.tolist(), "freq": freq.tolist(), "count": count.tolist(),
                     "ece": ev.expected_calibration_error(p, test.event[:, H60])}
    metrics["calibration"] = cal
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # example episode: the canonical rollout with the most violation onsets
    th = metrics["operational"]["Early-warning net"]["threshold"]
    best = None
    for roll in data["canonical"]:
        c = window([roll], norm)
        if len(c) == 0:
            continue
        n_ep = len(ev.episodes(c.nh_now))
        if n_ep and (best is None or n_ep > best[0]):
            best = (n_ep, roll, c)
    if best is None:
        print("no canonical episode found")
        return
    _, roll, c = best
    levels, p = predict(net, c.x, drop=drop)
    t = roll["t"][c.step]
    onset = ev.episodes(c.nh_now)[0]
    lo, hi = max(0, onset - 240), min(len(t), onset + 240)        # +-24 h
    sl = slice(lo, hi)
    shift = HORIZONS[H60] * STEP_MINUTES / 1440.0
    r = lambda a, d=3: np.round(np.asarray(a, float), d).tolist()
    ep = {"name": roll["name"].replace("|", ", ") + " operating point",
          "t": r(t[sl], 4), "t_shift": r(t[sl] + shift, 4),
          "nh": r(c.nh_now[sl]), "analyser": r(c.current[sl]),
          "q10": r(levels[sl, H60, 0]), "q50": r(levels[sl, H60, 1]), "q90": r(levels[sl, H60, 2]),
          "p120": r(p[sl, H120]), "threshold": th}
    (ROOT / "results/warning/episode.json").write_text(json.dumps(ep))
    print(f"episode from {roll['name']}, onset at day {t[onset]:.2f}; calibration ECE "
          f"net {cal['Early-warning net']['ece']:.3f}, gbm {cal['Gradient boosting']['ece']:.3f}")


if __name__ == "__main__":
    main()
