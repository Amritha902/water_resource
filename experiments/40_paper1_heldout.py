"""Paper 1: the held-out test of docs/PAPER1_HELDOUT_PROTOCOL.md.

Every controller frozen as in experiments/33_paper1_study.py (nothing re-tuned
or retrained), run on

  H1  six 14-day windows of the official BSM2 influent (ideal sensors)
  H2  the official dry, rain and storm files with the official BSM1 sensors
      (delay and noise), three noise seeds
  H3  as H2 with the sensor delay only

scored on days 7-14 with the true concentrations.  Writes
results/summary_paper1_heldout.json (and traces of the seed-0 runs).

    python experiments/40_paper1_heldout.py
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s33", ROOT / "experiments" / "33_paper1_study.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

WINDOWS = (245, 301, 357, 413, 469, 525)
WEATHERS = ("dry", "rain", "storm")
NOISE_SEEDS = (0, 1, 2)
SINGLE = ("PID", "MAACC", "PIDx3", "PIDx3+ILC", "PIDtuned", "PIDtuned+ILC")
LEARNED = ("rMAACC", "PIDx3+ILC+rMAACC", "Ours", "OursT")      # OursT: protocol amendment 1
SEEDS = (0, 1, 2, 3, 4)


def controllers():
    return [(c, 0) for c in SINGLE] + [(c, s) for c in LEARNED for s in SEEDS]


def jobs():
    J = []
    for w in WINDOWS:
        J += [("H1", f"bsm2:{w}", None, c, s) for c, s in controllers()]
    for w in WEATHERS:
        for n in NOISE_SEEDS:
            J += [("H2", w, n, c, s) for c, s in controllers()]
        J += [("H3", w, "lag", c, s) for c, s in controllers()]
    return J


def run(job):
    cond, weather, sensors, ctrl, seed = job
    r = S.run((ctrl, weather, seed, False, 0, seed == 0), sensors=sensors)
    r.update(condition=cond, sensors=sensors)
    return r


def main():
    J = jobs()
    print(f"{len(J)} runs", flush=True)
    rows, traces = [], {}
    with mp.get_context("spawn").Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(run, J), 1):
            tr = r.pop("_trace")
            if tr is not None and r["sensors"] in (None, 0, "lag"):
                traces[f"{r['condition']}|{r['controller']}|{r['weather']}|{r['sensors']}"] = tr
            rows.append(r)
            print(f"[{i}/{len(J)}] {r['condition']} {r['weather']:9s} sens={r['sensors']!s:4s} {r['controller']:17s} "
                  f"s{r['seed']} O {r['IAE_SO5_test']:.5f} N {r['IAE_SNO2_test']:.5f} fail {r['fail']} "
                  f"gate {100 * r['gated_share']:.0f}%", flush=True)
    (ROOT / "results" / "summary_paper1_heldout.json").write_text(json.dumps(
        {"protocol": "docs/PAPER1_HELDOUT_PROTOCOL.md", "windows": WINDOWS, "runs": rows}, indent=1))
    (ROOT / "results" / "traces_paper1_heldout.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
