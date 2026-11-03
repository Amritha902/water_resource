"""Paper 1: held-out test 2, docs/PAPER1_HELDOUT2_PROTOCOL.md.

Every controller frozen (nothing re-tuned or retrained), run on

  H4  six new 14-day windows of the BSM2 influent after a simple primary clarifier
  H5  the same windows raw (overload: behaviour at the actuator limits)
  H6  the official dry, rain and storm files with the official sensors, new noise seeds
  H7  H4 with the official sensors

scored on days 7-14 with the true concentrations.  Writes
results/summary_paper1_heldout2.json (and traces of the seed-0 runs).

    python experiments/43_paper1_heldout2.py
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

WINDOWS = (273, 329, 385, 441, 497, 553)
WEATHERS = ("dry", "rain", "storm")
NOISE_SEEDS = (10, 11, 12)
SINGLE = ("PID", "MAACC", "PIDtuned", "PIDtuned+ILC", "PIDtuned+ILC-AW", "Robust")
LEARNED = ("rMAACC", "Ours")
SEEDS = (0, 1, 2, 3, 4)


def controllers():
    return [(c, 0) for c in SINGLE] + [(c, s) for c in LEARNED for s in SEEDS]


def jobs():
    J = []
    for w in WINDOWS:
        J += [("H4", f"bsm2c:{w}", None, c, s) for c, s in controllers()]
        J += [("H5", f"bsm2:{w}", None, c, s) for c, s in controllers()]
        J += [("H7", f"bsm2c:{w}", 10, c, s) for c, s in controllers()]
    for w in WEATHERS:
        for n in NOISE_SEEDS:
            J += [("H6", w, n, c, s) for c, s in controllers()]
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
            if tr is not None and r["sensors"] in (None, 10):
                traces[f"{r['condition']}|{r['controller']}|{r['weather']}|{r['sensors']}"] = tr
            rows.append(r)
            print(f"[{i}/{len(J)}] {r['condition']} {r['weather']:9s} sens={r['sensors']!s:4s} {r['controller']:17s} "
                  f"s{r['seed']} O {r['IAE_SO5_test']:.5f} N {r['IAE_SNO2_test']:.5f} fail {r['fail']} "
                  f"gate {100 * r['gated_share']:.0f}%", flush=True)
    (ROOT / "results" / "summary_paper1_heldout2.json").write_text(json.dumps(
        {"protocol": "docs/PAPER1_HELDOUT2_PROTOCOL.md", "windows": WINDOWS, "runs": rows}, indent=1))
    (ROOT / "results" / "traces_paper1_heldout2.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
