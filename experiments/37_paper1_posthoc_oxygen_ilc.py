"""Paper 1 study, a check made after seeing the test results (not part of the main result).

The ILC on the nitrate loop, chosen on dry week 1, made nitrate worse in rain.
Here PID x3 + ILC and our trained SAC-IL policies are run again with the ILC
on the oxygen loop only (nothing retrained).  Writes
results/summary_paper1_posthoc_oxygen_ilc.json, used by step 9 of
docs/PAPER1_STUDY.md.

    python experiments/37_paper1_posthoc_oxygen_ilc.py
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
S.ILC = dict(beta=[S.ILC["beta"][0], 0.0], lead=[S.ILC["lead"][0], 0], smooth=S.ILC["smooth"])


def run(job):
    r = S.run(job)
    r.pop("_trace")
    return r


def main():
    jobs = [("PIDx3+ILC", w, 0, False, 0, False) for w in ("dry", "rain", "storm")]
    jobs += [("Ours", w, s, False, 0, False) for w in ("dry", "rain", "storm") for s in range(5)]
    jobs += [("PIDx3+ILC", "dry", 0, True, n, False) for n in range(5)]
    jobs += [("Ours", "dry", s, True, n, False) for s in range(5) for n in range(2)]
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(run, jobs)
    (ROOT / "results" / "summary_paper1_posthoc_oxygen_ilc.json").write_text(
        json.dumps({"ilc": S.ILC, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
