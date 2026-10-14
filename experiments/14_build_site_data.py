"""Pack every result the website shows into ``docs/data/site-data.js``.

Sources, all in the repository:

* ``results/summary_paper1.json``      paper-1 tracking benchmark (54 runs)
* ``results/summary_offset_control.json``  learning-free controls for MAACC
* ``results/summary_aeration*.json``   paper-2 aeration benchmark, merged the
                                       same way ``09_write_results.py`` does
* ``results/summary_frontier.json``, ``summary_weights*.json``,
  ``summary_dyna.json``, ``curves_*.json``  PANDA-RL studies
* ``results/site/*.json``              re-checks and traces from
                                       ``13_export_site_data.py`` and
                                       ``16_replay_paper2.py``
* ``results/warning/*.json``           early-warning evaluation

The output is a script that assigns ``window.SITE`` rather than JSON, so the
pages also work when opened straight from disk (browsers block ``fetch`` on
``file://``).
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "data" / "site-data.js"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _merge(base: list, extra: list | None, key) -> list:
    """Later files fill gaps; the base file wins on a collision (as in 09)."""
    if not extra:
        return base
    seen = {key(r) for r in base}
    return base + [r for r in extra if key(r) not in seen]


def aggregate_paper1(runs: list[dict]) -> list[dict]:
    keys = ["IAE_SO5", "ISE_SO5", "DEVmax_SO5", "IAE_SNO2", "ISE_SNO2",
            "DEVmax_SNO2", "EQ", "AE", "PE", "OCI", "viol_NH", "viol_TN"]
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["condition"], r["weather"], r["controller"]), []).append(r)
    out = []
    for (c, w, n), rows in groups.items():
        row = {"condition": c, "weather": w, "controller": n, "n_seeds": len(rows),
               "seeds": sorted(r["seed"] for r in rows)}
        for k in keys:
            v = np.array([x[k] for x in rows], float)
            row[k] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return out


def tests_summary():
    log = RES / "site" / "pytest.log"
    if not log.exists():
        return None
    text = log.read_text()
    items = re.findall(r"^(tests/\S+::\S+) (PASSED|FAILED|SKIPPED|ERROR)", text, re.M)
    tail = re.findall(r"=+ (\d+ (?:passed|failed).*?) =+", text)
    return {"items": [{"id": i, "status": s} for i, s in items],
            "summary": tail[-1] if tail else ""}


def main() -> None:
    site: dict = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    try:
        site["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        site["commit"] = "unknown"

    # -- paper 1 -------------------------------------------------------------
    p1 = _load(RES / "summary_paper1.json")
    if p1:
        site["paper1_runs"] = p1
        site["paper1"] = aggregate_paper1(p1)
    oc = _load(RES / "summary_offset_control.json")
    if oc:
        site["paper1_offset"] = oc
    cl = _load(RES / "site" / "closed_loop.json")
    if cl:
        site["traces"] = {f"{r['condition']}|{r['weather']}|{r['controller']}": r["trace"]
                          for r in cl if "trace" in r}
        # the re-run's own summaries, to check against the committed benchmark
        site["paper1_rerun"] = [{k: r[k] for k in ("controller", "weather", "seed", "condition")}
                                | {m: r["summary"][m] for m in ("IAE_SO5", "IAE_SNO2")} for r in cl]

    # -- paper 2 -------------------------------------------------------------
    aer = _load(RES / "summary_aeration.json") or []
    for extra in ("aeration_rs", "aeration_storm_base"):
        aer = _merge(aer, _load(RES / f"summary_{extra}.json"),
                     lambda r: (r["weather"], r["method"], r["seed"]))
    site["paper2"] = {"runs": aer}
    ws = _load(RES / "summary_weights.json") or []
    ws = _merge(ws, _load(RES / "summary_weights_split.json"),
                lambda r: (r["arm"], r["beta2"], r["seed"]))
    site["paper2"]["weights"] = ws
    site["paper2"]["frontier"] = _load(RES / "summary_frontier.json")
    site["paper2"]["dyna"] = _load(RES / "summary_dyna.json")
    curves = {}
    for name in ("curves_aeration_rs", "curves_aeration_storm_base", "curves_dyna"):
        c = _load(RES / f"{name}.json")
        if c:
            curves.update({k: [{kk: h[kk] for kk in h if kk in
                                ("episode", "eval_AE", "eval_viol_NH", "eval_S_O5_mean", "train_reward")}
                               for h in v] for k, v in c.items()})
    site["paper2"]["curves"] = curves
    replays = {}
    for p in sorted((RES / "site" / "aeration").glob("replay_*.json")):
        r = json.loads(p.read_text())
        replays[f"{r['weather']}|{r['method']}"] = r
    site["paper2"]["replays"] = replays
    t2 = {}
    for p in sorted((RES / "site" / "aeration_t2").glob("replay_*.json")):
        r = json.loads(p.read_text())
        t2[f"{r['weather']}|{r['method']}"] = {"summary": r["summary"]}
    site["paper2"]["replays_t2"] = t2
    site["paper2"]["fig4"] = _load(RES / "site" / "fig4.json")

    # -- shared models -------------------------------------------------------
    for stage in ("critic", "twin", "forecaster"):
        d = _load(RES / "site" / f"{stage}.json")
        if d is not None:
            site[stage] = d
    w = _load(RES / "warning" / "metrics.json")
    if w is not None:
        site["warning"] = w
    ep = _load(RES / "warning" / "episode.json")
    if ep is not None:
        site["warning_episode"] = ep

    site["tests"] = tests_summary()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    js = "window.SITE = " + json.dumps(site, separators=(",", ":"),
                                       allow_nan=False, default=float) + ";\n"
    OUT.write_text(js)
    print(f"wrote {OUT} ({len(js) / 1e6:.2f} MB); sections: {sorted(site)}")


if __name__ == "__main__":
    main()
