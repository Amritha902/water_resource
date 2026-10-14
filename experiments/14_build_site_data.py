"""Pack the experiment outputs into ``docs/data/site-data.js`` for the website.

Reads ``results/site/*.json`` (from ``06_export_site_data.py``),
``results/warning/metrics.json`` (from ``05_warning_benchmark.py``) and the
test log, and writes one JavaScript file that assigns ``window.SITE``.  A
script file rather than JSON so the pages also work when opened straight from
disk (browsers block ``fetch`` on ``file://``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "data" / "site-data.js"


def load(name):
    p = RES / "site" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def aggregate(runs):
    """mean / std over seeds per (condition, weather, controller)."""
    keys = ["IAE_SO5", "ISE_SO5", "DEVmax_SO5", "IAE_SNO2", "ISE_SNO2",
            "DEVmax_SNO2", "EQ", "AE", "PE", "OCI", "viol_NH", "viol_TN"]
    groups = {}
    for r in runs:
        groups.setdefault((r["condition"], r["weather"], r["controller"]), []).append(r["summary"])
    out = []
    for (c, w, n), rows in sorted(groups.items()):
        row = {"condition": c, "weather": w, "controller": n, "n_seeds": len(rows)}
        for k in keys:
            v = np.array([x[k] for x in rows], float)
            row[k] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return out


def tests_summary():
    log = ROOT / "results" / "site" / "pytest.log"
    if not log.exists():
        return None
    text = log.read_text()
    items = re.findall(r"^(tests/\S+::\S+) (PASSED|FAILED|SKIPPED|ERROR)", text, re.M)
    tail = re.findall(r"=+ (.*(?:passed|failed).*) =+", text)
    return {"items": [{"id": i, "status": s} for i, s in items],
            "summary": tail[-1] if tail else ""}


def main():
    site = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    try:
        site["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        site["commit"] = "unknown"

    for stage in ("critic", "twin", "forecaster"):
        d = load(stage)
        if d is not None:
            site[stage] = d

    traces = {}
    for stage in ("closed_loop", "realistic"):
        runs = load(stage)
        if not runs:
            continue
        site.setdefault("metrics", []).extend(aggregate(runs))
        for r in runs:
            if "trace" in r:
                traces[f"{r['condition']}|{r['weather']}|{r['controller']}"] = r["trace"]
    site["traces"] = traces

    wm = RES / "warning" / "metrics.json"
    if wm.exists():
        site["warning"] = json.loads(wm.read_text())
    ep = RES / "warning" / "episode.json"
    if ep.exists():
        site["warning_episode"] = json.loads(ep.read_text())

    site["tests"] = tests_summary()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    js = "window.SITE = " + json.dumps(site, separators=(",", ":"),
                                       allow_nan=False, default=float) + ";\n"
    OUT.write_text(js)
    print(f"wrote {OUT} ({len(js) / 1e6:.2f} MB); sections: {sorted(site)}")


if __name__ == "__main__":
    sys.exit(main())
