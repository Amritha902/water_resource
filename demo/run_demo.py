"""The whole live demo from one command.

    python demo/run_demo.py            # the three live runs, one after another, hands-free
    python demo/run_demo.py --manual   # wait for you between runs (close the window, press Enter)
    python demo/run_demo.py --run 2    # only one run
    python demo/run_demo.py --check    # check this computer is ready (run it the evening before)

Each run opens a window and plays live (about 1 min each).  When it finishes
the results are printed in the terminal and the final picture stays on screen
for --hold seconds before the next run starts.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "live_demo.py"

RUNS = [
    ("Run 1 of 3 · Paper 1 set-up, dry weather",
     "orange = paper's PID prior   green = MAACC as printed   blue = our retuned prior",
     ["--paper", "1", "--weather", "dry", "--days", "2"]),
    ("Run 2 of 3 · Paper 1 set-up, the storm (days 8 to 11)",
     "the same three controllers through the storm events",
     ["--paper", "1", "--weather", "storm", "--start-day", "8", "--days", "3", "--every", "30"]),
    ("Run 3 of 3 · Paper 3 set-up, dynamic reference",
     "orange = paper's PID prior   green = IOQL as printed   blue = our retuned prior",
     ["--paper", "3", "--weather", "dry", "--days", "7", "--every", "40"]),
]


def banner(title, sub=""):
    print("\n" + "=" * 78)
    print("  " + title)
    if sub:
        print("  " + sub)
    print("=" * 78, flush=True)


def check():
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for mod in ("numpy", "scipy", "matplotlib", "bsm2_python", "openpyxl"):
        try:
            m = importlib.import_module(mod)
            print(f"  ok      {mod} {getattr(m, '__version__', '')}")
        except Exception as e:                       # noqa: BLE001
            ok = False
            print(f"  MISSING {mod}: {e}")
    import matplotlib
    backend = matplotlib.get_backend()
    gui = backend.lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template")
    print(f"  {'ok     ' if gui else 'NO     '} plot window (matplotlib backend: {backend})")
    if not gui:
        ok = False
        print("          no window can open here: install Tk (python.org Python includes it)"
              " or PyQt5, or play the recorded video")
    for f in ("warm_paper1.pkl", "warm_paper3.pkl", "Paper1_storm_day8.pkl"):
        p = ROOT / "demo" / "cache" / f
        print(f"  {'ok     ' if p.exists() else 'missing'} demo/cache/{f}")
    infl = ROOT / "data" / "official"
    have = all((infl / f"{w}.npy").exists() for w in ("dry", "storm"))
    print(f"  {'ok     ' if have else 'not yet'} influent files in data/official/"
          + ("" if have else " (downloaded on the first run; needs internet once)"))
    if ok:
        print("\nTrying a short run (a window opens for about 10 s) ...", flush=True)
        r = subprocess.run([sys.executable, str(DEMO), "--days", "0.1", "--hold", "2"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print("  ok      the plant runs and the plot window opens")
        else:
            ok = False
            print("  FAILED  short run:\n" + r.stderr[-1500:])
    print("\nREADY" if ok else "\nNOT READY: install the missing packages with  pip install -r requirements.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check the computer is ready")
    ap.add_argument("--run", type=int, choices=(1, 2, 3), help="only this run")
    ap.add_argument("--manual", action="store_true", help="wait for you between runs")
    ap.add_argument("--hold", type=float, default=12.0, help="seconds each final picture stays up")
    a = ap.parse_args()
    if a.check:
        return check()

    banner("Reinforcement-learning control of a wastewater treatment plant",
           "Live demo on the official BSM1 benchmark plant (each step = 45 s of plant time)")
    runs = [RUNS[a.run - 1]] if a.run else RUNS
    for i, (title, legend, args) in enumerate(runs):
        banner(title, legend)
        t = time.time()
        hold = [] if a.manual else ["--hold", str(a.hold)]
        subprocess.run([sys.executable, str(DEMO), *args, *hold], check=False)
        print(f"  ({time.time() - t:.0f} s)")
        if a.manual and i < len(runs) - 1:
            input("\nPress Enter for the next run ... ")
    banner("End of the live demo")


if __name__ == "__main__":
    main()
