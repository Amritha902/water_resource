"""Live demo: the base papers' controllers running side by side on the official BSM1 plant.

    python demo/live_demo.py                        # paper 1 set-up, 2 days of dry weather
    python demo/live_demo.py --weather storm --start-day 8 --days 3
    python demo/live_demo.py --paper 3              # paper 3 set-up, dynamic reference
    python demo/live_demo.py --save demo.mp4        # write a video instead of a window

Paper 1 set-up: S_O,5 -> 2 mg/L via KLa_5 and S_NO,2 -> 1 mg/L via Q_a, with
  the paper's PID prior, MAACC exactly as printed, and our retuned prior (x3).
Paper 3 set-up: S_O,5 follows the dynamic reference of eq. (33) via KLa_5, with
  the PID prior, IOQL exactly as printed, and our retuned prior (x4).

Each controller drives its own copy of the plant, starting from the same state
as in the experiments (14 days of dry weather under the paper's PID).  That
warm-up, and any fast-forward to --start-day, is computed once and cached in
demo/cache/.  The running IAE in the legend is over the period shown; the
14-day results of the report are printed at the end.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.baselines.pid import PAPER_KD, PAPER_KI, PAPER_KP, IncrementalPID  # noqa: E402
from wwtp.ioql.ioql import IOQL, reference  # noqa: E402
from wwtp.maacc.exact import ExactMAACC, U_MAX, U_MIN  # noqa: E402
from wwtp.official.plant import DT_DAYS, OfficialBSM1  # noqa: E402

CACHE = ROOT / "demo" / "cache"
STEPS_PER_DAY = int(round(1.0 / DT_DAYS))
COLORS = ("#eb6834", "#1baf7a", "#2a78d6")        # paper's PID, paper's learner, ours


def _experiment(name):
    """Load experiments/<name>.py, so the demo uses the experiments' own warm-up."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cached(key, make):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:                     # written by another numpy/python version
            pass
    print(f"computing {key} (once; cached in demo/cache/) ...", flush=True)
    obj = make()
    f.write_bytes(pickle.dumps(obj))
    return obj


# ---------------------------------------------------------------- set-ups
class Paper1:
    names = ("PID prior (paper 1)", "MAACC as printed", "Ours: retuned prior ×3")
    outputs = (("S_O,5 (mg/L)", (1.75, 2.15)), ("S_NO,2 (mg/L)", (0.8, 1.2)))
    R = np.array([2.0, 1.0])
    legend_loc = "lower right"
    widen = True

    def __init__(self):
        st, u_end = _cached("warm_paper1", lambda: _experiment("20_exact_paper1").warm_state())
        self.state, self.u_end = st, u_end

    def controllers(self):
        u = self.u_end
        return [IncrementalPID(u_min=U_MIN, u_max=U_MAX, u0=u.copy()),
                ExactMAACC(seed=0, critic="paper", u0=u.copy()),
                IncrementalPID(PAPER_KP * 3, PAPER_KI * 3, PAPER_KD * 3, U_MIN, U_MAX, u0=u.copy())]

    def act(self, c, s, k):
        u = c(s, self.R) if isinstance(c, ExactMAACC) else c(self.R - s)
        return np.clip(u, U_MIN, U_MAX)

    def ref(self, k):
        return self.R

    def y(self, s):
        return s


class Paper3:
    names = ("PID prior (paper 3)", "IOQL as printed (run 2)", "Ours: retuned prior ×4")
    outputs = (("S_O,5 (mg/L)", (0.0, 3.0)),)
    QA = 55338.0
    legend_loc = "upper right"
    widen = False          # the IOQL run leaves the range; keep the others readable

    def __init__(self):
        st, u0 = _cached("warm_paper3", lambda: _experiment("23_ioql").warm())
        self.state, self.u0 = st, u0

    def controllers(self):
        pid = IOQL(u0=self.u0, learn=False)
        pid.A[:] = 0.0
        ours = IOQL(u0=self.u0, learn=False)
        ours.A[:] = 0.0
        ours.pid.kp, ours.pid.ki, ours.pid.kd = ours.pid.kp * 4, ours.pid.ki * 4, ours.pid.kd * 4
        return [pid, IOQL(seed=1, u0=self.u0), ours]

    def act(self, c, s, k):
        return np.array([c(s[0], reference(k * DT_DAYS, True)), self.QA])

    def ref(self, k):
        return np.array([reference(k * DT_DAYS, True)])

    def y(self, s):
        return s[:1]


# ---------------------------------------------------------------- run
def start(setup, weather, start_day):
    """Plants and controllers at start_day (fast-forwarded and cached if > 0)."""
    def make():
        loops = []
        for c in setup.controllers():
            p = OfficialBSM1(weather)
            p.set_state(setup.state)
            s = p.measurement()
            for k in range(int(start_day * STEPS_PER_DAY)):
                s = p.step(setup.act(c, s, k))
            loops.append((p.get_state(), c, s))
        return loops

    if start_day == 0:
        loops = make()
    else:
        loops = _cached(f"{type(setup).__name__}_{weather}_day{start_day:g}", make)
    out = []
    for st, c, s in loops:
        p = OfficialBSM1(weather)
        p.set_state(st)
        p.k = int(start_day * STEPS_PER_DAY)
        out.append([p, c, np.array(s)])
    return out


def stored_results(paper):
    """The 14-day results of the report, from results/."""
    res = ROOT / "results"
    if paper == 1:
        ex = json.loads((res / "summary_paper1_exact2.json").read_text())
        ours = json.loads((res / "summary_paper1_ours_p1_clean.json").read_text())
        rows = []
        for w in ("dry", "rain", "storm"):
            pid = [x for x in ours if x["weather"] == w and x["controller"] == "PID"][0]
            ma = [x for x in ex if x["weather"] == w and not x["noise"] and x["controller"] == "MAACC-paper"]
            o = [x for x in ours if x["weather"] == w and x["controller"] == "PIDx3"][0]
            rows.append((w, pid["IAE_SO5"], np.mean([x["IAE_SO5"] for x in ma]), o["IAE_SO5"],
                         pid["IAE_SNO2"], np.mean([x["IAE_SNO2"] for x in ma]), o["IAE_SNO2"]))
        print("\n14-day results (report), IAE:        oxygen                     nitrate")
        print(f"{'weather':8s} {'PID':>9s} {'MAACC':>9s} {'ours':>9s}   {'PID':>9s} {'MAACC':>9s} {'ours':>9s}")
        for r in rows:
            print(f"{r[0]:8s} " + " ".join(f"{v:9.5f}" for v in r[1:4]) + "   "
                  + " ".join(f"{v:9.5f}" for v in r[4:]))
    else:
        io = json.loads((res / "summary_ioql.json").read_text())
        ours = json.loads((res / "summary_ours_p3.json").read_text())
        print("\n14-day results (report), IAE of S_O,5, dynamic reference:")
        print(f"{'weather':8s} {'PID':>9s} {'IOQL (median)':>14s} {'ours':>9s}")
        for w in ("dry", "rain", "storm"):
            pid = [x["IAE"] for x in ours if x["weather"] == w and x["dynamic"] and x["method"] == "PID"][0]
            iq = np.median([x["IAE"] for x in io if x["weather"] == w and x["dynamic"] and x["method"] == "IOQL"])
            o = [x["IAE"] for x in ours if x["weather"] == w and x["dynamic"] and x["method"] == "PIDx4"][0]
            print(f"{w:8s} {pid:9.5f} {iq:14.3f} {o:9.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", type=int, choices=(1, 3), default=1)
    ap.add_argument("--weather", choices=("dry", "rain", "storm"), default="dry")
    ap.add_argument("--start-day", type=float, default=0.0)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=20, help="redraw every N control periods (45 s each)")
    ap.add_argument("--save", help="write an .mp4 instead of opening a window")
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()

    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup = Paper1() if a.paper == 1 else Paper3()
    loops = start(setup, a.weather, a.start_day)
    n = int(a.days * STEPS_PER_DAY)
    k0 = int(a.start_day * STEPS_PER_DAY)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                         "grid.color": "#e4e3de", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(setup.outputs), 1, figsize=(10, 3.2 + 2.4 * len(setup.outputs)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    lines, refl = [], []
    for j, (ax, (lab, lim)) in enumerate(zip(axes, setup.outputs)):
        # the paper's PID is drawn wide underneath, so it stays visible where
        # the paper's learner follows it exactly
        lines.append([ax.plot([], [], color=c, lw=lw, label=nm, zorder=z)[0]
                      for c, nm, lw, z in zip(COLORS, setup.names, (3.2, 1.0, 1.2), (1, 2, 3))])
        refl.append(ax.plot([], [], color="k", lw=0.9, ls=":", label="set point")[0])
        ax.set_ylabel(lab)
        ax.set_ylim(*lim)
        ax.set_xlim(a.start_day, a.start_day + a.days)
    axes[-1].set_xlabel("time (day)")
    title = fig.suptitle("", fontsize=12, x=0.01, ha="left")
    legends = [ax.legend(loc=setup.legend_loc, fontsize=9, ncol=2, frameon=True) for ax in axes]
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    T, Y, Rf = [], [[] for _ in loops], []
    err = np.zeros((len(loops), len(setup.outputs)))
    writer = None
    if a.save:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=a.fps, bitrate=2400)
        writer.setup(fig, a.save, dpi=110)
    else:
        plt.ion()
        plt.show()

    label = {1: "Paper 1 set-up (MAACC)", 3: "Paper 3 set-up (IOQL)"}[a.paper]
    wall = time.time()
    for i in range(n):
        k = k0 + i
        for j, lp in enumerate(loops):
            p, c, s = lp
            s = p.step(setup.act(c, s, k))
            lp[2] = s
            y = setup.y(s)
            Y[j].append(y)
            err[j] += np.abs(np.nan_to_num(y - setup.ref(k + 1), nan=10.0))
        T.append(a.start_day + (i + 1) / STEPS_PER_DAY)
        Rf.append(setup.ref(k + 1))
        if (i + 1) % a.every == 0 or i == n - 1:
            iae = err / (i + 1)
            for o in range(len(setup.outputs)):
                for j in range(len(loops)):
                    lines[o][j].set_data(T, [v[o] for v in Y[j]])
                    ratio = iae[j, o] / iae[0, o]
                    d = round(100 * (ratio - 1))
                    pct = ("" if j == 0 else f" ({ratio:.0f}× PID)" if ratio > 3
                           else " (±0 % vs PID)" if d == 0 else f" ({d:+d} % vs PID)")
                    legends[o].get_texts()[j].set_text(f"{setup.names[j]}: IAE {iae[j, o]:.4f}{pct}")
                refl[o].set_data(T, [v[o] for v in Rf])
                # widen (never shrink) the axis when a trace leaves it, e.g. in the storm
                lo, hi = axes[o].get_ylim()
                ys = np.concatenate([np.array(Y[j])[:, o] for j in range(len(loops))])
                ys = ys[np.isfinite(ys)]
                if setup.widen and ys.size and (ys.min() < lo or ys.max() > hi):
                    pad = 0.05 * (hi - lo)
                    axes[o].set_ylim(min(lo, ys.min() - pad), max(hi, ys.max() + pad))
            title.set_text(f"{label}, official BSM1, {a.weather} weather — day {T[-1]:.2f}   "
                           f"(each step = 45 s of plant time)")
            if writer:
                writer.grab_frame()
            else:
                fig.canvas.draw_idle()
                plt.pause(0.001)
    if writer:
        writer.finish()
        print("wrote", a.save)

    iae = err / n
    print(f"\n{label}, {a.weather}, days {a.start_day:g}–{a.start_day + a.days:g} "
          f"({n} control periods, {time.time() - wall:.0f} s wall time)")
    for o, (lab, _) in enumerate(setup.outputs):
        print(f"  IAE {lab.split(' ')[0]}:  " + "   ".join(
            f"{nm}: {iae[j, o]:.4f}" for j, nm in enumerate(setup.names)))
    stored_results(a.paper)
    if not a.save:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
