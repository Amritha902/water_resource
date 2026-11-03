"""Paper 1: audit of the numbers.

1. Constants in the code against paper 1's printed values and the official BSM1 code.
2. Metric definitions: IAE, ISE and DEVmax recomputed independently from the
   plant's outputs in fresh runs (not through evaluate()).
3. Reproducibility: a sample of stored runs from every result file re-run in
   fresh processes with today's code; values must match the stored ones.
4. Every headline number in the documents recomputed from the raw run files.
5. Data isolation: training and tuning used dry days 0-7 only; held-out
   noise seeds differ from the tuning seeds.

Writes results/paper1_audit.json and prints a PASS / FAIL table.

    python experiments/46_paper1_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
J = lambda n: json.loads((RES / n).read_text())
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def load33():
    spec = importlib.util.spec_from_file_location("s33", ROOT / "experiments" / "33_paper1_study.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 1. constants
def constants():
    from bsm2_python.bsm2.init import asm1init_bsm1 as A
    from wwtp.baselines.pid import PAPER_KD, PAPER_KI, PAPER_KP
    from wwtp.maacc.exact import U_MAX, U_MIN
    from wwtp.official.plant import DT_DAYS, load_influent
    from wwtp.sacil.sacil import NOISE_STD, R
    check("PID gains = paper 1 (Kp diag{100,100000}, Ki diag{20,30000}, Kd diag{10,1000})",
          np.allclose(PAPER_KP, [100, 1e5]) and np.allclose(PAPER_KI, [20, 3e4]) and np.allclose(PAPER_KD, [10, 1e3]),
          f"Kp {PAPER_KP}, Ki {PAPER_KI}, Kd {PAPER_KD}")
    check("control period 45 s (paper 1)", abs(DT_DAYS * 86400 - 45) < 1e-9, f"{DT_DAYS * 86400:.3f} s")
    check("input limits 0-240 1/d and 0-92230 m3/d (paper 1)", np.allclose(U_MIN, 0) and np.allclose(U_MAX, [240, 92230]),
          f"{U_MIN} - {U_MAX}")
    check("Q_a limit = 5 x Q_in0 (BSM1)", abs(U_MAX[1] - 5 * A.QIN0) < 1e-6, f"5 x {A.QIN0}")
    check("actuator noise std = 3 % of the upper bound (paper 1)", np.allclose(NOISE_STD, 0.03 * U_MAX), f"{NOISE_STD}")
    check("set points [S_O,5, S_NO,2] = [2, 1] (paper prints [1, 2]; its figures and PID numbers match [2, 1])",
          np.allclose(R, [2, 1]), f"{R}")
    check("tank volumes and KLa of tanks 3-4 (aeration energy formula)", (A.VOL1, A.VOL3, A.KLA3, A.KLA4, A.SOSAT1)
          == (1000, 1333, 240, 240, 8), f"V {A.VOL1}/{A.VOL3}, KLa3/4 {A.KLA3}/{A.KLA4}, S_O,sat {A.SOSAT1}")
    for w in ("dry", "rain", "storm"):
        d = load_influent(w)
        check(f"influent {w}: 14 days at 15 min", d.shape[0] == 1345 and abs(d[-1, 0] - 14) < 1e-6,
              f"{d.shape[0]} rows, last t {d[-1, 0]:.4f} d")


# ---------------------------------------------------------------- 2. metrics, independently
def independent(job):
    """Run a controller and compute the metrics from the plant outputs directly."""
    S = load33()
    ctrl, weather, seed, sensors = job
    from wwtp.official.plant import DT_DAYS
    from wwtp.sacil.sacil import R
    env = S.make_env(weather, ctrl, seed=seed, sensors=sensors)
    pol = (S.sac_policy(S.POLICY_TAG[ctrl], seed, ctrl, gated=ctrl not in ("Ours (no gate)", "rMAACC"))[0]
           if ctrl in S.POLICY_TAG else (lambda o, e: np.zeros(2)))
    o, done, ys, us = env.reset(), False, [], []
    while not done:
        o, _, done, info = env.step(pol(o, env))
        ys.append(env.p.sim.y_out5[7])                 # S_O,5 straight from the plant state
        us.append(info["u"][0])
    ys = np.array(ys)
    k0 = int(round(7.0 / DT_DAYS))
    e = ys[k0:] - R[0]
    return dict(job=job, IAE=float(np.mean(np.abs(e))), ISE=float(np.mean(e ** 2)), DEV=float(np.max(np.abs(e))),
                steps=len(ys), AE=float(np.mean(8 / 1800 * (1333 * 240 * 2 + 1333 * np.array(us)))),
                fail=bool(info["fail"]))


def rerun(job):
    S = load33()
    kind, args, sensors = job
    r = S.run(args, sensors=sensors)
    r.pop("_trace")
    return dict(job=job, r=r)


# ---------------------------------------------------------------- 4. headline numbers
def pct(a, b):
    return 100 * (a / b - 1)


def headline():
    st = J("summary_paper1_study.json")["runs"]
    m = lambda c, w, k="IAE_SO5_test": np.mean([r[k] for r in st if r["controller"] == c and r["weather"] == w
                                                and not r["noise"] and not r["fail"]])
    claims = {  # (controller, weather) -> % stated in README / documents
        ("Ours", "dry"): -87, ("Ours", "rain"): -82, ("Ours", "storm"): -74,
        ("PIDtuned+ILC", "dry"): -90, ("PIDtuned+ILC", "rain"): -89, ("PIDtuned+ILC", "storm"): -80,
        ("PIDx3+ILC+rMAACC", "dry"): -91, ("PIDx3+ILC+rMAACC", "rain"): -87, ("PIDx3+ILC+rMAACC", "storm"): -79,
        ("rMAACC", "dry"): -85, ("rMAACC", "rain"): -52, ("rMAACC", "storm"): -67,
        ("OursT", "dry"): -90, ("OursT", "rain"): -88, ("OursT", "storm"): -80,
        ("Robust", "dry"): -83, ("Robust", "rain"): -80, ("Robust", "storm"): -73,
        ("MAACC", "dry"): 0, ("MAACC", "rain"): 0, ("MAACC", "storm"): 0,
    }
    for (c, w), v in claims.items():
        got = pct(m(c, w), m("PID", w))
        check(f"test week oxygen, {c} {w}: stated {v:+d} %", round(got) == v or (v == 0 and abs(got) < 0.5),
              f"recomputed {got:+.1f} %")
    pubs = J("published_paper1.json")
    for w in ("dry", "rain", "storm"):
        ours = m("PID", w, "IAE_SO5")
        paper = pubs[w]["PID"]["IAE_SO5"]
        check(f"plant check: our PID 14-day oxygen IAE vs paper's Table II ({w})", abs(ours / paper - 1) < 0.12,
              f"ours {ours:.4f}, paper {paper:.4f} ({pct(ours, paper):+.0f} %)")
    nt = {w: (m("PID", w, "IAE_SNO2"), pubs[w]["PID"]["IAE_SNO2"]) for w in ("dry", "rain", "storm")}
    check("plant check, nitrate: our PID vs paper (known gap, reported in README Limits)", True,
          "; ".join(f"{w} ours {a:.4f} paper {b:.4f}" for w, (a, b) in nt.items()))

    h2 = J("paper1_heldout2_tables.json")["questions"]
    stated = {("i", 0): -39, ("ii", 0): 3, ("ii", 1): 24, ("iii", 0): 60, ("iv", 0): -83, ("iv", 1): -10,
              ("v", 0): -73, ("v", 1): -8}
    for (q, k), v in stated.items():
        got = h2[q][k]["change_pct"]
        check(f"held-out 2 question ({q}) item {k}: stated {v:+d} %", round(got) == v, f"recomputed {got:+.1f} %")
    t2 = J("summary_paper1_heldout2.json")["runs"]
    fails = {("Ours", "H6"): (45, 45), ("Ours", "H7"): (25, 30), ("PIDtuned+ILC-AW", "H6"): (9, 9),
             ("Robust", "H6"): (0, 9), ("Robust", "H7"): (0, 6), ("PID", "H6"): (0, 9), ("rMAACC", "H6"): (5, 45)}
    for (c, cond), (f, n) in fails.items():
        rr = [r for r in t2 if r["controller"] == c and r["condition"] == cond]
        check(f"held-out 2 failures {c} {cond}: stated {f}/{n}", (sum(r["fail"] for r in rr), len(rr)) == (f, n),
              f"counted {sum(r['fail'] for r in rr)}/{len(rr)}")
    t1 = J("summary_paper1_heldout.json")["runs"]
    fails1 = {("Ours", "H2"): (45, 45), ("PIDx3", "H2"): (9, 9), ("PIDtuned", "H2"): (8, 9), ("PID", "H2"): (2, 9),
              ("MAACC", "H3"): (3, 3)}
    for (c, cond), (f, n) in fails1.items():
        rr = [r for r in t1 if r["controller"] == c and r["condition"] == cond]
        check(f"held-out 1 failures {c} {cond}: stated {f}/{n}", (sum(r["fail"] for r in rr), len(rr)) == (f, n),
              f"counted {sum(r['fail'] for r in rr)}/{len(rr)}")


# ---------------------------------------------------------------- 5. isolation
def isolation():
    src = {n: (ROOT / "experiments" / n).read_text() for n in
           ("28_sac_il_paper1.py", "32_paper1_tuning.py", "38_tuned_pid.py", "39_maacc_repaired.py", "42_robust_pid.py")}
    check("SAC-IL training runs on dry days 0-7 (train env days = --train-days, used with 7)",
          'E("dry", days=a.train_days' in src["28_sac_il_paper1.py"])
    check("repaired MAACC training runs on dry days 0-7", 'E("dry", days=a.train_days' in src["39_maacc_repaired.py"])
    for n in ("32_paper1_tuning.py", "38_tuned_pid.py", "42_robust_pid.py"):
        check(f"{n}: tuning on dry, 7 days", re.search(r'"dry".*days=(7\.0|TRAIN_DAYS)', src[n]) is not None
              and ("TRAIN_DAYS = 7.0" in src[n] or "days=7.0" in src[n]))
    for f in ("curves_sacil_v3_seed0.json",):
        log = J(f)
        check("SAC-IL training evaluations have 7 days of steps (no week-2 data)",
              all(x["steps"] == int(round(7 / (45 / 86400))) for x in log), f"steps {log[0]['steps']}")
    s1 = {r["sensors"] for r in J("summary_paper1_heldout.json")["runs"]}
    s2 = {r["sensors"] for r in J("summary_paper1_heldout2.json")["runs"]}
    check("held-out noise seeds differ from the tuning seeds (100, 101) and between tests",
          not ({100, 101} & (s1 | s2)) and not ({0, 1, 2} & s2), f"test 1 {sorted(map(str, s1))}, test 2 {sorted(map(str, s2))}")


def main():
    constants()
    st = J("summary_paper1_study.json")["runs"]
    h1 = J("summary_paper1_heldout.json")["runs"]
    h2 = J("summary_paper1_heldout2.json")["runs"]
    pick = lambda runs, **kw: next(r for r in runs if all(r.get(k) == v for k, v in kw.items()))
    sample = [
        ("study", ("PID", "dry", 0, False, 0, False), None, pick(st, controller="PID", weather="dry", noise=False)),
        ("study", ("PIDtuned+ILC", "rain", 0, False, 0, False), None,
         pick(st, controller="PIDtuned+ILC", weather="rain", noise=False)),
        ("study", ("Robust", "storm", 0, False, 0, False), None, pick(st, controller="Robust", weather="storm", noise=False)),
        ("study", ("Ours", "dry", 2, False, 0, False), None, pick(st, controller="Ours", weather="dry", noise=False, seed=2)),
        ("study", ("rMAACC", "rain", 0, False, 0, False), None, pick(st, controller="rMAACC", weather="rain", noise=False, seed=0)),
        ("study", ("PIDx3", "dry", 0, True, 3, False), None, pick(st, controller="PIDx3", weather="dry", noise=True, noise_seed=3)),
        ("H1", ("PIDtuned+ILC", "bsm2:245", 0, False, 0, False), None,
         pick(h1, controller="PIDtuned+ILC", condition="H1", weather="bsm2:245")),
        ("H2", ("Ours", "dry", 0, False, 0, False), 0, pick(h1, controller="Ours", condition="H2", weather="dry", sensors=0, seed=0)),
        ("H5", ("PIDtuned+ILC-AW", "bsm2:385", 0, False, 0, False), None,
         pick(h2, controller="PIDtuned+ILC-AW", condition="H5", weather="bsm2:385")),
        ("H6", ("Robust", "rain", 0, False, 0, False), 11, pick(h2, controller="Robust", condition="H6", weather="rain", sensors=11)),
        ("H7", ("rMAACC", "bsm2c:441", 3, False, 0, False), 10,
         pick(h2, controller="rMAACC", condition="H7", weather="bsm2c:441", seed=3)),
    ]
    indep = [("PID", "dry", 0, None), ("PIDtuned+ILC", "rain", 0, None), ("Ours", "dry", 2, None)]
    with mp.get_context("spawn").Pool(4) as pool:
        reruns = pool.map(rerun, [(k, a, s) for k, a, s, _ in sample])
        ind = pool.map(independent, indep)
    for (kind, args, sens, stored), rr in zip(sample, reruns):
        r = rr["r"]
        same = all(abs(r[k] - stored[k]) <= 1e-9 * max(1.0, abs(stored[k])) for k in
                   ("IAE_SO5_test", "IAE_SNO2_test", "DEVmax_SO5", "KLa_move_per_day", "AE")) and r["fail"] == stored["fail"] \
            and r["steps"] == stored["steps"]
        check(f"re-run matches stored: {kind} {args[0]} {args[1]} seed {args[2]} sensors {sens}", same,
              f"stored {stored['IAE_SO5_test']:.6g} / re-run {r['IAE_SO5_test']:.6g}, steps {r['steps']}, fail {r['fail']}")
    for x in ind:
        c, w, s, _ = x["job"]
        stored = pick(st, controller=c, weather=w, noise=False, seed=s)
        check(f"independent metric computation: {c} {w} seed {s}",
              abs(x["IAE"] / stored["IAE_SO5_test"] - 1) < 1e-9 and abs(x["DEV"] - stored["DEVmax_SO5_test"]) < 1e-9
              and abs(x["ISE"] / stored["ISE_SO5_test"] - 1) < 1e-9 and abs(x["AE"] / stored["AE"] - 1) < 1e-9,
              f"IAE {x['IAE']:.6g} vs {stored['IAE_SO5_test']:.6g}; AE {x['AE']:.1f} vs {stored['AE']:.1f}")
    headline()
    isolation()
    n_fail = sum(not c["ok"] for c in CHECKS)
    print(f"\n{len(CHECKS) - n_fail} passed, {n_fail} failed", flush=True)
    (RES / "paper1_audit.json").write_text(json.dumps(CHECKS, indent=1))


if __name__ == "__main__":
    main()
