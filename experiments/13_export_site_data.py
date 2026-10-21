"""Run every experiment the project website shows and export the raw numbers.

Nothing on the website is drawn by hand: each chart reads a file written by
this script, from the same code the tests exercise.  Stages (each cached in
``results/site/<stage>.json``; pass ``--force`` to recompute):

``closed_loop``  PID / MAACC / PANDA on the canonical dry / rain / storm
                 profiles under the paper's noise protocol, with every
                 internal signal of the control law recorded (b, a, g, f, Q).
``realistic``    the same three controllers on the storm profile with
                 realistic instrumentation (actuator slew limits, DO-probe
                 lag, a stale nitrate analyser) -- the regime in which a
                 forecast should matter.
``forecaster``   the influent forecaster on the canonical profiles and on 24
                 fresh random scenarios it has never seen, against a
                 persistence forecast.
``twin``         step responses of the real plant vs the digital twin, and
                 free-running prediction on fresh rollouts against
                 persistence.
``critic``       eq. (15) of the paper vs the TD(0) correction on the
                 synthetic problem of ``tests/test_critic_update.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "results" / "site"
WEATHERS = ("dry", "rain", "storm")
CONTROLLERS = ("PID", "MAACC", "PANDA")
#: the paper's supplementary noise protocol: 3 % of each actuator's range
ACT_NOISE = np.array([0.03 * 240.0, 0.03 * 92230.0])
SENSOR_NOISE = np.array([0.05, 0.10])
#: realistic instrumentation
REALISTIC = dict(slew_rate=np.array([6.0, 2200.0]),
                 sensor_tau_seconds=np.array([120.0, 1e-9]),
                 analyzer_period_seconds=np.array([45.0, 600.0]))
EVERY = 20                         # record one sample per 15 min


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


# -- closed loop ---------------------------------------------------------------
class Recorder:
    """Wraps a controller and records the terms of its control law."""

    def __init__(self, ctrl):
        self.c = ctrl
        self.name = ctrl.name
        self.rows = []

    def reset(self):
        self.c.reset()
        self.rows = []

    def act(self, k, s, r, ctx):
        u = self.c.act(k, s, r, ctx)
        if k % EVERY == 0:
            row = {"u": u.tolist()}
            if hasattr(self.c, "agents"):
                row["b"] = self.c.pid.u.tolist()
                row["a"] = [ag.a for ag in self.c.agents]
                row["q"] = [ag.q_prev if ag.q_prev is not None else 0.0
                            for ag in self.c.agents]
                row["g"] = self.c._gate(ctx).tolist()
                row["f"] = np.asarray(self.c._feedforward(ctx)).tolist()
            else:
                row["b"] = u.tolist()
            if hasattr(self.c, "predictor"):
                row["p_wet"] = float(self.c.predictor.regime[1:].sum())
                row["nh_risk"] = float(self.c._nh_risk)
            self.rows.append(row)
        return u

    @property
    def diagnostics(self):
        return {}


def make(name, seed=0):
    from wwtp.baselines.pid_controller import PIDController
    from wwtp.maacc.agent import MAACCController
    from wwtp.panda.controller import PANDAController
    if name == "PID":
        return PIDController()
    if name == "MAACC":
        return MAACCController(seed=seed)
    return PANDAController(ART / "forecaster.pt", ART / "twin.pt", seed=seed)


def _run(job):
    name, weather, seed, cond = job
    import torch
    torch.set_num_threads(1)
    from wwtp.bsm1 import asm1, influent
    from wwtp.envs import run_closed_loop
    t0 = time.time()
    rec = Recorder(make(name, seed))
    kw = dict(actuator_noise_std=ACT_NOISE, sensor_noise_std=SENSOR_NOISE,
              seed=100 + seed)
    if cond == "realistic":
        kw.update(REALISTIC)
    res = run_closed_loop(rec, influent.canonical_scenario(weather), **kw)
    summ = res.summary(burn_in_days=1.0)
    out = {"controller": name, "weather": weather, "seed": seed,
           "condition": cond, "summary": summ,
           "seconds": round(time.time() - t0, 1)}
    if seed == 0:
        sl = slice(None, None, EVERY)
        tn = asm1.total_nitrogen(res.effluent)
        out["trace"] = {
            "t": _r(res.time_days[sl], 4),
            "s": _r(res.state[sl], 4),
            "u": _r(res.control[sl], 2),
            "q_in": _r(res.influent_flow[sl], 0),
            "nh_eff": _r(res.effluent[sl, asm1.S_NH], 3),
            "tn_eff": _r(tn[sl], 3),
            "aeration": _r(res.aeration[sl], 1),
            "law": rec.rows,
        }
    print(f"  {cond:9s} {weather:5s} {name:5s} s{seed}  IAE "
          f"{summ['IAE_SO5']:.4f}/{summ['IAE_SNO2']:.4f}  ({out['seconds']}s)",
          flush=True)
    return out


def stage_closed_loop(pool, seeds):
    jobs = [(n, w, s, "noisy") for w in WEATHERS for n in CONTROLLERS
            for s in (range(seeds) if n != "PID" else range(seeds))]
    return pool.map(_run, jobs, chunksize=1)


def stage_realistic(pool, seeds):
    jobs = [(n, w, s, "realistic") for w in ("storm", "rain") for n in CONTROLLERS
            for s in range(seeds)]
    return pool.map(_run, jobs, chunksize=1)


# -- forecaster ----------------------------------------------------------------
def stage_forecaster():
    import torch
    from wwtp.bsm1.influent import canonical_scenario, random_scenario
    from wwtp.forecast.dataset import (HORIZON, LOOKBACK, QUANTILES, REGIMES,
                                       add_sensor_noise, observable)
    from wwtp.forecast.models import InfluentForecaster
    blob = torch.load(ART / "forecaster.pt", map_location="cpu", weights_only=False)
    model = InfluentForecaster()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    mean, std = blob["mean"], blob["std"]

    def run(obs):
        """All forecasts over a series: (T', H, C, Q) physical, regime probs."""
        z = (np.log(np.maximum(obs, 1e-3)) - mean) / std
        starts = np.arange(0, len(z) - LOOKBACK + 1)
        x = np.stack([z[s:s + LOOKBACK] for s in starts]).astype(np.float32)
        with torch.no_grad():
            q, logits, _ = model(torch.from_numpy(x))
        q = np.exp(q.numpy() * std[None, None, :, None] + mean[None, None, :, None])
        return starts + LOOKBACK - 1, q, torch.softmax(logits, -1).numpy()

    def score(obs, idx, q):
        """Median MAPE at +2 h vs persistence, and 10-90 % coverage."""
        last = idx[idx + HORIZON < len(obs)]
        qq = q[:len(last)]
        truth = obs[last + HORIZON]                       # (N, C)
        med = qq[:, -1, :, 1]
        pers = obs[last]
        mape = np.mean(np.abs(med - truth) / truth, 0)
        mape_p = np.mean(np.abs(pers - truth) / truth, 0)
        cov = np.mean((truth >= qq[:, -1, :, 0]) & (truth <= qq[:, -1, :, 2]), 0)
        return mape, mape_p, cov

    rng = np.random.default_rng(777)       # seeds never used in training
    acc, n, mapes, mapes_p, covs, conf = 0, 0, [], [], [], np.zeros((3, 3))
    for _ in range(24):
        series = random_scenario(rng)
        obs = add_sensor_noise(observable(series), rng)
        idx, q, p = run(obs)
        true_r = REGIMES.index(series.weather)
        pred_r = p.argmax(1)
        acc += int((pred_r == true_r).sum()); n += len(pred_r)
        for pr in pred_r:
            conf[true_r, pr] += 1
        m, mp, c = score(obs, idx, q)
        mapes.append(m); mapes_p.append(mp); covs.append(c)
    heldout = {"n_scenarios": 24, "regime_accuracy": acc / n,
               "regime_confusion": conf.tolist(),
               "mape_2h": np.mean(mapes, 0).tolist(),
               "mape_2h_persistence": np.mean(mapes_p, 0).tolist(),
               "coverage_10_90_2h": np.mean(covs, 0).tolist(),
               "channels": ["Q_in", "S_NH_in", "S_S_in"]}

    canon = {}
    for w in WEATHERS:
        series = canonical_scenario(w)
        obs = add_sensor_noise(observable(series), np.random.default_rng(5))
        idx, q, p = run(obs)
        m, mp, c = score(obs, idx, q)
        t = idx * 900.0 / 86400.0
        canon[w] = {"t": _r(t, 4), "actual": _r(obs[:, 0], 0),
                    "t_actual": _r(np.arange(len(obs)) * 900.0 / 86400.0, 4),
                    "q10_2h": _r(q[:, -1, 0, 0], 0), "q50_2h": _r(q[:, -1, 0, 1], 0),
                    "q90_2h": _r(q[:, -1, 0, 2], 0),
                    "nh_actual": _r(obs[:, 1], 2), "nh50_2h": _r(q[:, -1, 1, 1], 2),
                    "regime": _r(p, 3), "mape_2h": m.tolist(),
                    "mape_2h_persistence": mp.tolist(), "coverage_10_90_2h": c.tolist()}
    hist = json.loads((ART / "forecaster_history.json").read_text())
    return {"heldout": heldout, "canonical": canon, "training_history": hist,
            "quantiles": list(QUANTILES), "horizon_steps": HORIZON,
            "lookback_steps": LOOKBACK}


# -- digital twin --------------------------------------------------------------
def stage_twin():
    import torch
    from wwtp.bsm1 import asm1
    from wwtp.bsm1.influent import DRY_AVERAGE_COMPOSITION, DRY_AVERAGE_FLOW
    from wwtp.bsm1.plant import BSM1Plant
    from wwtp.twin.dataset import (COARSE_STRIDE, D_SCALE, S_SCALE, U_SCALE,
                                   collect_rollout)
    from wwtp.twin.models import DigitalTwin
    blob = torch.load(ART / "twin.pt", map_location="cpu", weights_only=False)
    twin = DigitalTwin()
    twin.load_state_dict(blob["state_dict"])
    twin.eval()
    f32 = torch.float32
    S, U, D = (torch.tensor(v, dtype=f32) for v in (S_SCALE, U_SCALE, D_SCALE))

    # 1. step responses under constant dry-average influent
    d_in = np.array([DRY_AVERAGE_FLOW, DRY_AVERAGE_COMPOSITION[asm1.S_NH],
                     DRY_AVERAGE_COMPOSITION[asm1.S_S]])
    steps = {}
    for label, u0, u1 in (("KLa_5 84 -> 160", (84.0, 55338.0), (160.0, 55338.0)),
                          ("KLa_5 84 -> 40", (84.0, 55338.0), (40.0, 55338.0)),
                          ("Q_a 55338 -> 80000", (84.0, 55338.0), (84.0, 80000.0)),
                          ("Q_a 55338 -> 25000", (84.0, 55338.0), (84.0, 25000.0))):
        plant = BSM1Plant()
        u0, u1 = np.array(u0), np.array(u1)
        hist_s, hist_u = [], []
        n_pre, n_post = 2 * 24 * 80, 4 * 80            # 2 d settle, 4 h step
        for k in range(n_pre + n_post):
            u = u0 if k < n_pre else u1
            if k % COARSE_STRIDE == 0:
                hist_s.append(plant.measurement()); hist_u.append(u.copy())
            plant.step(45.0, DRY_AVERAGE_COMPOSITION, DRY_AVERAGE_FLOW, u)
        hist_s, hist_u = np.array(hist_s), np.array(hist_u)
        k0 = n_pre // COARSE_STRIDE
        warm = 30
        with torch.no_grad():
            h = twin.init_hidden(1)
            d = torch.tensor(d_in, dtype=f32)[None] / D
            for j in range(k0 - warm, k0):            # teacher-forced warm-up
                s = torch.tensor(hist_s[j], dtype=f32)[None] / S
                u = torch.tensor(hist_u[j], dtype=f32)[None] / U
                _, _, h = twin.step(s, u, d, h)
            s = torch.tensor(hist_s[k0], dtype=f32)[None] / S
            pred = [hist_s[k0]]
            for j in range(k0, len(hist_s) - 1):
                u = torch.tensor(hist_u[j], dtype=f32)[None] / U
                s, _, h = twin.step(s, u, d, h)
                pred.append((s[0] * S).numpy())
        t_min = (np.arange(len(hist_s) - k0) * COARSE_STRIDE * 45.0 / 60.0)
        steps[label] = {"t_min": _r(t_min, 1), "plant": _r(hist_s[k0:], 4),
                        "twin": _r(np.array(pred), 4)}

    # 2. free-running prediction on fresh rollouts
    rng = np.random.default_rng(4242)
    errs, errs_p, example = [], [], None
    for i in range(6):
        roll = collect_rollout(rng, days=6.0, mode="pid" if i % 2 else "prbs")
        s_all, u_all, d_all = roll["s"], roll["u"], roll["d"]
        for start in range(200, len(s_all) - 70, 37):
            with torch.no_grad():
                h = twin.init_hidden(1)
                for j in range(start - 16, start):
                    _, _, h = twin.step(torch.tensor(s_all[j], dtype=f32)[None] / S,
                                        torch.tensor(u_all[j], dtype=f32)[None] / U,
                                        torch.tensor(d_all[j], dtype=f32)[None] / D, h)
                s = torch.tensor(s_all[start], dtype=f32)[None] / S
                pred = []
                for j in range(start, start + 48):
                    s, _, h = twin.step(s, torch.tensor(u_all[j], dtype=f32)[None] / U,
                                        torch.tensor(d_all[j], dtype=f32)[None] / D, h)
                    pred.append((s[0] * S).numpy())
            pred = np.array(pred)
            truth = s_all[start + 1:start + 49]
            errs.append((pred - truth) ** 2)
            errs_p.append((s_all[start][None] - truth) ** 2)
            if example is None and roll["mode"] == "prbs":
                example = {"t_min": _r(np.arange(1, 49) * 6.0, 1),
                           "truth": _r(truth, 4), "twin": _r(pred, 4),
                           "u": _r(u_all[start:start + 48], 1)}
    errs, errs_p = np.array(errs), np.array(errs_p)
    free = {"n_windows": int(len(errs)), "horizon_min": 48 * 6,
            "rmse_twin": np.sqrt(errs.mean((0, 1))).tolist(),
            "rmse_persistence": np.sqrt(errs_p.mean((0, 1))).tolist(),
            "rmse_twin_by_step": _r(np.sqrt(errs.mean((0, 2))), 4),
            "rmse_persistence_by_step": _r(np.sqrt(errs_p.mean((0, 2))), 4),
            "channels": ["S_O,5", "S_NO,2"], "example": example}
    hist = json.loads((ART / "twin_history.json").read_text())
    return {"steps": steps, "free_run": free, "training_history": hist}


# -- paper 2, Figure 4 ------------------------------------------------------------
def _fig4_point(sp):
    """Same procedure as experiments/07_figures.py::fig_dose_response."""
    from wwtp.bsm1 import asm1, influent
    from wwtp.bsm1.plant import BSM1Plant
    z_in, q = influent.DRY_AVERAGE_COMPOSITION, influent.DRY_AVERAGE_FLOW
    plant = BSM1Plant()
    kla, e1 = 84.0, 0.0
    for _ in range(int(20 * 86400 / 45)):
        m = plant.step(45.0, z_in, q, np.array([kla, 55338.0]))
        e = sp - m[0]
        kla = float(np.clip(kla + 60.0 * (e - e1) + 12.0 * e, 0, 240))
        e1 = e
    ze = plant.z_effluent
    return {"do": sp, "kla": kla, "nh": float(ze[asm1.S_NH]),
            "tn": float(asm1.total_nitrogen(ze)), "cod": float(asm1.total_cod(ze)),
            "bod": float(asm1.bod5(ze)), "tss": float(asm1.tss(ze)),
            "ae": plant.aeration_energy_rate(np.array([kla, 0.0]))}


def stage_fig4(pool):
    pts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    return pool.map(_fig4_point, pts, chunksize=1)


# -- critic ----------------------------------------------------------------------
def stage_critic():
    from wwtp.maacc.networks import ShallowNet
    lam, lr = 0.9, 0.02
    out = {}
    for mode in ("paper", "td0"):
        net = ShallowNet(4, 24, 1, rng=np.random.default_rng(0), inner_scale=1.0)
        rng = np.random.default_rng(1)
        a, x_prev, q_prev = 0.0, None, None
        probe, steps = [], []
        for k in range(40_000):
            a = float(np.clip(0.995 * a + rng.normal(0.0, 0.1), -1.0, 1.0))
            z = rng.normal(0.0, 0.3, 2)
            x = np.array([z[0], z[1], 0.0, a])
            util = z[0] ** 2 + 0.05 * a ** 2
            q = float(net.forward(x)[0])
            if x_prev is not None:
                if mode == "paper":
                    net.sgd_outer(np.array([(util + lam * q - q_prev) * lam]), lr)
                else:
                    err = float(net.forward(x_prev)[0]) - (util + lam * q)
                    net.sgd_outer(np.array([err]), lr)
            q_prev = float(net.forward(x)[0])
            x_prev = x
            if k % 200 == 0:
                probe.append(float(net.forward(np.zeros(4))[0]))
                steps.append(k)
        out[mode] = {"step": steps, "q0": _r(probe, 4)}
    # ground truth by Monte Carlo.  The critic is trained towards
    # Q(x_{k-1}) = U_k + lambda Q(x_k), so Q(x) = sum_{j>=1} lambda^{j-1} U_j
    # over the utilities of the *following* states.
    mc = np.random.default_rng(9)
    a_mc, tot = np.zeros(200_000), np.zeros(200_000)
    for j in range(1, 150):
        a_mc = np.clip(0.995 * a_mc + mc.normal(0.0, 0.1, a_mc.size), -1.0, 1.0)
        z_mc = mc.normal(0.0, 0.3, a_mc.size)
        tot += lam ** (j - 1) * (z_mc ** 2 + 0.05 * a_mc ** 2)
    out["true_q0"] = float(tot.mean())
    out["lambda"] = lam
    out["lr"] = lr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="*",
                    default=["critic", "twin", "forecaster", "closed_loop"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="trace seeds; the metric tables come from results/summary_paper1.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        for stage in args.stages:
            path = OUT / f"{stage}.json"
            if path.exists() and not args.force:
                print(f"{stage}: cached")
                continue
            t0 = time.time()
            print(f"== {stage}", flush=True)
            if stage == "closed_loop":
                data = stage_closed_loop(pool, args.seeds)
            elif stage == "fig4":
                data = stage_fig4(pool)
            elif stage == "realistic":
                data = stage_realistic(pool, args.seeds)
            else:
                data = globals()[f"stage_{stage}"]()
            path.write_text(json.dumps(data))
            print(f"{stage}: {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
