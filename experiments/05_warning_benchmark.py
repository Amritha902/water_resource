"""Train and evaluate the effluent-ammonium early warning.

Needs the rollouts from ``04_collect_warning_data.py``.  Writes

    artifacts/warning.pt                 the trained network
    results/warning/metrics.json         every number quoted in docs/05
    docs/figures/warning_*.png           the figures
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wwtp.warning import baselines, evaluate as ev
from wwtp.warning.dataset import (CHANNELS, GROUPS, HORIZONS, NH_LIMIT,
                                  STEP_MINUTES, Normaliser, window)
from wwtp.warning.train import load_warning, predict, train_warning

H60 = HORIZONS.index(10)          # 60-min horizon
H120 = HORIZONS.index(20)         # alarm horizon
FALSE_ALARM_BUDGET = 1.0          # per compliant day

ABLATIONS = {
    "no influent forecast": GROUPS["influent forecast"],
    "no effluent analyser": GROUPS["effluent analyser"],
}


def sequences(c, score):
    """Split a stride-1 corpus back into per-rollout score / truth sequences."""
    out_s, out_nh = [], []
    for r in np.unique(c.rollout):
        m = c.rollout == r
        order = np.argsort(c.step[m])
        out_s.append(score[m][order])
        out_nh.append(c.nh_now[m][order])
    return out_s, out_nh


def permutation_importance(model, c, drop, rng, repeats: int = 3) -> dict:
    """Drop in onset AUPRC@60 min when one signal group is shuffled."""
    m = ev.onset_mask(c)
    x = c.x[m]
    y = c.event[m, H60]
    from sklearn.metrics import average_precision_score
    base = average_precision_score(y, predict(model, x, drop=drop)[1][:, H60])
    out = {}
    for g, chans in GROUPS.items():
        idx = [CHANNELS.index(ch) for ch in chans]
        drops = []
        for _ in range(repeats):
            xp = x.copy()
            perm = rng.permutation(len(xp))
            xp[..., idx] = x[perm][..., idx]
            drops.append(base - average_precision_score(
                y, predict(model, xp, drop=drop)[1][:, H60]))
        out[g] = float(np.mean(drops))
    return {"base_auprc": float(base), "drop": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/warning_rollouts.pkl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results/warning")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-ablations", action="store_true")
    args = ap.parse_args()

    data = pickle.load(open(args.data, "rb"))
    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    norm = Normaliser().fit([r["x"] for r in data["train"]])
    train = window(data["train"], norm, stride=2)
    val = window(data["val"], norm)
    test = window(data["test"], norm)
    print(f"windows: train {len(train)} val {len(val)} test {len(test)}; "
          f"test prevalence@120min {test.event[:, H120].mean():.3f}, "
          f"onset share {ev.onset_mask(test).mean():.3f}", flush=True)

    # -- networks ---------------------------------------------------------
    variants = {"Early-warning net": ()}
    if not args.no_ablations:
        variants.update({f"net, {k}": v for k, v in ABLATIONS.items()})
    nets = {}
    for name, drop_ch in variants.items():
        tag = "warning" if not drop_ch else "warning_" + name.split(", ")[1].replace(" ", "_")
        path = art / f"{tag}.pt"
        if args.retrain or not path.exists():
            print(f"\n== training {name}", flush=True)
            train_warning(train, val, norm, path, epochs=args.epochs,
                          drop_channels=tuple(drop_ch))
        nets[name] = load_warning(path)

    # -- baselines --------------------------------------------------------
    print("\n== gradient boosting", flush=True)
    gbm = baselines.GBMWarning().fit(train)

    def scores(c):
        s = {name: predict(m, c.x, drop=d)[1] for name, (m, _, d) in nets.items()}
        s["Gradient boosting"] = gbm.predict(c)
        s["Analyser trend"] = baselines.trend_score(c, norm)
        s["Analyser alarm"] = baselines.persistence_score(c, norm)
        return s

    s_val, s_test = scores(val), scores(test)
    prob = {k for k in s_test if k.startswith(("Early", "net", "Gradient"))}

    results = {"n_windows": {"train": len(train), "val": len(val), "test": len(test)},
               "test_prevalence": {f"{int(h * STEP_MINUTES)}min": float(test.event[:, j].mean())
                                   for j, h in enumerate(HORIZONS)},
               "sample": {}, "operational": {}, "canonical": {}}
    for k, s in s_test.items():
        results["sample"][k] = ev.sample_metrics(s, test, k in prob)

    # -- operational: equal false-alarm budget, thresholds from validation --
    for k in s_test:
        sv, nv = sequences(val, s_val[k][:, H120])
        st, nt = sequences(test, s_test[k][:, H120])
        cands = np.unique(np.quantile(np.concatenate(sv), np.linspace(0.3, 0.999, 200)))
        th = ev.threshold_for_budget(sv, nv, FALSE_ALARM_BUDGET, cands)
        results["operational"][k] = ev.operational(st, nt, th)
    # today's practice, untuned: alarm when the analyser reads above the limit
    st, nt = sequences(test, s_test["Analyser alarm"][:, H120])
    results["operational"]["Analyser alarm (at limit)"] = ev.operational(st, nt, NH_LIMIT)

    # -- canonical BSM1 profiles at fixed stress levels -------------------
    net, _, d0 = nets["Early-warning net"]
    for roll in data["canonical"]:
        c = window([roll], norm)
        if len(c) == 0:
            continue
        p = predict(net, c.x, drop=d0)[1][:, H120]
        th = results["operational"]["Early-warning net"]["threshold"]
        results["canonical"][roll["name"]] = {
            "violation_fraction": float((c.nh_now > NH_LIMIT).mean()),
            "net": ev.operational(*sequences(c, p), th),
            "analyser_at_limit": ev.operational(
                *sequences(c, baselines.persistence_score(c, norm)[:, H120]), NH_LIMIT),
        }

    # -- attribution --------------------------------------------------------
    results["importance"] = permutation_importance(
        net, test, d0, np.random.default_rng(0))

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    report(results)
    figures(results, data, nets, norm, s_test, test, Path(args.figures))


def report(r: dict) -> None:
    print("\n== onset subset (plant compliant now) -- AUPRC by horizon")
    keys = [f"onset@{int(h * STEP_MINUTES)}min" for h in HORIZONS]
    print(f"{'method':32s}" + "".join(f"{k.split('@')[1]:>9s}" for k in keys))
    for m, rows in r["sample"].items():
        print(f"{m:32s}" + "".join(f"{rows.get(k, {}).get('auprc', np.nan):9.3f}" for k in keys))
    print("\n== operational (alarm on the 120-min score, <= 1 false alarm / day on val)")
    for m, o in r["operational"].items():
        print(f"{m:32s} lead {o['median_lead_min']:6.0f} min  warned>=30min "
              f"{o['frac_warned_30min']:.2f}  before onset {o['frac_warned_before']:.2f}  "
              f"FA/day {o['false_alarms_per_day']:.2f}  episodes {o['n_episodes']}")
    print("\n== permutation importance (drop in onset AUPRC@60min)")
    for g, v in sorted(r["importance"]["drop"].items(), key=lambda t: -t[1]):
        print(f"  {g:22s} {v:+.3f}")


def figures(r, data, nets, norm, s_test, test, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from wwtp.utils import style
    style.apply()
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["Early-warning net", "Gradient boosting", "Analyser trend", "Analyser alarm"]
    mins = [int(h * STEP_MINUTES) for h in HORIZONS]

    # 1. onset AUPRC vs horizon
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for m in methods:
        y = [r["sample"][m].get(f"onset@{h}min", {}).get("auprc", np.nan) for h in mins]
        ax.plot(mins, y, marker="o", ms=3.5, color=style.WARNING[m], label=m)
        ax.annotate(m, (mins[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=style.WARNING[m], va="center")
    ax.set_xlabel("warning horizon [min]")
    ax.set_ylabel("AUPRC, plant compliant now")
    ax.set_title("Predicting new ammonium violations")
    ax.set_xlim(0, 175)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "warning_auprc_horizon.png")
    plt.close(fig)

    # 2. reliability of the 60-min probability
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=style.REFERENCE, lw=1, ls="--")
    for m in ("Early-warning net", "Gradient boosting"):
        conf, freq, _ = ev.reliability(s_test[m][:, H60], test.event[:, H60])
        ax.plot(conf, freq, marker="o", ms=3.5, color=style.WARNING[m], label=m)
    ax.set_xlabel("predicted P(violation within 60 min)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration, test plants")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "warning_calibration.png")
    plt.close(fig)

    # 3. importance
    imp = sorted(r["importance"]["drop"].items(), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh([g for g, _ in imp], [v for _, v in imp], color=style.WARNING["Early-warning net"])
    for i, (_, v) in enumerate(imp):
        ax.text(v, i, f" {v:+.3f}", va="center", fontsize=7.5, color=style.INK_SECONDARY)
    ax.set_xlabel("drop in onset AUPRC@60 min when shuffled")
    ax.set_title("Which measured signals the warning relies on")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "warning_importance.png")
    plt.close(fig)

    # 4. one episode on a canonical storm: what the operator would see
    net, _, d0 = nets["Early-warning net"]
    roll = next((c for c in data["canonical"] if c["name"] == "storm|trimmed"), None)
    if roll is None:
        return
    c = window([roll], norm)
    levels, p = predict(net, c.x, drop=d0)
    t = roll["t"][c.step]
    th = r["operational"]["Early-warning net"]["threshold"]
    sel = (t > 7.5) & (t < 11.5)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.axhline(NH_LIMIT, color=style.REFERENCE, lw=1, ls="--")
    a1.text(t[sel][0], NH_LIMIT + 0.15, "discharge limit", fontsize=7.5, color=style.INK_MUTED)
    shift = HORIZONS[H60] * STEP_MINUTES / 1440.0
    a1.fill_between(t[sel] + shift, levels[sel, H60, 0], levels[sel, H60, 2],
                    color=style.WARNING["Early-warning net"], alpha=0.18, lw=0,
                    label="60-min-ahead 10-90 % band")
    a1.plot(t[sel] + shift, levels[sel, H60, 1], color=style.WARNING["Early-warning net"],
            lw=1.2, label="60-min-ahead median")
    a1.plot(t[sel], c.nh_now[sel], color=style.INK, lw=1.4, label="true effluent NH4")
    a1.plot(t[sel], c.current[sel], color=style.WARNING["Analyser alarm"], lw=1,
            label="analyser reading")
    a1.set_ylabel("NH4-N [g/m3]")
    a1.set_title("Canonical storm, trimmed aeration: forecast vs what happened")
    a1.legend(loc="upper left", ncol=2, fontsize=7.5)
    a2.plot(t[sel], p[sel, H120], color=style.WARNING["Early-warning net"])
    a2.axhline(th, color=style.REFERENCE, lw=1, ls="--")
    a2.text(t[sel][0], th + 0.03, "alarm threshold", fontsize=7.5, color=style.INK_MUTED)
    a2.set_ylabel("P(violation\nwithin 2 h)")
    a2.set_xlabel("time [d]")
    a2.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "warning_episode.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
