"""Figures for the aeration study.

Reads ``results/summary_aeration.json``, ``traces_aeration.json`` and
``curves_aeration.json`` written by ``06_aeration_benchmark.py``, plus it can
regenerate the Figure-4 sweep from the plant directly.

Every figure carries a legend and direct labels, because three of the five
series colours sit below 3:1 contrast on the light surface, and no figure
uses two y-scales.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wwtp.utils import style

ORDER = ("PID", "Fuzzy", "DDPG-A", "DDPG-B", "PANDA-RL")


def _label_last(ax, x, y, text, color) -> None:
    """Direct label at the end of a line (the contrast relief rule)."""
    if len(x) == 0:
        return
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, 0),
                textcoords="offset points", color=color, fontsize=8,
                va="center", fontweight="bold")


def fig_dose_response(out: Path) -> None:
    """Reproduce Figure 4 of Du et al.: effluent versus DO set point."""
    from wwtp.bsm1 import asm1, influent
    from wwtp.bsm1.plant import BSM1Plant

    z_in, q = influent.DRY_AVERAGE_COMPOSITION, influent.DRY_AVERAGE_FLOW
    setpoints = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    nh, tn, cod, bod, tss, ae = ([] for _ in range(6))
    for sp in setpoints:
        plant = BSM1Plant()
        kla, e1 = 84.0, 0.0
        for _ in range(int(20 * 86400 / 45)):
            m = plant.step(45.0, z_in, q, np.array([kla, 55338.0]))
            e = sp - m[0]
            kla = float(np.clip(kla + 60.0 * (e - e1) + 12.0 * e, 0, 240))
            e1 = e
        ze = plant.z_effluent
        nh.append(float(ze[asm1.S_NH]))
        tn.append(float(asm1.total_nitrogen(ze)))
        cod.append(float(asm1.total_cod(ze)))
        bod.append(float(asm1.bod5(ze)))
        tss.append(float(asm1.tss(ze)))
        ae.append(plant.aeration_energy_rate(np.array([kla, 0.0])))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axes[0]
    ax.plot(setpoints, nh, color=style.SERIES["PID"], marker="o", ms=4)
    ax.axhline(4.0, color=style.REFERENCE, ls="--", lw=1)
    _label_last(ax, setpoints, nh, "NH4-N", style.SERIES["PID"])
    ax.annotate("limit 4", xy=(setpoints[0], 4.0), xytext=(0, 4),
                textcoords="offset points", color=style.INK_MUTED, fontsize=8)
    ax.set_title("Effluent ammonium falls with DO")
    ax.set_xlabel("DO set point in tank 5 [g/m$^3$]")
    ax.set_ylabel("S$_{NH,e}$ [g N/m$^3$]")

    ax = axes[1]
    ax.plot(setpoints, tn, color=style.SERIES["MAACC"], marker="o", ms=4)
    ax.axhline(18.0, color=style.REFERENCE, ls="--", lw=1)
    _label_last(ax, setpoints, tn, "TN", style.SERIES["MAACC"])
    ax.annotate("limit 18", xy=(setpoints[0], 18.0), xytext=(0, 4),
                textcoords="offset points", color=style.INK_MUTED, fontsize=8)
    ax.set_title("Effluent total nitrogen rises with DO")
    ax.set_xlabel("DO set point in tank 5 [g/m$^3$]")
    ax.set_ylabel("N$_{tot,e}$ [g N/m$^3$]")

    ax = axes[2]
    for values, name, colour in ((cod, "COD", style.SERIES["PID"]),
                                 (tss, "TSS", style.SERIES["MAACC"]),
                                 (bod, "BOD5", style.SERIES["PANDA"])):
        ax.plot(setpoints, values, color=colour, marker="o", ms=3)
        _label_last(ax, setpoints, values, name, colour)
    ax.set_title("The others barely move")
    ax.set_xlabel("DO set point in tank 5 [g/m$^3$]")
    ax.set_ylabel("effluent [g/m$^3$]")
    ax.set_ylim(0, max(cod) * 1.25)

    fig.suptitle("The conflict paper 2 exploits, reproduced on our plant",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig1_dose_response.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_dose_response.png")


def fig_learning_curves(curves: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, key, ylabel, title in (
            (axes[0], "eval_AE", "aeration energy [kWh/d]",
             "Aeration energy over training"),
            (axes[1], "eval_S_O5_mean", "mean S$_{O,5}$ [g/m$^3$]",
             "Where the agent puts dissolved oxygen")):
        seen = set()
        for tag, history in curves.items():
            weather, method, seed = tag.split("|")
            if weather != "dry" or method not in ORDER or seed != "s0":
                continue
            xs = [h["episode"] for h in history if key in h]
            ys = [h[key] for h in history if key in h]
            if not xs:
                continue
            colour = style.color_for(method)
            ax.plot(xs, ys, color=colour, marker="o", ms=3,
                    label=method if method not in seen else None)
            _label_last(ax, xs, ys, method, colour)
            seen.add(method)
        ax.set_xlabel("training week (pass over the learning week)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig2_learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_learning_curves.png")


def fig_tradeoff(rows: list[dict], out: Path) -> None:
    """Energy against effluent quality. Direct labels, no colour-only identity."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, weather in zip(axes, ("dry", "rain", "storm")):
        for method in ORDER:
            sel = [r for r in rows if r["weather"] == weather
                   and r["method"] == method]
            if not sel:
                continue
            x = float(np.mean([r["AE"] for r in sel]))
            y = float(np.mean([r["EQ"] for r in sel]))
            colour = style.color_for(method)
            ax.scatter([x], [y], s=70, color=colour, zorder=3,
                       edgecolor=style.SURFACE, linewidth=1.5)
            ax.annotate(method, xy=(x, y), xytext=(6, 4),
                        textcoords="offset points", fontsize=8,
                        color=style.INK_SECONDARY)
        ax.set_title(f"{weather} weather")
        ax.set_xlabel("aeration energy [kWh/d]  (lower is better)")
        ax.set_ylabel("effluent quality index  (lower is better)")
    fig.suptitle("Energy against effluent quality", fontsize=11,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig3_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_tradeoff.png")


def fig_trajectories(traces: dict, out: Path, weather: str = "storm") -> None:
    wanted = [m for m in ("PID", "DDPG-B", "PANDA-RL")
              if f"{weather}|{m}|s0" in traces]
    if not wanted:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)

    ax = axes[0]
    tag = f"{weather}|{wanted[0]}|s0"
    t = np.array(traces[tag]["t"])
    ax.plot(t, np.array(traces[tag]["q_in"]) / 1000.0,
            color=style.INK_SECONDARY, lw=1.2)
    ax.set_ylabel("influent\n[10$^3$ m$^3$/d]")
    ax.set_title(f"{weather.capitalize()} weather, evaluation week")

    for ax, key, ylabel in ((axes[1], "S_O5", "S$_{O,5}$ [g/m$^3$]"),
                            (axes[2], "KLa5", "KLa$_5$ [1/d]")):
        for method in wanted:
            tr = traces[f"{weather}|{method}|s0"]
            colour = style.color_for(method)
            ax.plot(tr["t"], tr[key], color=colour, lw=1.3, label=method)
            _label_last(ax, tr["t"], tr[key], method, colour)
        ax.set_ylabel(ylabel)
    axes[1].legend(loc="upper right", ncols=3)
    axes[2].set_xlabel("time [d]")
    fig.tight_layout()
    fig.savefig(out / f"fig4_trajectories_{weather}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4_trajectories_{weather}.png")


def fig_ablation(rows: list[dict], out: Path) -> None:
    names = ["PANDA-RL", "PANDA-noForecast", "PANDA-noCVaR",
             "PANDA-noConstraint"]
    present = [n for n in names if any(r["method"] == n for r in rows)]
    if len(present) < 2:
        return
    base = {r["method"]: r for r in rows if r["weather"] == "dry"
            and r["method"] == "PID"}
    pid_ae = float(base["PID"]["AE"]) if base else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, key, ylabel, title in (
            (axes[0], "AE", "aeration energy saved vs PID [%]",
             "Energy saving"),
            (axes[1], "viol_NH", "time above 4 g N/m$^3$ [%]",
             "Ammonium limit")):
        values, labels, colours = [], [], []
        for name in present:
            sel = [r for r in rows if r["method"] == name]
            v = float(np.mean([r[key] for r in sel]))
            if key == "AE":
                v = 100.0 * (pid_ae - v) / pid_ae
            else:
                v = 100.0 * v
            values.append(v)
            labels.append(name.replace("PANDA-", ""))
            colours.append(style.color_for("PANDA-RL") if name == "PANDA-RL"
                           else style.INK_MUTED)
        bars = ax.bar(labels, values, color=colours, width=0.6)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8,
                        color=style.INK_SECONDARY)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
    if any(r["method"] == "PANDA-RL" for r in rows):
        axes[1].axhline(5.0, color=style.REFERENCE, ls="--", lw=1)
        axes[1].annotate("budget 5%", xy=(0, 5.0), xytext=(0, 4),
                         textcoords="offset points", fontsize=8,
                         color=style.INK_MUTED)
    fig.suptitle("Which part of PANDA-RL does the work", fontsize=11,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig5_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_ablation.png")




def fig_frontier(frontier: list[dict], rows: list[dict], out: Path) -> None:
    """Aeration energy against the ammonium violation rate.

    One measure per axis -- no second y-scale.  The PANDA-RL points are a
    frontier traced by varying the budget; PID and DDPG-B are single
    operating points on the same axes.  Everything is directly labelled
    because three of the series colours sit below 3:1 on this surface.
    """
    fr = sorted(frontier, key=lambda r: r["viol_NH"])
    if not fr:
        return

    def dominated(r):
        return any(o["viol_NH"] <= r["viol_NH"] and o["AE"] <= r["AE"]
                   and (o["viol_NH"] < r["viol_NH"] or o["AE"] < r["AE"])
                   for o in fr if o is not r)

    # Only the Pareto-efficient runs are joined up.  Joining every point
    # would draw a frontier that the data does not support.
    front = [r for r in fr if not dominated(r)]
    bad = [r for r in fr if dominated(r)]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    colour = style.color_for("PANDA-RL")
    ax.plot([100.0 * r["viol_NH"] for r in front], [r["AE"] for r in front],
            color=colour, lw=1.8, marker="o", ms=6, zorder=3,
            markeredgecolor=style.SURFACE, markeredgewidth=1.5,
            label="PANDA-RL (Pareto-efficient runs)")
    if bad:
        ax.scatter([100.0 * r["viol_NH"] for r in bad], [r["AE"] for r in bad],
                   s=52, facecolor="none", edgecolor=colour, linewidth=1.5,
                   zorder=3, label="dominated run (not converged)")
    for r in fr:
        ax.annotate(f"budget {100 * r['nh_budget']:.0f}%",
                    xy=(100.0 * r["viol_NH"], r["AE"]), xytext=(6, -11),
                    textcoords="offset points", fontsize=7.5,
                    color=style.INK_MUTED)

    for method, marker in (("PID", "s"), ("DDPG-B", "D")):
        sel = [r for r in rows if r["weather"] == "dry"
               and r["method"] == method]
        if not sel:
            continue
        mx = 100.0 * float(np.mean([r["viol_NH"] for r in sel]))
        my = float(np.mean([r["AE"] for r in sel]))
        c = style.color_for(method)
        ax.scatter([mx], [my], s=95, marker=marker, color=c, zorder=4,
                   edgecolor=style.SURFACE, linewidth=1.5, label=method)
        ax.annotate(method, xy=(mx, my), xytext=(8, 5),
                    textcoords="offset points", fontsize=9,
                    color=style.INK_SECONDARY, fontweight="bold")

    ax.set_xlabel("time above the 4 g N/m$^3$ ammonium limit [% of week]")
    ax.set_ylabel("aeration energy [kWh/d]")
    ax.set_title("What the aeration saving actually costs")
    ax.legend(loc="lower left", fontsize=8)
    fig.text(0.01, -0.09,
             "Down is cheaper, left is cleaner; a method beats another only "
             "if it sits down-and-left of it.\nThe budget labels are what was "
             "asked for, the position is what was achieved.",
             fontsize=8, color=style.INK_MUTED, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(out / "fig6_frontier.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_frontier.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--tag", default="aeration")
    ap.add_argument("--frontier-tag", default="frontier")
    ap.add_argument("--dose-response", action="store_true",
                    help="also regenerate the Figure-4 sweep (slow)")
    args = ap.parse_args()

    style.apply()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("writing figures to", out)

    if args.dose_response:
        fig_dose_response(out)

    def load(name):
        path = res / f"{name}_{args.tag}.json"
        return json.loads(path.read_text()) if path.exists() else None

    rows, traces, curves = load("summary"), load("traces"), load("curves")
    fpath = res / f"summary_{args.frontier_tag}.json"
    frontier = json.loads(fpath.read_text()) if fpath.exists() else None
    if rows:
        fig_tradeoff(rows, out)
        fig_ablation(rows, out)
        if frontier:
            fig_frontier(frontier, rows, out)
    if curves:
        fig_learning_curves(curves, out)
    if traces:
        for weather in ("storm", "dry"):
            fig_trajectories(traces, out, weather)
    if not any((rows, traces, curves)):
        print("  no benchmark results yet -- run 06_aeration_benchmark.py")


if __name__ == "__main__":
    main()
