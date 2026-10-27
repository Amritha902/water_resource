"""Redraw every result figure of the three base papers from our runs.

Output: docs/figures/reproduced/p{1,2,3}_fig{n}.png, same panels and axes as
the originals.  Inputs: results/traces_paper1_exact2.json,
results/traces_fig_p2_*.json, results/summary_fig4_official.json,
results/traces_ioql.json and the official influent files.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wwtp.official.plant import load_influent  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "reproduced"
C = {"MAACC": "#2a78d6", "PID": "#eb6834", "IOQL": "#2a78d6", "DDPG-A": "#2a78d6",
     "DDPG-B": "#d63c6b", "Fuzzy": "#1baf7a", "set": "#1f1f1e", "K1": "#2a78d6",
     "K2": "#1baf7a", "K3": "#d63c6b", "dry": "#2a78d6", "rain": "#eb6834", "storm": "#c79a00"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
                     "grid.color": "#e4e3de", "axes.spines.top": False, "axes.spines.right": False})
DT4 = 4 * 45 / 86400            # trace sampling of papers 1 and 3


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def influent_fig(name, title_units="Flow rate (m³/d)"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for w, lab in (("dry", "Dry"), ("rain", "Rainy"), ("storm", "Stormy")):
        d = load_influent(w)
        ax.plot(d[:, 0], d[:, 15], color=C[w], lw=1.0, label=lab, ls="-" if w == "dry" else "--")
    ax.set_xlabel("Time (day)")
    ax.set_ylabel(title_units)
    ax.set_xlim(0, 14)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, name)


# ----------------------------------------------------------------- paper 1
def p1_series(tr, key):
    s = np.array(tr[key]["s"])
    return np.arange(len(s)) * DT4, s


def p1_tracking(tr, cond, weather, name):
    if not all(f"{weather}|{cond}|{c}" in tr for c in ("PID", "MAACC-paper")):
        return
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
    lims = ((1.8, 2.1), (0.8, 1.3))                  # the paper's axes
    for ax, i, lab, sp, lim in zip(axes, (0, 1), ("S_O,5 (mg/L)", "S_NO,2 (mg/L)"), (2.0, 1.0), lims):
        for ctrl, ls in (("MAACC-paper", "-"), ("PID", "--")):
            t, s = p1_series(tr, f"{weather}|{cond}|{ctrl}")
            ax.plot(t, s[:, i], color=C[ctrl.split("-")[0]], lw=0.9, ls=ls,
                    label="MAACC" if "MAACC" in ctrl else "PID")
        ax.axhline(sp, color=C["set"], lw=0.9, ls=":", label="Set point")
        ax.set_ylabel(lab)
        ax.set_xlim(0, 14)
        ax.set_ylim(*lim)
    axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    axes[1].set_xlabel("Time (day)")
    save(fig, name)


def p1_inputs_weathers(tr):
    if not all(f"{w}|clean|MAACC-paper" in tr for w in ("dry", "rain", "storm")):
        return
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
    for w, lab in (("dry", "Dry"), ("rain", "Rainy"), ("storm", "Stormy")):
        u = np.array(tr[f"{w}|clean|MAACC-paper"]["u"])
        t = np.arange(len(u)) * DT4
        axes[0].plot(t, u[:, 0], color=C[w], lw=0.8, label=lab, ls="-" if w == "dry" else "--")
        axes[1].plot(t, u[:, 1], color=C[w], lw=0.8, label=lab, ls="-" if w == "dry" else "--")
    axes[0].set_ylabel("KLa,5 (1/d)")
    axes[1].set_ylabel("Q_a (m³/d)")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].set_xlabel("Time (day)")
    axes[1].set_xlim(0, 14)
    save(fig, "p1_fig7.png")


def p1_inputs_noise(tr):
    k = "dry|noise|MAACC-paper"
    if k not in tr:
        return
    u, ui = np.array(tr[k]["u"]), np.array(tr[k]["u_ideal"])
    t = np.arange(len(u)) * DT4
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
    for ax, i, lab in zip(axes, (0, 1), ("KLa,5 (1/d)", "Q_a (m³/d)")):
        ax.plot(t, ui[:, i], color=C["MAACC"], lw=0.8, label="Ideal control input")
        ax.plot(t, u[:, i], color=C["PID"], lw=0.5, alpha=0.8, label="Control input with noise")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("Time (day)")
    axes[1].set_xlim(0, 14)
    save(fig, "p1_fig9.png")


def paper1():
    influent_fig("p1_fig3.png")
    tr = load("traces_paper1_exact2.json")
    if not tr:
        return
    for w, n in (("dry", 4), ("rain", 5), ("storm", 6)):
        p1_tracking(tr, "clean", w, f"p1_fig{n}.png")
    p1_inputs_weathers(tr)
    p1_tracking(tr, "noise", "dry", "p1_fig8.png")
    p1_inputs_noise(tr)


# ----------------------------------------------------------------- paper 2
def p2_traces():
    tr = {}
    for f in sorted(RES.glob("traces_fig_p2_*.json")):
        for k, v in json.loads(f.read_text()).items():
            if v.get("t"):
                tr[k.rsplit("|", 1)[0]] = v              # weather|method
    return tr


def p2_fig4():
    rows = load("summary_fig4_official.json")
    if not rows:
        return
    do = [r["do"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2))
    for ax, key, lab, tag in zip(axes.flat, ("nh", "tn", "bod", "cod"),
                                 ("S_NH (mg/L)", "N_tot (mg/L)", "BOD (mg/L)", "COD (mg/L)"),
                                 "abcd"):
        ax.plot(do, [r[key] for r in rows], color=C["MAACC"], lw=1.5)
        ax.set_xlabel("Setpoint value (mg/L)")
        ax.set_ylabel(lab)
        ax.set_title(f"({tag})", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "p2_fig4.png")


def p2_do(ax, tr, weather):
    for m in ("DDPG-A", "DDPG-B", "PID", "Fuzzy"):
        k = f"{weather}|{m}"
        if k in tr:
            ax.plot(tr[k]["t"], tr[k]["S_O5"], color=C[m], lw=0.9, label=m,
                    ls="--" if m in ("PID", "Fuzzy") else "-")
    ax.set_xlabel("Time (d)")
    ax.set_ylabel("S_O,5 (mg/L)")
    ax.set_xlim(7, 14)


def paper2():
    p2_fig4()
    tr = p2_traces()
    if not tr:
        return
    if "rain|PID" in tr:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        p2_do(ax, tr, "rain")
        ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
        save(fig, "p2_fig6.png")
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
        for ax, key, lab in zip(axes.flat, ("S_NH", "N_tot", "BOD5", "COD"),
                                ("S_NH,e (mg/L)", "N_tot,e (mg/L)", "BOD_e (mg/L)", "COD_e (mg/L)")):
            for m in ("DDPG-A", "DDPG-B", "PID", "Fuzzy"):
                k = f"rain|{m}"
                if k in tr and key in tr[k]:
                    ax.plot(tr[k]["t"], tr[k][key], color=C[m], lw=0.9, label=m,
                            ls="--" if m in ("PID", "Fuzzy") else "-")
            if key == "S_NH":
                ax.axhline(4, color="#c0392b", lw=0.8, ls=":")
            if key == "N_tot":
                ax.axhline(18, color="#c0392b", lw=0.8, ls=":")
            ax.set_ylabel(lab)
            ax.set_xlim(7, 14)
        for ax in axes[1]:
            ax.set_xlabel("Time (d)")
        axes[0, 0].legend(frameon=False, ncol=2, fontsize=7.5)
        fig.tight_layout()
        save(fig, "p2_fig7.png")
    if "dry|PID" in tr and "storm|PID" in tr:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, w, tag in zip(axes, ("dry", "storm"), "ab"):
            p2_do(ax, tr, w)
            ax.set_title(f"({tag}) {w} weather", loc="left", fontsize=9)
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        save(fig, "p2_fig8.png")


# ----------------------------------------------------------------- paper 3
def p3_panels(tr, ref, methods, name):
    keys = [f"{w}|{ref}|PID" for w in ("dry", "rain", "storm")]
    if not all(k in tr for k in keys):
        return
    fig, axes = plt.subplots(3, 1, figsize=(6, 7.2), sharex=True)
    for ax, w, tag in zip(axes, ("dry", "rain", "storm"), "abc"):
        for m, lab in methods:
            k = f"{w}|{ref}|{m}"
            if k not in tr:
                continue
            x = np.array(tr[k]["x"])
            t = np.arange(len(x)) * DT4
            ax.plot(t, x, color=C[lab.split(" ")[-1].strip("()")] if "(" in lab else C[m.split("-")[0]],
                    lw=0.8, label=lab, ls="--" if m == "PID" else "-")
        r = np.array(tr[keys[0]]["r"])
        ax.plot(np.arange(len(r)) * DT4, r, color=C["set"], lw=0.8, ls=":", label="r_k")
        ax.set_ylabel("S_O,5 (mg/L)")
        ax.set_title(f"({tag})", loc="left", fontsize=9)
        ax.set_xlim(0, 14)
    axes[0].legend(frameon=False, ncol=len(methods) + 1, fontsize=8, loc="upper right")
    axes[2].set_xlabel("Time (day)")
    save(fig, name)


def paper3():
    influent_fig("p3_fig3.png", "Inflow (m³/d)")
    tr = load("traces_ioql.json")
    if not tr:
        return
    base = [("PID", "PID"), ("IOQL", "IOQL")]
    priors = [("PID", "PID"), ("IOQL-K1", "IOQL (K1)"), ("IOQL-K2", "IOQL (K2)"), ("IOQL-K3", "IOQL (K3)")]
    p3_panels(tr, "fixed", base, "p3_fig4.png")
    p3_panels(tr, "fixed", priors, "p3_fig5.png")
    p3_panels(tr, "dynamic", base, "p3_fig6.png")
    p3_panels(tr, "dynamic", priors, "p3_fig7.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    paper1()
    paper2()
    paper3()
