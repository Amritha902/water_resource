"""Our figures in the layout of the base paper (Figs. 4-6, 9), black and white, from stored traces."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/home/user/water_resource"
OUT = sys.argv[1]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": "white"})
T = json.load(open(f"{ROOT}/results/traces_paper1_study.json"))
def tr(k):
    s, u = np.array(T[k]["s"]), np.array(T[k]["u"]); return np.arange(len(s)) * 14 / len(s), s, u
for w in ("dry", "rain", "storm"):
    t, sp, _ = tr(f"PID|{w}|clean"); _, sm, _ = tr(f"MAACC|{w}|clean"); _, so, _ = tr(f"PIDtuned+ILC|{w}|clean")
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
    for c, (ref, lo, hi, lab) in enumerate(((2, 1.8, 2.1, r"$S_{O,5}$ (mg/L)"), (1, 0.4, 1.3, r"$S_{NO,2}$ (mg/L)"))):
        ax = axs[c]
        ax.plot(t, sp[:, c], color="#9a9a9a", lw=0.6, label="PID prior / MAACC")
        ax.plot(t, so[:, c], color="black", lw=0.6, label="Proposed")
        ax.axhline(ref, color="black", lw=0.6, ls=(0, (4, 3)), label="Set point")
        ax.set_ylim(lo, hi); ax.set_ylabel(lab)
    axs[0].legend(loc="lower right", fontsize=7, ncol=3)
    axs[1].set_xlabel("Time (d)"); axs[1].set_xlim(0, 14); axs[1].set_xticks(range(0, 15, 2))
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/o_traj_{w}.png", dpi=300); plt.close(fig)
    print(w, "MAACC == PID max |diff|", float(np.abs(sm - sp).max()))
t, _, ui = tr("PIDtuned+ILC|dry|clean"); _, _, un = tr("PIDtuned+ILC|dry|noise")
fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
for c, (lab, sc, lim) in enumerate(((r"$K_La_{,5}$ (d$^{-1}$)", 1, 300), (r"$Q_a$ ($10^4$ m$^3$/d)", 1e4, 10))):
    axs[c].plot(t, un[:, c] / sc, color="#9a9a9a", lw=0.4, label="Input with noise")
    axs[c].plot(t, ui[:, c] / sc, color="black", lw=0.7, label="Ideal input")
    axs[c].set_ylabel(lab); axs[c].set_ylim(0, lim)
axs[0].legend(loc="upper right", fontsize=7, ncol=2)
axs[1].set_xlabel("Time (d)"); axs[1].set_xlim(0, 14); axs[1].set_xticks(range(0, 15, 2))
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/o_inputs_noise.png", dpi=300); plt.close(fig)
