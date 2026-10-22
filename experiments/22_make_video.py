"""Results video: docs/video/results.mp4 (1280x720, 20 fps).

Scenes: title; paper 1 exactly as printed on the official plant; our modified
paper 1 on the rebuilt plant; paper 2 on the official plant (PID vs DDPG-B,
when ``results/traces_video_p2_dry.json`` exists); summary.
Data: ``experiments/21_video_data.py`` and the paper 2 runs.
"""

import json
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "video" / "results.mp4"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#6b6a64"})
PID, PRINTED, MODIFIED, DDPG = "#eb6834", "#1baf7a", "#2a78d6", "#2a78d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
FPS = 20

d = np.load(RES / "video_data.npz")
step = 20
t_off = np.arange(len(d["off_pid"])) * 45 / 86400
off_t, off_pid, off_ex, off_a = t_off[::step], d["off_pid"][::step], d["off_exact"][::step], d["off_exact_a"][::step]
reb_t, reb_pid, reb_mod = d["reb_t"][::step], d["reb_pid"][::step], d["reb_mod"][::step]

p2 = None
f = RES / "traces_video_p2_dry.json"
base = json.loads((RES / "traces_aeration_official_baselines.json").read_text())
if f.exists():
    tr = json.loads(f.read_text())
    key = next((k for k in tr if "DDPG-B" in k and tr[k]["t"]), None)
    if key:
        p2 = (base["dry|PID|s0"], tr[key], key)

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)


def card(lines, sub=None):
    fig.clf()
    fig.patch.set_facecolor("white")
    y = 0.62
    for i, l in enumerate(lines):
        fig.text(0.5, y - i * 0.09, l, ha="center", va="center", color=INK,
                 fontsize=25 if i == 0 else 19, weight="bold" if i == 0 else "normal")
    if sub:
        fig.text(0.5, 0.2, sub, ha="center", color=MUTED, fontsize=13)


def tracking(title, t, series, upto, note, xl):
    fig.clf()
    fig.suptitle(title, x=0.06, ha="left", fontsize=17, color=INK, weight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    for ax, i, lab, sp in zip(axes, (0, 1), ("S_O,5 [mg/L]", "S_NO,2 [mg N/L]"), (1.0, 2.0)):
        for name, data, col, ls in series:
            ax.plot(t[:upto], data[:upto, i], color=col, lw=1.8, ls=ls, label=name)
        ax.axhline(sp, color=MUTED, lw=0.9, ls=":")
        ax.set_ylabel(lab, fontsize=12)
        ax.set_xlim(*xl)
        ax.grid(axis="y", color=GRID)
        lo, hi = np.percentile(np.concatenate([s[2][:, i] if False else s[1][:, i] for s in series]), [0.5, 99.5])
        pad = 0.15 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].legend(frameon=False, ncol=len(series), loc="upper right", fontsize=11)
    axes[1].set_xlabel("time [days]", fontsize=12)
    fig.text(0.06, 0.015, note, fontsize=12, color=INK)
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.08, right=0.97, hspace=0.12)


def paper2(upto):
    pid, dd, key = p2
    fig.clf()
    fig.suptitle("Paper 2 · official BSM1, dry weather, evaluation week: PID vs DDPG-B (" + key.split("|")[-1] + ")",
                 x=0.06, ha="left", fontsize=17, color=INK, weight="bold")
    axes = fig.subplots(3, 1, sharex=True)
    for ax, k, lab in zip(axes, ("S_O5", "S_NH", "KLa5"), ("S_O,5 [mg/L]", "effluent S_NH [g N/m³]", "KLa_5 [1/d]")):
        for name, tr, col in (("PID (DO held at 2)", pid, PID), ("DDPG-B (oxygen floats)", dd, DDPG)):
            n = int(len(tr["t"]) * upto)
            ax.plot(tr["t"][:n], tr[k][:n], color=col, lw=1.8, label=name)
        if k == "S_NH":
            ax.axhline(4.0, color="#c0392b", lw=1, ls="--")
            ax.text(pid["t"][0], 4.15, "limit 4", color="#c0392b", fontsize=10)
        ax.set_ylabel(lab, fontsize=11)
        ax.grid(axis="y", color=GRID)
        ax.set_xlim(min(pid["t"]), max(pid["t"]))
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=11)
    axes[2].set_xlabel("time [days]", fontsize=12)
    fig.text(0.06, 0.015, "Energy: PID 3728 kWh/d · DDPG-B 3331 kWh/d mean of 3 seeds (−10.6%; paper −7.24%). "
             "Ammonia above limit 56% of the week vs 21% (PID).", fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.08, right=0.97, hspace=0.15)


scenes = []
scenes += [("card", 60, (["Base paper replication on the official BSM1 benchmark",
                          "Paper 1: MAACC (Wang et al., IEEE T-II 2026)",
                          "Paper 2: floating DO by DDPG (Du et al., Sci China 2023)"],
                         "Simulations from github.com/Amritha902/water_resource"))]
scenes += [("card", 50, (["Paper 1, exactly as printed", "Official plant · official dry influent · Table I values"], None))]
scenes += [("p1exact", 220, None)]
scenes += [("card", 50, (["Paper 1, our modified version", "Bigger steps · scaled inputs · safety fallback to PID",
                          "(our variant on our rebuilt plant, not the paper's algorithm)"], None))]
scenes += [("p1mod", 200, None)]
if p2:
    scenes += [("card", 50, (["Paper 2 on the official plant", "Oxygen is allowed to float to save aeration energy"], None))]
    scenes += [("p2", 220, None)]
scenes += [("card", 110, (["Summary",
                           "Paper 1 as printed = PID in 180 of 180 runs: not reproducible",
                           "Our modified Paper 1: oxygen error −81% (dry), 18 of 20 runs beat PID",
                           "Paper 2: PID within 1%; DDPG-B saves 10.6% (paper 7.24%), more ammonia"],
                          "Every number from saved runs · docs/REPRODUCTION.md"))]

OUT.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=FPS, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with writer.saving(fig, str(OUT), dpi=100):
    for kind, n, arg in scenes:
        for i in range(n):
            frac = min(1.0, (i + 1) / (n - 20)) if kind != "card" else 1
            if kind == "card":
                if i == 0:
                    card(*arg)
            elif kind == "p1exact":
                upto = max(2, int(len(off_t) * frac))
                tracking("Paper 1 exactly as printed · official BSM1, dry weather, 14 days",
                         off_t, [("PID (expert prior)", off_pid, PID, "-"),
                                 ("MAACC as printed", off_ex, PRINTED, "--")], upto,
                         f"The two lines overlap: the learned correction never exceeds "
                         f"{np.abs(off_a[:, 0]).max():.0e} (KLa) / {np.abs(off_a[:, 1]).max():.0e} m³/d (Qa). "
                         "MAACC = PID in all 180 runs.", (0, 14))
            elif kind == "p1mod":
                lo = np.searchsorted(reb_t, 8.0)
                hi = np.searchsorted(reb_t, 10.0)
                upto = lo + max(2, int((hi - lo) * frac))
                tracking("Our modified MAACC vs PID · rebuilt plant, dry weather, days 8–10",
                         reb_t, [("PID (expert prior)", reb_pid, PID, "-"),
                                 ("Our modified MAACC", reb_mod, MODIFIED, "-")], upto,
                         "Oxygen error −81%, nitrate −84% (median run) vs PID over 14 days; 18 of 20 runs beat PID. "
                         "Worse than PID under noise.", (8.0, 10.0))
            elif kind == "p2":
                paper2(frac)
            writer.grab_frame()
print("wrote", OUT, "paper 2 scene:", bool(p2))
