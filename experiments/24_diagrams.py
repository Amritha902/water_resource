"""Redraw the diagram figures of the three base papers (docs/figures/reproduced/).

P1 Fig. 1, 2; P2 Fig. 1, 2, 3, 5; P3 Fig. 1, 2.  Drawn from the papers'
descriptions; same content, our own drawing.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "reproduced"
INK, MUTED = "#1f1f1e", "#6b6a64"
ANOX, AER, CTRL, NET, SET = "#d8e6f6", "#cdeee0", "#fbe3d6", "#e7e1f5", "#eeeeea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", color=INK)


def arrow(ax, p, q, text=None, off=(0, 0.12), ls="-", color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                 color=color, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text,
                ha="center", va="center", fontsize=8, color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 2)
    ax.set_ylim(0, h * 2)
    ax.axis("off")
    return fig, ax


def bsm1(ax, controllers=()):
    """Five reactors, settler and recycles on an 18.6 x 7.2 canvas."""
    y0, h, w = 2.4, 1.4, 1.8
    xs = [2.0 + 2.4 * i for i in range(5)]
    for i, x in enumerate(xs):
        box(ax, x, y0, w, h, f"Unit {i + 1}", ANOX if i < 2 else AER, fs=10)
    ax.text(xs[0] + w / 2, y0 + h + 0.3, "Anoxic section", ha="center", fontsize=8.5, color=MUTED)
    ax.text(xs[3] + w / 2, y0 + h + 0.3, "Aerated section", ha="center", fontsize=8.5, color=MUTED)
    ym = y0 + h / 2
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, ym), (b, ym))
    arrow(ax, (0.2, ym), (xs[0], ym))
    ax.text(0.2, ym + 0.3, "Influent", fontsize=9)
    sx = 14.6
    ax.add_patch(Polygon([[sx, 3.9], [sx + 2.4, 3.9], [sx + 1.5, 1.6], [sx + 0.9, 1.6]],
                         closed=True, fc=SET, ec=INK, lw=1))
    ax.text(sx + 1.2, 3.35, "Settler", ha="center", fontsize=9)
    ax.text(sx + 1.2, 2.9, "(10 layers)", ha="center", fontsize=7.5, color=MUTED)
    arrow(ax, (xs[-1] + w, ym), (sx + 0.25, ym))
    arrow(ax, (sx + 2.3, 3.75), (18.4, 3.75))
    ax.text(17.3, 4.0, "Effluent", fontsize=9)
    ax.plot([sx + 1.2, sx + 1.2], [1.6, 0.9], color=INK, lw=1.1)
    arrow(ax, (sx + 1.2, 0.9), (18.4, 0.9))
    ax.text(16.9, 0.45, "Wastage Q_w", fontsize=8.5)
    ax.plot([sx + 1.2, xs[0] + 1.2], [0.9, 0.9], color=INK, lw=1.1)
    arrow(ax, (xs[0] + 1.2, 0.9), (xs[0] + 1.2, y0))
    ax.text(8.6, 0.45, "External (sludge) recycle Q_r", ha="center", fontsize=8.5)
    ax.plot([xs[-1] + 0.9, xs[-1] + 0.9, xs[0] + 0.5], [y0, 1.7, 1.7], color=INK, lw=1.1)
    arrow(ax, (xs[0] + 0.5, 1.7), (xs[0] + 0.5, y0))
    ax.text(8.6, 1.95, "Internal (nitrate) recycle Q_a", ha="center", fontsize=8.5)
    if "KLa5" in controllers:
        x = xs[-1]
        box(ax, x - 0.3, 5.5, 2.4, 0.8, "DO controller", CTRL, fs=9)
        arrow(ax, (x + 1.5, y0 + h), (x + 1.5, 5.5))
        ax.text(x + 1.6, 4.6, "S_O,5", fontsize=8.5)
        arrow(ax, (x + 0.3, 5.5), (x + 0.3, y0 + h))
        ax.text(x - 0.75, 4.6, "KLa,5", fontsize=8.5)
    if "Qa" in controllers:
        x = xs[1]
        box(ax, x - 0.6, 5.5, 3.0, 0.8, "Nitrate controller", CTRL, fs=9)
        arrow(ax, (x + 1.5, y0 + h), (x + 1.5, 5.5))
        ax.text(x + 1.6, 4.6, "S_NO,2", fontsize=8.5)
        ax.plot([x - 0.6, 1.2, 1.2], [5.9, 5.9, 1.7], color=INK, lw=1, ls="--")
        arrow(ax, (1.2, 1.7), (xs[0] + 0.5, 1.7), ls="--")
        ax.text(0.55, 4.4, "Q_a", fontsize=8.5)
    return xs


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def p1_fig1():
    fig, ax = canvas(9.3, 3.6)
    bsm1(ax, controllers=("KLa5", "Qa"))
    save(fig, "p1_fig1.png")


def p1_fig2():
    fig, ax = canvas(10.5, 5.0)                       # 21 x 10 units
    for row, (i, u) in enumerate(((1, "KLa,5"), (2, "Q_a"))):
        y = 6.0 - row * 4.6
        ax.add_patch(FancyBboxPatch((0.3, y - 1.4), 14.4, 4.0, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    fc="none", ec=MUTED, lw=0.8, ls="--"))
        ax.text(0.6, y + 2.25, f"Agent {i}", fontsize=10, weight="bold")
        box(ax, 0.8, y + 0.5, 3.4, 1.2, f"Action network\nΔa_{i} = π_{i}(z_{i})", NET)
        box(ax, 0.8, y - 1.0, 3.4, 1.2, f"Critic network\nQ_{i}(z, Δa_{i}, a_{i})", NET)
        box(ax, 5.3, y + 0.5, 3.4, 1.2, f"a_{i} ← a_{i} + Δa_{i}", SET)
        box(ax, 5.3, y - 1.0, 3.4, 1.2, f"Prior strategy {i}\n(incremental PID) b_{i}", CTRL, fs=8.5)
        ax.add_patch(plt.Circle((10.4, y + 0.35), 0.35, fc="white", ec=INK))
        ax.text(10.4, y + 0.35, "+", ha="center", va="center", fontsize=13)
        arrow(ax, (4.2, y + 1.1), (5.3, y + 1.1))
        arrow(ax, (8.7, y + 1.1), (10.15, y + 0.6), f"a_{i}", off=(0, 0.35))
        arrow(ax, (8.7, y - 0.4), (10.15, y + 0.1), f"b_{i}", off=(0, -0.35))
        arrow(ax, (2.5, y + 0.2), (2.5, y + 0.5), None)
        ax.text(2.7, y + 0.28, "∂Q/∂a", fontsize=8, color=MUTED)
        arrow(ax, (10.75, y + 0.35), (16.0, y + 0.35), f"u_{i} = {u}", off=(-1.2, 0.3))
    box(ax, 16.0, 1.4, 2.6, 6.2, "WWTP\n(BSM1)\n\ntwo coupled\nsubsystems", AER, fs=10)
    arrow(ax, (18.6, 4.5), (20.4, 4.5))
    ax.text(18.8, 4.9, "z = s − r", fontsize=9)
    ax.text(18.8, 3.0, "to both\nagents", fontsize=8, color=MUTED)
    save(fig, "p1_fig2.png")


def p2_fig1():
    fig, ax = canvas(9.3, 3.6)
    bsm1(ax, controllers=("KLa5",))
    save(fig, "p2_fig1.png")


def p2_fig2():
    fig, ax = canvas(3.2, 2.4)
    box(ax, 1.6, 3.2, 3.2, 1.2, "Agent\nPolicy  ·  RL algorithm", NET, fs=9)
    box(ax, 1.6, 0.4, 3.2, 1.0, "WWTP environment", AER, fs=9)
    arrow(ax, (4.8, 3.8), (5.8, 3.8))
    arrow(ax, (5.8, 3.8), (5.8, 0.9))
    arrow(ax, (5.8, 0.9), (4.8, 0.9), "Action", off=(0.5, 0.2))
    arrow(ax, (1.6, 0.9), (0.6, 0.9))
    arrow(ax, (0.6, 0.9), (0.6, 3.8), "State", off=(-0.35, 0))
    arrow(ax, (0.6, 3.8), (1.6, 3.8))
    arrow(ax, (3.2, 1.4), (3.2, 3.2), "Reward", off=(0.45, 0))
    save(fig, "p2_fig2.png")


def p2_fig3():
    fig, ax = canvas(10.0, 4.4)                       # 20 x 8.8 units
    box(ax, 0.3, 3.6, 3.2, 1.6, "WWTP\nenvironment\n(BSM1)", AER, fs=10)
    box(ax, 5.2, 6.0, 3.8, 1.5, "Actor μ(s | θ)\n+ OU exploration noise", NET)
    box(ax, 5.2, 1.4, 3.8, 1.5, "Critic Q(s, a | w)", NET)
    box(ax, 11.0, 6.0, 4.2, 1.5, "Target actor\nμ'(s | θ')", SET)
    box(ax, 11.0, 1.4, 4.2, 1.5, "Target critic\nQ'(s, a | w')", SET)
    box(ax, 16.2, 3.6, 3.4, 1.6, "Replay buffer\n(s, a, r, s')", CTRL)
    arrow(ax, (3.5, 4.9), (5.2, 6.5)); ax.text(3.7, 6.0, "state s", fontsize=8.5)
    arrow(ax, (5.2, 7.0), (1.9, 5.2), rad=0.25); ax.text(2.0, 7.0, "action a = ΔKLa,5", fontsize=8.5)
    arrow(ax, (3.5, 4.1), (16.2, 4.1)); ax.text(8.0, 4.35, "store transition (s, a, r, s')", fontsize=8.5)
    arrow(ax, (17.9, 3.6), (17.9, 0.6)); ax.plot([17.9, 7.1], [0.6, 0.6], color=INK, lw=1.1)
    arrow(ax, (7.1, 0.6), (7.1, 1.4)); ax.text(13.0, 0.15, "mini-batch, N = 256", fontsize=8.5)
    arrow(ax, (7.1, 2.9), (7.1, 6.0)); ax.text(7.3, 5.0, "∇_a Q", fontsize=8.5)
    arrow(ax, (9.0, 6.75), (11.0, 6.75), ls="--"); ax.text(9.3, 7.05, "soft update τ", fontsize=8)
    arrow(ax, (9.0, 1.8), (11.0, 1.8), ls="--"); ax.text(9.25, 1.15, "soft update τ", fontsize=8)
    arrow(ax, (13.1, 6.0), (13.1, 2.9)); ax.text(13.3, 5.0, "a' = μ'(s')", fontsize=8.5)
    arrow(ax, (11.0, 2.6), (9.0, 2.6)); ax.text(9.15, 2.85, "TD target y = r + γQ'", fontsize=7.5)
    save(fig, "p2_fig3.png")


def p2_fig5():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.axis("off")
    layers = [3, 7, 7, 1]
    xs = [0, 1.4, 2.8, 4.2]
    cols = ["#9fd18b", "#7fb3e6", "#7fb3e6", "#eb8a5a"]
    pos = []
    for n, x, c in zip(layers, xs, cols):
        ys = [(i - (n - 1) / 2) * 0.42 for i in range(n)]
        pos.append([(x, y) for y in ys])
        for y in ys:
            ax.add_patch(plt.Circle((x, y), 0.14, fc=c, ec=INK, lw=0.8, zorder=3))
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#888", lw=0.35, zorder=1)
    for x, t in zip(xs, ["Input layer\n(3)", "Hidden layer 1\n(256, ReLU)", "Hidden layer 2\n(256, ReLU)", "Output layer\n(1, tanh)"]):
        ax.text(x, 1.75, t, ha="center", fontsize=7.5)
    ax.set_xlim(-0.6, 4.8)
    ax.set_ylim(-1.7, 2.1)
    save(fig, "p2_fig5.png")


def p3_fig1():
    fig, ax = canvas(9.3, 3.6)
    bsm1(ax, controllers=("KLa5",))
    save(fig, "p3_fig1.png")


def p3_fig2():
    fig, ax = canvas(10.0, 5.0)                        # 19 x 10 units
    box(ax, 0.8, 7.4, 3.6, 1.4, "Prior policy α(e_k)\nPID (direct or\nindirect method)", CTRL)
    box(ax, 0.8, 4.4, 3.6, 1.4, "Action network\nβ_k = Aᵀ Θ(Sᵀ e_k)", NET)
    ax.add_patch(plt.Circle((6.0, 6.6), 0.4, fc="white", ec=INK))
    ax.text(6.0, 6.6, "+", ha="center", va="center", fontsize=13)
    arrow(ax, (4.4, 8.1), (5.75, 6.9)); ax.text(4.8, 7.8, "× ω", fontsize=9)
    arrow(ax, (4.4, 5.1), (5.75, 6.3))
    box(ax, 8.6, 5.9, 4.0, 1.4, "WWTP (BSM1)\nx_{k+1} = F(x_k, u_k)", AER)
    arrow(ax, (6.4, 6.6), (8.6, 6.6)); ax.text(6.3, 7.4, "u_k = ωα_k + β_k", fontsize=8.5)
    arrow(ax, (12.6, 6.6), (15.4, 6.6)); ax.text(12.9, 6.9, "e_k = x_k − r_k", fontsize=8.5)
    ax.plot([15.4, 15.4, 0.2, 0.2], [6.6, 9.6, 9.6, 5.1], color=INK, lw=1)
    arrow(ax, (0.2, 8.1), (0.8, 8.1)); arrow(ax, (0.2, 5.1), (0.8, 5.1))
    box(ax, 8.6, 2.6, 4.6, 1.4, "Critic network\nQ_k = Vᵀ Ψ(Wᵀ[e_k, u_k])", NET)
    arrow(ax, (15.4, 6.6), (12.6, 3.5), rad=0.0)
    arrow(ax, (6.0, 6.2), (8.6, 3.3)); ax.text(6.6, 4.4, "u_k", fontsize=8.5)
    box(ax, 8.6, 0.2, 4.6, 1.4, "e_c = γQ_k + G_k − Q_{k−1}\nupdate V  (T_c = 50)", SET, fs=8.5)
    arrow(ax, (10.8, 2.6), (10.8, 1.6))
    arrow(ax, (8.6, 3.0), (2.6, 4.4), rad=-0.15)
    ax.text(3.4, 2.7, "∇_β Q: update A  (T_a = 60)", fontsize=8.5)
    save(fig, "p3_fig2.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (p1_fig1, p1_fig2, p2_fig1, p2_fig2, p2_fig3, p2_fig5, p3_fig1, p3_fig2):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
