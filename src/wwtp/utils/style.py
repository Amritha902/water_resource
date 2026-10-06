"""Shared plotting style.

The three controllers are the only categorical series in this repository, so
they take the first three slots of the validated categorical palette (checked
with the data-visualisation validator on the all-pairs list, light mode:
worst CVD dE 9.2, worst normal-vision dE 24.0).  Aqua sits below 3:1 against
the light surface, so every figure that uses it also carries a legend and
direct labels -- identity is never colour alone.
"""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

SERIES = {
    "PID": "#2a78d6",          # slot 1, blue
    "MAACC": "#eb6834",        # slot 2, orange
    "PANDA": "#1baf7a",        # slot 3, aqua
}
REFERENCE = "#8a8880"          # set points and limits: neutral, not a series

#: aeration methods (paper 2 and ours).  Five slots of the validated
#: categorical palette; checked on the adjacent pairlist in light mode --
#: worst CVD dE 9.1, worst normal-vision dE 19.6.  Aqua, yellow and magenta
#: sit below 3:1 against the light surface, so every figure using them also
#: carries a legend and direct labels: identity is never colour alone.
AERATION = {
    "PID": "#2a78d6",
    "Fuzzy": "#eb6834",
    "DDPG-A": "#1baf7a",
    "DDPG-B": "#eda100",
    "PANDA-RL": "#e87ba4",
}

ABLATION = {
    "PANDA-noFF": "#eda100",
    "PANDA-noCtx": "#e87ba4",
    "PANDA-noGate": "#4a3aa7",
}


def color_for(name: str) -> str:
    for table in (SERIES, AERATION, ABLATION):
        if name in table:
            return table[name]
    return INK_SECONDARY


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "font.size": 9,
        "figure.dpi": 130,
    })

#: early-warning methods (docs/05_early_warning.md).  The analyser alarm is
#: today's practice and takes the neutral reference colour.
WARNING = {
    "Early-warning net": "#2a78d6",
    "Gradient boosting": "#eb6834",
    "Analyser trend": "#eda100",
    "Analyser alarm": "#8a8880",
}
