"""Paper 2, Fig. 4 on the official plant: effluent vs DO set point.

Constant influent = flow-weighted mean of the official dry week; DO held at
each set point by paper 2's PID on KLa_5 (Q_a at the default) for 40 days;
effluent averaged over the last 5 days.  Output: results/summary_fig4_official.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bsm2_python.bsm1_base import BSM1Base  # noqa: E402

from wwtp.bsm1 import asm1  # noqa: E402
from wwtp.official.plant import DT_DAYS, load_influent  # noqa: E402
from wwtp.rl.baselines import PAPER2_KD, PAPER2_KI, PAPER2_KP  # noqa: E402

DAYS, AVG = 40.0, 5.0
VOL = np.array([1000, 1000, 1333, 1333, 1333.0])

d = load_influent("dry")
q = d[:, 15]
row = (d[:, 1:15] * q[:, None]).sum(0) / q.sum()
const = np.zeros((int(DAYS * 96) + 1, 22))
const[:, 0] = np.arange(len(const)) / 96.0
const[:, 1:15] = row
const[:, 15] = q.mean()
const[:, 16] = 15.0

rows = []
for sp in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    sim = BSM1Base(data_in=const, timestep=DT_DAYS, endtime=DAYS)
    kla, e1, e2 = 84.0, 0.0, 0.0
    acc = []
    n = len(sim.simtime) - 1
    for k in range(n):
        e = sp - sim.y_out5[7]
        du = PAPER2_KP * (e - e1) + PAPER2_KI * e + PAPER2_KD * (e - 2 * e1 + e2)
        e2, e1 = e1, e
        kla = float(np.clip(kla + np.clip(du, -6, 6), 0, 240))
        sim.klas[4] = kla
        sim.step(k)
        if k * DT_DAYS > DAYS - AVG:
            acc.append(np.r_[np.array(sim.ys_eff[:13]), kla])
    a = np.array(acc)
    z = a[:, :13].mean(0)
    r = dict(do=sp, kla=float(a[:, 13].mean()), nh=float(z[asm1.S_NH]),
             tn=float(asm1.total_nitrogen(z)), bod=float(asm1.bod5(z)),
             cod=float(asm1.total_cod(z)), tss=float(asm1.tss(z)),
             ae=float(8 / 1800 * (VOL[2] * 240 + VOL[3] * 240 + VOL[4] * a[:, 13].mean())))
    rows.append(r)
    print(r, flush=True)
(ROOT / "results" / "summary_fig4_official.json").write_text(json.dumps(rows, indent=1))
