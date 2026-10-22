"""Trajectories for the results video (results/video_data.npz).

Paper 1, official plant, dry: PID and MAACC exactly as printed (seed 0),
including the learned correction ``a``.  Paper 1, our rebuilt plant, dry:
PID and our modified MAACC (seed 0).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwtp.baselines.pid import IncrementalPID  # noqa: E402
from wwtp.baselines.pid_controller import PIDController  # noqa: E402
from wwtp.bsm1 import influent  # noqa: E402
from wwtp.envs import run_closed_loop  # noqa: E402
from wwtp.maacc.agent import MAACCController, tuned_agent_configs  # noqa: E402
from wwtp.maacc.exact import ExactMAACC, U_MAX, U_MIN  # noqa: E402
from wwtp.official.plant import OfficialBSM1  # noqa: E402

spec = importlib.util.spec_from_file_location("x", ROOT / "experiments" / "20_exact_paper1.py")
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)
R = np.array([1.0, 2.0])


def official(ctrl):
    st, u_end = x.warm_state()
    p = OfficialBSM1("dry")
    p.set_state(st)
    if ctrl == "PID":
        pid = IncrementalPID(u_min=U_MIN, u_max=U_MAX, u0=u_end.copy())
        law, m = (lambda s: pid(R - s)), None
    else:
        m = ExactMAACC(seed=0, critic="paper", u0=u_end.copy())
        law = lambda s: m(s, R)
    s, S, A, Q = p.measurement(), [], [], []
    for _ in range(p.n - 1):
        s = p.step(np.clip(law(s), U_MIN, U_MAX))
        S.append(s)
        A.append([ag.a for ag in m.agents] if m else [0.0, 0.0])
        Q.append(p.sim.ys_eff[14] if hasattr(p.sim, "ys_eff") else 0.0)
    return np.array(S), np.array(A)


if __name__ == "__main__":
    s_pid, _ = official("PID")
    s_ex, a_ex = official("MAACC")
    ser = influent.canonical_scenario("dry")
    rp = run_closed_loop(PIDController(), ser, seed=100)
    rm = run_closed_loop(MAACCController(configs=tuned_agent_configs(), seed=0), ser, seed=100)
    np.savez_compressed(ROOT / "results" / "video_data.npz",
                        off_pid=s_pid, off_exact=s_ex, off_exact_a=a_ex,
                        reb_t=rp.time_days, reb_pid=rp.state, reb_mod=rm.state)
    print("wrote results/video_data.npz", s_pid.shape, s_ex.shape, rp.state.shape,
          "max |a| as printed:", np.abs(a_ex).max(axis=0))
