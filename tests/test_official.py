"""Official BSM1 plant (bsm2-python) and the as-printed MAACC."""

import numpy as np
import pytest

pytest.importorskip("bsm2_python")

from wwtp.maacc.exact import ExactMAACC, bipolar  # noqa: E402
from wwtp.official.plant import (OfficialBSM1, OfficialPlantAdapter,  # noqa: E402
                                 load_influent, official_series)


def test_rain_file_is_repaired():
    d, r = load_influent("dry"), load_influent("rain")
    assert r[:, 15].min() > 1000.0                    # flows in m3/d, not thousands
    assert np.allclose(np.diff(r[:, 0]), 15.0 / 1440.0)
    assert np.allclose(r[:672, 15], d[:672, 15])      # rain week 1 = dry week 1


def test_adapter_reproduces_the_package_step():
    a, b, ser = OfficialPlantAdapter(), OfficialBSM1("dry"), official_series("dry")
    for k in range(200):
        u = np.array([100.0 + 20.0 * np.sin(k / 30.0), 40000.0])
        a.step(45.0, ser.composition[k], ser.flow[k], u)
        b.step(u)
    assert np.allclose(a.z[4], b.sim.y_out5[:13]) and np.allclose(a.z[1], b.sim.y_out2[:13])


def test_state_transfer_restarts_identically():
    p = OfficialBSM1("dry")
    for _ in range(100):
        p.step(np.array([120.0, 50000.0]))
    st = p.get_state()
    q = OfficialBSM1("dry")
    q.set_state(st)
    assert np.allclose(q.measurement(), p.measurement())


def test_exact_maacc_starts_as_the_pid_prior():
    """AN outer weights start at zero (paper Sec. IV), so u_0 = b_0."""
    m = ExactMAACC(seed=0)
    s, r = np.array([1.1, 1.9]), np.array([1.0, 2.0])
    u = m(s, r)
    from wwtp.baselines.pid import IncrementalPID, PAPER_KD, PAPER_KI, PAPER_KP
    pid = IncrementalPID(PAPER_KP, PAPER_KI, PAPER_KD, np.zeros(2),
                         np.array([240.0, 92230.0]), u0=np.array([84.0, 55338.0]))
    assert np.allclose(u, pid(r - s))


def test_bipolar_is_the_papers_activation():
    x = np.linspace(-5, 5, 11)
    assert np.allclose(bipolar(x), (1 - np.exp(-x)) / (1 + np.exp(-x)))
