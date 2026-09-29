"""The critic-update correction documented in ``docs/02_reproduction.md``.

Equation (15) of the reference paper forms the approximation error
``e_k = U_k + lambda Q(x_k) - Q(x_{k-1})`` and eq. (18) descends it in the
weights that produce ``Q(x_k)`` -- it corrects the *later* prediction.  For
slowly varying states that recursion has homogeneous gain
``1 + l_c (1 - lambda) ||vartheta||^2 > 1`` and diverges.  Descending the
same Bellman residual in the TD(0) direction converges.
"""

import numpy as np
import pytest

from wwtp.maacc.networks import ShallowNet

LAMBDA = 0.9
N_STEPS = 40_000


def _synthetic_td(mode: str, lr: float = 0.02, n: int = N_STEPS) -> ShallowNet:
    """Fit Q for a utility ``U = z^2 + 0.05 a^2`` on a slowly varying signal."""
    net = ShallowNet(4, 24, 1, rng=np.random.default_rng(0), inner_scale=1.0)
    rng = np.random.default_rng(1)
    a, x_prev, q_prev = 0.0, None, None
    for _ in range(n):
        a = float(np.clip(0.995 * a + rng.normal(0.0, 0.1), -1.0, 1.0))
        z = rng.normal(0.0, 0.3, 2)
        x = np.array([z[0], z[1], 0.0, a])
        util = z[0] ** 2 + 0.05 * a ** 2
        q = float(net.forward(x)[0])
        if x_prev is not None:
            if mode == "paper":
                net.sgd_outer(np.array([(util + LAMBDA * q - q_prev) * LAMBDA]), lr)
            else:
                target = util + LAMBDA * q
                err = float(net.forward(x_prev)[0]) - target
                net.sgd_outer(np.array([err]), lr)
        q_prev = float(net.forward(x)[0])
        x_prev = x
    return net


def _probe(net: ShallowNet, a: float) -> tuple[float, float]:
    q = float(net.forward(np.array([0.0, 0.0, 0.0, a]))[0])
    return q, float(net.grad_wrt_input()[0][3])


def test_literal_equation_15_update_diverges():
    q, _ = _probe(_synthetic_td("paper"), 0.0)
    assert abs(q) > 100.0, "expected the literal update to blow up"


def test_td0_update_recovers_the_bellman_fixed_point():
    net = _synthetic_td("td0")
    # E[U] = 0.09 + 0.05 E[a^2]; Q(0) should sit near E[U]/(1 - lambda)
    q0, _ = _probe(net, 0.0)
    assert 0.3 < q0 < 3.0
    # Q must increase with the magnitude of the standing control offset,
    # otherwise the actor has no restoring gradient and drifts to its rail.
    q1, _ = _probe(net, 1.0)
    assert q1 > q0


def test_networks_carry_a_bias_so_they_are_not_pinned_at_the_origin():
    net = ShallowNet(3, 8, 1, rng=np.random.default_rng(0))
    assert net.theta.shape[0] == 4          # three inputs plus the bias
    assert float(net.forward(np.zeros(3))[0]) != 0.0


@pytest.mark.parametrize("n_in", [1, 4, 9])
def test_input_gradient_matches_finite_differences(n_in):
    net = ShallowNet(n_in, 12, 1, rng=np.random.default_rng(3))
    x = np.random.default_rng(4).normal(0.0, 0.5, n_in)
    net.forward(x)
    analytic = net.grad_wrt_input()[0]
    eps = 1e-6
    numeric = np.empty(n_in)
    for j in range(n_in):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        numeric[j] = (float(net.forward(xp)[0]) - float(net.forward(xm)[0])) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-6)
