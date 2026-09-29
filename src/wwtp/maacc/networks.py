"""Single-hidden-layer networks used by the adaptive critic agents.

The reference paper approximates the Q-function with a critic network (CN)
and the incremental policy with an action network (AN); in both cases only
the outer-layer weights are adapted online while the inner layer keeps its
random initialisation.  Both networks are plain NumPy: they are tiny and are
updated one sample at a time inside the control loop.

A constant bias input is appended to the feature vector.  Without it the
networks are odd functions that are pinned to zero at the origin, so a critic
could not represent the (strictly positive) Q-function of eq. (9) near the
set point at all.
"""

from __future__ import annotations

import numpy as np


class ShallowNet:
    """``y = beta^T tanh(theta^T [x; 1])`` with only ``beta`` trainable."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int = 1, *,
                 rng: np.random.Generator, inner_scale: float = 1.0,
                 zero_outer: bool = False, outer_scale: float = 0.05):
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.theta = rng.uniform(-inner_scale, inner_scale,
                                 size=(n_in + 1, n_hidden))
        if zero_outer:
            self.beta = np.zeros((n_hidden, n_out))
        else:
            self.beta = rng.uniform(-outer_scale, outer_scale,
                                    size=(n_hidden, n_out))
        self._h = np.zeros(n_hidden)
        self._x = np.zeros(n_in + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = np.append(np.asarray(x, dtype=float).reshape(-1), 1.0)
        self._h = np.tanh(self.theta.T @ self._x)
        return self.beta.T @ self._h

    @property
    def hidden(self) -> np.ndarray:
        return self._h

    def grad_wrt_input(self) -> np.ndarray:
        """d(output)/d(input) at the last forward pass, shape (n_out, n_in).

        The bias column is dropped, so the result lines up with the caller's
        feature vector.
        """
        d = 1.0 - self._h ** 2                                  # (H,)
        g = (self.beta * d[:, None]).T @ self.theta.T           # (n_out, n_in+1)
        return g[:, :self.n_in]

    def sgd_outer(self, grad_out: np.ndarray, lr: float,
                  clip: float = 10.0, weight_decay: float = 0.0) -> None:
        """Gradient step on ``beta`` given dLoss/d(output).

        ``weight_decay`` shrinks the outer weights towards zero.  For the
        action network that is what keeps the learned term a bounded
        *correction* to the expert prior: a saturated ``tanh`` output has a
        vanishing gradient, so without decay an early transient can latch the
        policy at its rail permanently.
        """
        g = np.outer(self._h, np.asarray(grad_out, dtype=float).reshape(-1))
        norm = float(np.linalg.norm(g))
        if norm > clip:
            g *= clip / norm
        if weight_decay:
            self.beta *= (1.0 - lr * weight_decay)
        self.beta -= lr * g
