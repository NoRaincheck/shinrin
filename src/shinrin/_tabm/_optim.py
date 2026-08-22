"""Optimizers for the vendored TabM model (NumPy reference implementations).

All optimizers operate on a single flat float32 parameter vector so the
same interface serves both this module and the Mojo kernels.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _ParamsLike(Protocol):
    """Structural view of TabM/MLP parameter containers."""

    def names(self) -> list[str]: ...

    @property
    def arrays(self) -> dict[str, np.ndarray]: ...


class FlatSpace:
    """Maps between :class:`TabMParams`-style dictionaries and a flat vector."""

    def __init__(self, params: _ParamsLike) -> None:
        self.names = params.names()
        self.shapes = [params.arrays[n].shape for n in self.names]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total = int(sum(self.sizes))

    def flatten(self, params: _ParamsLike) -> np.ndarray:
        return np.concatenate([params.arrays[n].ravel() for n in self.names])

    def flatten_grads(self, grads: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([grads[n].ravel() for n in self.names])

    def scatter(self, theta: np.ndarray, params: _ParamsLike) -> None:
        """Copy slices of ``theta`` into the existing parameter arrays."""
        for name, start, size, shape in zip(
            self.names, self.offsets, self.sizes, self.shapes
        ):
            params.arrays[name] = theta[start : start + size].reshape(shape)


class AdamState:
    """Moment buffers for the Adam optimizer."""

    def __init__(self, total: int) -> None:
        self.m = np.zeros(total, dtype=np.float32)
        self.v = np.zeros(total, dtype=np.float32)
        self.t = 0


def adam_step(
    theta: np.ndarray,
    grad: np.ndarray,
    state: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """In-place Adam update on ``theta``."""
    state.t += 1
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)
    m_hat = state.m / np.float32(1.0 - beta1**state.t)
    v_hat = state.v / np.float32(1.0 - beta2**state.t)
    theta -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)


def sgd_step(
    theta: np.ndarray,
    grad: np.ndarray,
    velocity: np.ndarray | None,
    lr: float,
    momentum: float = 0.0,
) -> np.ndarray | None:
    """In-place SGD (optionally with momentum) update on ``theta``."""
    if momentum <= 0.0:
        theta -= (lr * grad).astype(np.float32)
        return velocity
    if velocity is None:
        velocity = np.zeros_like(theta)
    velocity[:] = momentum * velocity + grad
    theta -= (lr * velocity).astype(np.float32)
    return velocity


def lbfgs_minimize(
    fn,
    theta: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    history_size: int = 10,
    max_line_search: int = 20,
    c1: float = 1e-4,
) -> tuple[np.ndarray, int, list[float]]:
    """Minimize ``fn(theta) -> (loss, grad)`` with L-BFGS (scipy backend).

    Delegates to ``scipy.optimize.minimize`` (L-BFGS-B) for robustness on
    ill-conditioned nonconvex problems. Returns ``(theta, iterations,
    losses)``; ``theta`` is updated in place and also returned.
    """
    from scipy.optimize import minimize

    theta = np.ascontiguousarray(theta, dtype=np.float32)
    cache_key: np.ndarray | None = None
    cache_loss = 0.0

    def value_and_grad(t: np.ndarray):
        nonlocal cache_key, cache_loss
        loss, grad = fn(t)
        cache_key = t.copy()
        cache_loss = float(loss)
        return float(loss), grad.astype(np.float64)

    losses: list[float] = []

    def callback(t: np.ndarray) -> None:
        key = cache_key
        if key is not None and key.shape == t.shape and bool(np.array_equal(key, t)):
            losses.append(cache_loss)
        else:
            loss, _ = value_and_grad(t)
            losses.append(loss)

    result = minimize(
        value_and_grad,
        theta,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": max_iter,
            "maxcor": history_size,
            "gtol": tol,
            "ftol": 1e-12,
            "maxls": max_line_search,
        },
    )
    out = np.asarray(result.x, dtype=np.float32)
    theta[:] = out
    if not losses:
        losses.append(float(result.fun))
    return theta, int(result.nit), losses
