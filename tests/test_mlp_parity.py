"""Parity tests between the NumPy MLP reference and the Mojo kernels.

The Mojo kernels implement the same forward/backward as
``shinrin._mlp._model.MLPCore`` over an identical flat parameter layout,
so loss, gradients and predictions must agree up to float32
accumulation-order noise. Optimizer trajectories are only compared
loosely (shuffle RNGs differ between backends).
"""

from __future__ import annotations

import numpy as np
import pytest

from shinrin._mlp._backend import get_mlp_native
from shinrin._mlp._layers import MLPConfig, MLPParams
from shinrin._mlp._model import Batch, MLPCore
from shinrin._mlp._mojo_trainer import get_native_trainer
from shinrin._tabm._optim import FlatSpace, lbfgs_minimize
from shinrin._tabm._transforms import PiecewiseLinearEncoder, build_num_bins

TASK_CODES = {"regression": 0, "binary": 1, "multiclass": 2}


def _native_available() -> bool:
    try:
        get_mlp_native()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="Mojo MLP kernels required (run `just build-mlp-mojo`)",
)


def make_case(
    task: str,
    use_emb: bool = True,
    activation: str = "relu",
    seed: int = 0,
    n_samples: int = 96,
):
    """Build a deterministic config/params/batch triple for one task."""
    rng = np.random.RandomState(seed)
    n_features = 5
    X = rng.randn(n_samples, n_features).astype(np.float32)
    bins = build_num_bins(X, 32)
    x_enc = None
    if use_emb:
        x_enc = PiecewiseLinearEncoder(bins).transform(X).astype(np.float32)
    if task == "regression":
        d_out, y = 2, rng.randn(n_samples, 2).astype(np.float32)
    elif task == "binary":
        d_out, y = 1, (rng.rand(n_samples) > 0.5).astype(np.float32)
    elif task == "multiclass":
        d_out, y = 3, rng.randint(0, 3, size=n_samples).astype(np.float32)
    else:
        raise ValueError(f"unknown task {task!r}")
    d_in_effective = (n_features * (4 if use_emb else 1)) + sum([])
    config = MLPConfig(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=d_out,
        layer_sizes=[d_in_effective, 12, 8, d_out],
        activation=activation,
        dropout=0.0,
        use_embeddings=use_emb,
        bins=bins if use_emb else None,
        d_embedding=4,
    )
    params = MLPParams.init(config, seed=42)
    batch = Batch(X, x_enc, None, y)
    return config, params, batch


def _flatten(config, params):
    space = FlatSpace(params)
    theta = params.flatten()
    space.scatter(theta, params)
    return space, theta


# ---------------------------------------------------------------------------
# loss + gradient parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic", "identity"])
def test_loss_grad_parity_regression_activations(activation):
    config, params, batch = make_case("regression", use_emb=True, activation=activation)
    core = MLPCore(config, "regression")
    space, theta = _flatten(config, params)

    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)

    trainer = get_native_trainer(config)
    loss_m, grad_m = trainer.loss_grad(theta, batch, config, task=0)

    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("use_emb", [True, False])
def test_loss_grad_parity_tasks(task, use_emb):
    config, params, batch = make_case(task, use_emb=use_emb)
    core = MLPCore(config, task)
    space, theta = _flatten(config, params)

    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)

    trainer = get_native_trainer(config)
    loss_m, grad_m = trainer.loss_grad(theta, batch, config, task=TASK_CODES[task])

    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


def test_loss_grad_l2_parity():
    """The L2 term must enter loss and gradient identically."""
    config, params, batch = make_case("regression", use_emb=True)
    alpha = 1e-2
    core = MLPCore(config, "regression")
    space, theta = _flatten(config, params)

    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)
    if alpha > 0.0:
        g_np += (alpha * theta).astype(np.float32)
        loss_np += (
            0.5 * alpha * float(theta.astype(np.float64) @ theta.astype(np.float64))
        )

    trainer = get_native_trainer(config)
    loss_m, grad_m = trainer.loss_grad(theta, batch, config, task=0, alpha=alpha)
    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


def test_loss_grad_with_categorical_parity():
    """Categorical one-hot blocks flow through both backends identically."""
    rng = np.random.RandomState(3)
    n_samples, n_num, cards = 80, 3, [4, 3]
    X_num = rng.randn(n_samples, n_num).astype(np.float32)
    bins = build_num_bins(X_num, 16)
    x_enc = PiecewiseLinearEncoder(bins).transform(X_num).astype(np.float32)
    cats = [rng.randint(0, c, size=n_samples).astype(np.float32) for c in cards]
    x_cat = np.concatenate(
        [
            np.eye(c, dtype=np.float32)[cat.astype(np.int64)]
            for c, cat in zip(cards, cats)
        ],
        axis=1,
    )
    y = rng.randn(n_samples).astype(np.float32)

    d_in_effective = n_num * 4 + sum(cards)
    config = MLPConfig(
        n_num_features=n_num,
        cat_cardinalities=cards,
        d_out=1,
        layer_sizes=[d_in_effective, 10, 1],
        activation="relu",
        dropout=0.0,
        use_embeddings=True,
        bins=bins,
        d_embedding=4,
    )
    params = MLPParams.init(config, seed=7)
    batch = Batch(X_num, x_enc, x_cat, y)

    core = MLPCore(config, "regression")
    space, theta = _flatten(config, params)
    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)

    trainer = get_native_trainer(config)
    loss_m, grad_m = trainer.loss_grad(theta, batch, config, task=0)
    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


def test_loss_grad_no_embeddings_parity():
    """Plain raw-feature path (sklearn-equivalent) matches too."""
    config, params, batch = make_case("binary", use_emb=False)
    core = MLPCore(config, "binary")
    space, theta = _flatten(config, params)
    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)

    trainer = get_native_trainer(config)
    loss_m, grad_m = trainer.loss_grad(theta, batch, config, task=TASK_CODES["binary"])
    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# prediction parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_emb", [True, False])
def test_forward_parity(use_emb):
    config, params, batch = make_case("multiclass", use_emb)
    core = MLPCore(config, "multiclass")
    pred_np = core.predict(params, batch)

    trainer = get_native_trainer(config)
    theta = params.flatten()
    out = np.zeros((batch.n_samples, config.d_out), dtype=np.float32)
    trainer.forward(theta, batch, config, out)
    np.testing.assert_allclose(pred_np, out, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# optimizer quality (loose: shuffle RNGs differ between backends)
# ---------------------------------------------------------------------------


def test_adam_epoch_decreases_loss():
    config, params, batch = make_case("regression", use_emb=True, seed=5)
    core = MLPCore(config, "regression")
    space, theta = _flatten(config, params)
    loss_init, _ = core.loss_and_grads(params, batch)

    native = get_native_trainer(config)
    from shinrin._tabm._optim import AdamState

    state = AdamState(space.total)
    loss = loss_init
    for epoch in range(30):
        loss, state.t = native.adam_epoch(
            theta,
            state.m,
            state.v,
            state.t,
            batch,
            config,
            lr=1e-2,
            batch_size=32,
            dropout=0.0,
            alpha=0.0,
            seed=1000 + epoch,
            task=0,
        )
    assert float(loss) < loss_init * 0.9


def test_lbfgs_final_loss_agreement():
    """Both backends' L-BFGS must substantially beat the initial loss."""
    config, params, batch = make_case("regression", use_emb=True, seed=9)
    core = MLPCore(config, "regression")
    space, theta = _flatten(config, params)
    loss_init, _ = core.loss_and_grads(params, batch)

    def fg(t):
        space.scatter(t, params)
        loss, grads = core.loss_and_grads(params, batch)
        return loss, space.flatten_grads(grads)

    _, nit_np, losses_np = lbfgs_minimize(fg, theta.copy(), max_iter=60, tol=1e-6)
    final_np = losses_np[-1]

    native = get_native_trainer(config)
    nit_m, losses_m = native.lbfgs(
        theta.copy(),
        batch,
        config,
        max_iter=60,
        tol=1e-6,
        alpha=0.0,
        task=0,
    )
    final_m = losses_m[-1]

    assert nit_np > 0
    assert nit_m > 0
    assert final_np < loss_init * 0.5
    assert final_m < loss_init * 0.5
