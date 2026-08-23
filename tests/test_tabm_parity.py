"""Parity tests between the NumPy reference and the Mojo TabM kernels.

The Mojo kernels implement the same forward/backward as
``shinrin._tabm._model.TabMCore`` over an identical flat parameter layout,
so loss, gradients and predictions must agree up to float32
accumulation-order noise. Optimizer trajectories are only compared
loosely (shuffle RNGs differ between backends).
"""

from __future__ import annotations

import numpy as np
import pytest

from shinrin._tabm._backend import get_tabm_native
from shinrin._tabm._layers import TabMConfig, TabMParams
from shinrin._tabm._model import Batch, TabMCore
from shinrin._tabm._optim import AdamState, FlatSpace, lbfgs_minimize
from shinrin._tabm._transforms import PiecewiseLinearEncoder, build_num_bins

TASK_CODES = {"regression": 0, "binary": 1, "multiclass": 2}


def _native_available() -> bool:
    try:
        get_tabm_native()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="Mojo TabM kernels required (run `just build-tabm-mojo`)",
)


def make_case(
    task: str,
    use_emb: bool = True,
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
        d_out, y = 1, rng.randn(n_samples).astype(np.float32)
    elif task == "binary":
        d_out, y = 1, (rng.rand(n_samples) > 0.5).astype(np.float32)
    elif task == "multiclass":
        d_out, y = 3, rng.randint(0, 3, size=n_samples).astype(np.float32)
    else:
        raise ValueError(f"unknown task {task!r}")
    config = TabMConfig(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=d_out,
        k=8,
        n_blocks=2,
        d_block=16,
        dropout=0.0,
        arch_type="tabm",
        use_embeddings=use_emb,
        bins=bins if use_emb else None,
    )
    params = TabMParams.init(config, seed=42)
    batch = Batch(X, x_enc, None, y)
    return config, params, batch


def _numpy_loss_grad(config, params, batch, alpha=0.0):
    core = TabMCore(config, _task_name(config))
    space = FlatSpace(params)
    theta = params.flatten()
    space.scatter(theta, params)
    loss, grads = core.loss_and_grads(params, batch)
    g = space.flatten_grads(grads)
    if alpha > 0.0:
        g += (alpha * theta).astype(np.float32)
        loss += 0.5 * alpha * float(theta.astype(np.float64) @ theta.astype(np.float64))
    return loss, g.astype(np.float32)


def _task_name(config: TabMConfig) -> str:
    if config.d_out == 1:
        return "regression"
    return "multiclass"


# ---------------------------------------------------------------------------
# loss + gradient parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_emb", [True, False])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_loss_grad_parity(task, use_emb):
    config, params, batch = make_case(task, use_emb)
    core = TabMCore(config, task)
    space = FlatSpace(params)
    theta = params.flatten()

    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)

    from shinrin._tabm._mojo_trainer import get_tabm_trainer

    trainer = get_tabm_trainer(config)
    dummy = np.zeros(1, dtype=np.float32)
    x_enc = batch.x_enc if use_emb else dummy
    loss_m, grad_m = trainer.loss_grad(
        [theta, batch.x_num, x_enc, dummy, batch.y, TASK_CODES[task], 0.0]
    )

    # Loose tolerances: float32 accumulation order differs between the
    # chunked NumPy path and the Mojo kernels.
    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


def test_loss_grad_l2_parity():
    """The L2 term must enter loss and gradient identically."""
    config, params, batch = make_case("regression", use_emb=True)
    alpha = 1e-2
    loss_np, g_np = _numpy_loss_grad(config, params, batch, alpha=alpha)

    from shinrin._tabm._mojo_trainer import get_tabm_trainer

    trainer = get_tabm_trainer(config)
    theta = params.flatten()
    loss_m, grad_m = trainer.loss_grad(
        [
            theta,
            batch.x_num,
            batch.x_enc,
            np.zeros(1, dtype=np.float32),
            batch.y,
            0,
            alpha,
        ]
    )
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

    config = TabMConfig(
        n_num_features=n_num,
        cat_cardinalities=cards,
        d_out=1,
        k=8,
        n_blocks=2,
        d_block=16,
        dropout=0.0,
        arch_type="tabm",
        use_embeddings=True,
        bins=bins,
    )
    params = TabMParams.init(config, seed=7)
    batch = Batch(X_num, x_enc, x_cat, y)

    core = TabMCore(config, "regression")
    space = FlatSpace(params)
    theta = params.flatten()
    loss_np, grads_np = core.loss_and_grads(params, batch)
    g_np = space.flatten_grads(grads_np)

    from shinrin._tabm._mojo_trainer import get_tabm_trainer

    trainer = get_tabm_trainer(config)
    loss_m, grad_m = trainer.loss_grad([theta, X_num, x_enc, x_cat, y[:, None], 0, 0.0])
    assert loss_np == pytest.approx(float(loss_m), rel=1e-2, abs=1e-3)
    np.testing.assert_allclose(g_np, np.asarray(grad_m), rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# prediction parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_emb", [True, False])
def test_forward_avg_parity(use_emb):
    config, params, batch = make_case("regression", use_emb)
    core = TabMCore(config, "regression")
    pred_np = core.predict(params, batch)

    from shinrin._tabm._mojo_trainer import get_tabm_trainer

    trainer = get_tabm_trainer(config)
    theta = params.flatten()
    dummy = np.zeros(1, dtype=np.float32)
    x_enc = batch.x_enc if batch.x_enc is not None else dummy
    out = np.zeros((batch.n_samples, config.d_out), dtype=np.float32)
    trainer.forward_avg([theta, batch.x_num, x_enc, dummy, out])
    np.testing.assert_allclose(pred_np, out, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# optimizer quality (loose: shuffle RNGs differ between backends)
# ---------------------------------------------------------------------------


def test_adam_epoch_decreases_loss():
    config, params, batch = make_case("regression", use_emb=True, seed=5)
    core = TabMCore(config, "regression")
    space = FlatSpace(params)
    theta = params.flatten()
    space.scatter(theta, params)
    loss_init, _ = core.loss_and_grads(params, batch)

    from shinrin._tabm._mojo_trainer import NativeTrainer, get_tabm_trainer

    native = NativeTrainer(get_tabm_trainer(config), config)
    state = AdamState(space.total)
    loss = loss_init
    for epoch in range(30):
        loss, state.t = native.adam_epoch(
            theta,
            state.m,
            state.v,
            state.t,
            batch,
            params,
            space,
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
    core = TabMCore(config, "regression")
    space = FlatSpace(params)
    theta = params.flatten()
    space.scatter(theta, params)
    loss_init, _ = core.loss_and_grads(params, batch)

    def fg(t):
        space.scatter(t, params)
        loss, grads = core.loss_and_grads(params, batch)
        return loss, space.flatten_grads(grads)

    _, nit_np, losses_np = lbfgs_minimize(fg, theta.copy(), max_iter=60, tol=1e-6)
    final_np = losses_np[-1]

    from shinrin._tabm._mojo_trainer import NativeTrainer, get_tabm_trainer

    native = NativeTrainer(get_tabm_trainer(config), config)
    nit_m, losses_m = native.lbfgs(
        theta.copy(),
        batch,
        params,
        space,
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


def test_predict_with_cache_parity():
    """predict_with_cache must match forward_avg for the same query data."""
    config, params, batch = make_case("regression", use_emb=True, seed=42)
    space = FlatSpace(params)
    theta = params.flatten()
    space.scatter(theta, params)

    from shinrin._tabm._mojo_trainer import NativeTrainer, get_tabm_trainer

    native = NativeTrainer(get_tabm_trainer(config), config)

    # Build cache from training data
    native.build_cache(theta, batch, params)

    # Predict on query data using cache
    preds_cache = native.predict_with_cache(theta, batch, params)

    # Predict using regular forward_avg
    preds_normal = native.forward_avg(theta, batch, params)

    # Must match within floating-point tolerance
    np.testing.assert_allclose(
        preds_cache,
        preds_normal,
        rtol=1e-5,
        atol=1e-6,
        err_msg="cache predictions must match forward_avg",
    )


def test_predict_with_cache_vs_numpy():
    """Cache-based predictions must match NumPy reference implementation."""
    config, params, batch = make_case("binary", use_emb=False, seed=7)
    core = TabMCore(config, "binary")
    space = FlatSpace(params)
    theta = params.flatten()
    space.scatter(theta, params)

    # NumPy reference
    preds_np = core.predict(params, batch)

    from shinrin._tabm._mojo_trainer import NativeTrainer, get_tabm_trainer

    native = NativeTrainer(get_tabm_trainer(config), config)

    # Build cache from training data
    native.build_cache(theta, batch, params)

    # Predict on query data using cache
    preds_cache = native.predict_with_cache(theta, batch, params)

    # Must match NumPy reference
    np.testing.assert_allclose(
        preds_cache,
        preds_np,
        rtol=1e-4,
        atol=1e-5,
        err_msg="cache predictions must match NumPy reference",
    )
