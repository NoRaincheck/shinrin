"""MLP forward/backward passes in pure NumPy.

Implements a plain multi-layer perceptron matching
``sklearn.neural_network`` semantics:

- hidden activations ``relu`` / ``tanh`` / ``logistic`` / ``identity``
- linear output layer; task losses are applied on the raw outputs:
  squared loss (regression), binary cross entropy (logistic) and softmax
  cross entropy (multiclass)
- loss conventions follow scikit-learn: regression reports
  ``0.5 * mean((pred - y)**2)`` and gradients are averaged per sample

The optional PLE embedding (``use_embeddings=True``) mirrors TabM's
numerical embedding: piecewise-linear encoding followed by a trainable
per-feature linear map with ReLU.
"""

from __future__ import annotations

import numpy as np

from shinrin._tabm._model import Batch

from ._layers import MLPConfig, MLPParams

TASK_LOSSES = ("regression", "binary", "multiclass")

__all__ = ["TASK_LOSSES", "Batch", "MLPCore"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _activate(x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(activation, derivative_at_pre_activation)``."""
    if kind == "relu":
        return np.maximum(x, 0.0), (x > 0.0).astype(np.float32)
    if kind == "logistic":
        s = _sigmoid(x)
        return s, s * (1.0 - s)
    if kind == "tanh":
        t = np.tanh(x).astype(np.float32)
        return t, 1.0 - t * t
    if kind == "identity":
        return x.astype(np.float32, copy=False), np.ones_like(x)
    raise ValueError(f"Unknown activation: {kind!r}")


class MLPCore:
    """Forward/backward for a configured plain MLP."""

    def __init__(self, config: MLPConfig, task: str) -> None:
        if task not in TASK_LOSSES:
            raise ValueError(f"Unknown task: {task!r}")
        self.config = config
        self.task = task

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        params: MLPParams,
        batch: Batch,
        train: bool = False,
        rng: np.random.RandomState | None = None,
    ) -> tuple[np.ndarray, tuple]:
        """Return raw output predictions ``(B, d_out)`` and a backward cache."""
        cfg = self.config
        emb, pl = self._embed_forward(params, batch)
        parts: list[np.ndarray] = []
        if emb is not None:
            parts.append(emb)
        elif batch.x_num is not None:
            parts.append(batch.x_num)
        if batch.x_cat is not None:
            parts.append(batch.x_cat)
        h = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        h = np.ascontiguousarray(h, dtype=np.float32)

        caches: list = []
        use_dropout = cfg.dropout > 0.0 and train and rng is not None
        n_hidden = cfg.n_layers - 1
        for i in range(cfg.n_layers):
            w, b = params.arrays[f"l{i}_w"], params.arrays[f"l{i}_b"]
            z = h @ w.T + b[None, :]
            if i < n_hidden:
                a, deriv = _activate(z, cfg.activation)
                mask_scale = None
                if use_dropout:
                    keep = np.float32(1.0 / (1.0 - cfg.dropout))
                    mask = (rng.random_sample(a.shape) >= cfg.dropout).astype(
                        np.float32
                    )
                    a = a * mask * keep
                    mask_scale = (keep, mask)
                caches.append((h, deriv, mask_scale))
                h = a
            else:
                caches.append((h, None, None))
                h = z
        preds = h
        return preds.astype(np.float32, copy=False), (emb, pl, caches)

    def _embed_forward(
        self, params: MLPParams, batch: Batch
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """PLE embedding flattened ``(B, F * demb)`` plus pre-ReLU values.

        Returns ``(None, empty)`` when the embedding is disabled.
        """
        cfg = self.config
        if not (cfg.use_embeddings and cfg.n_num_features):
            return None, np.empty(0, dtype=np.float32)
        x_num = batch.x_num
        assert x_num is not None and batch.x_enc is not None
        w0, b0 = params.arrays["emb_w0"], params.arrays["emb_b0"]
        linear0 = x_num[:, :, None] * w0[None] + b0[None]  # (B, F, demb)
        pl = np.empty_like(linear0)
        offsets = np.cumsum([0] + cfg.bin_counts)
        for f, count in enumerate(cfg.bin_counts):
            wp = params.arrays[f"emb_wp_{f}"]
            pl[:, f] = batch.x_enc[:, offsets[f] : offsets[f + 1]] @ wp
            linear0[:, f] += np.maximum(pl[:, f], 0.0)
        return linear0.reshape(batch.n_samples, -1), pl

    def predict(self, params: MLPParams, batch: Batch) -> np.ndarray:
        """Raw output predictions ``(B, d_out)`` (no output activation)."""
        preds, _ = self.forward(params, batch, train=False)
        return preds

    # -- losses ----------------------------------------------------------------

    def loss_and_dpreds(
        self, preds: np.ndarray, y: np.ndarray, denom_b: int | None = None
    ) -> tuple[float, np.ndarray]:
        """Loss value and gradient wrt the raw predictions.

        ``denom_b`` lets callers normalize by the full minibatch size when
        chunking rows across threads; it defaults to ``len(preds)``.
        """
        b = preds.shape[0] if denom_b is None else denom_b
        eps = np.float32(1e-15)
        if self.task == "regression":
            diff = preds - y
            # sklearn convention: half the mean squared error over all elements
            loss = 0.5 * float(np.sum(diff * diff)) / b
            dpreds = diff / float(b)
        elif self.task == "binary":
            p = np.clip(_sigmoid(preds), eps, 1.0 - eps)
            loss = float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)) / b)
            dpreds = (_sigmoid(preds) - y) / float(b)
        else:
            logp = _log_softmax(preds)
            idx = y[:, 0].astype(np.int64)
            loss = float(-np.sum(np.take_along_axis(logp, idx[:, None], axis=-1)) / b)
            onehot = np.eye(preds.shape[-1], dtype=np.float32)[idx]
            dpreds = (np.exp(logp) - onehot) / float(b)
        return loss, dpreds.astype(np.float32)

    # -- backward ---------------------------------------------------------------

    def backward(
        self,
        params: MLPParams,
        batch: Batch,
        dpreds: np.ndarray,
        cache: tuple,
    ) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        _, pl, caches = cache

        da = dpreds  # (B, d_out); output layer is linear
        for i in reversed(range(self.config.n_layers)):
            h, _, _ = caches[i]
            grads[f"l{i}_w"] = da.T @ h
            grads[f"l{i}_b"] = da.sum(axis=0)
            if i > 0:
                # Multiply by the incoming activation's derivative (and the
                # inverted dropout mask) of layer i-1, whose outputs feed
                # layer i.
                deriv, mask_scale = caches[i - 1][1], caches[i - 1][2]
                da = da @ params.arrays[f"l{i}_w"] * deriv
                if mask_scale is not None:
                    keep, mask = mask_scale
                    da = da * mask * keep

        if pl.size:
            # Gradient wrt the embedding output = dz_0 @ W_0 (the embedding
            # feeds layer 0's input directly).
            d_in_grad = da @ params.arrays["l0_w"]
            self._embed_backward(params, batch, pl, d_in_grad, grads)
        return grads

    def _embed_backward(
        self,
        params: MLPParams,
        batch: Batch,
        pl: np.ndarray,
        da: np.ndarray,
        grads: dict[str, np.ndarray],
    ) -> None:
        cfg = self.config
        assert batch.x_num is not None and batch.x_enc is not None
        dh = da[:, : cfg.n_num_features * cfg.d_embedding].reshape(
            da.shape[0], cfg.n_num_features, cfg.d_embedding
        )
        grads["emb_w0"] = np.einsum("bf,bfo->fo", batch.x_num, dh)
        grads["emb_b0"] = dh.sum(axis=0)
        dpl = dh * (pl >= 0.0)
        offsets = np.cumsum([0] + cfg.bin_counts)
        for f, count in enumerate(cfg.bin_counts):
            grads[f"emb_wp_{f}"] = np.einsum(
                "bi,bo->io",
                batch.x_enc[:, offsets[f] : offsets[f + 1]],
                dpl[:, f],
            )

    # -- full-batch loss + gradients ---------------------------------------------

    def loss_and_grads(
        self,
        params: MLPParams,
        batch: Batch,
        rng: np.random.RandomState | None = None,
        max_chunk_rows: int = 8192,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Full-batch loss and gradients, accumulated over row chunks."""
        n = batch.n_samples
        chunk = max(1, min(n, max_chunk_rows))
        total_loss = 0.0
        grads: dict[str, np.ndarray] | None = None
        for start in range(0, n, chunk):
            sub = batch.take(slice(start, min(start + chunk, n)))
            preds, cache = self.forward(params, sub, train=rng is not None, rng=rng)
            loss, dpreds = self.loss_and_dpreds(preds, sub.y, denom_b=n)
            g = self.backward(params, sub, dpreds, cache)
            total_loss += loss * sub.n_samples
            if grads is None:
                grads = g
            else:
                for key, val in g.items():
                    grads[key] += val
        assert grads is not None
        return total_loss / max(n, 1), grads
