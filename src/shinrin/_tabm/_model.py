"""TabM forward/backward passes in pure NumPy.

Implements the three TabM architectures from yandex-research/tabm
(Apache-2.0, see NOTICE) with manual gradients:

- ``tabm``: BatchEnsemble blocks ``(x * r) @ W.T * s + b``
- ``tabm-mini``: one multiplicative adapter followed by a shared MLP
- ``tabm-packed``: fully independent per-member linears

The training objective is the mean of the per-member losses (each of the
``k`` predictions is compared against the target), matching the official
TabM reference implementation. Inference averages the ``k`` predictions.
"""

from __future__ import annotations

import numpy as np

from ._layers import TabMConfig, TabMParams


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


class Batch:
    """A bundle of model inputs (all float32, ``None`` when absent)."""

    __slots__ = ("x_cat", "x_enc", "x_num", "y")

    def __init__(
        self,
        x_num: np.ndarray | None,
        x_enc: np.ndarray | None,
        x_cat: np.ndarray | None,
        y: np.ndarray,
    ) -> None:
        self.x_num = x_num
        self.x_enc = x_enc
        self.x_cat = x_cat
        self.y = y if y.ndim == 2 else y[:, None]

    def take(self, idx) -> Batch:
        return Batch(
            None if self.x_num is None else self.x_num[idx],
            None if self.x_enc is None else self.x_enc[idx],
            None if self.x_cat is None else self.x_cat[idx],
            self.y[idx],
        )

    @property
    def n_samples(self) -> int:
        ref = self.x_num if self.x_num is not None else self.x_enc
        if ref is None:
            ref = self.x_cat
        assert ref is not None
        return len(ref)


class TabMCore:
    """Forward/backward for a configured TabM architecture."""

    def __init__(self, config: TabMConfig, task: str) -> None:
        if task not in ("regression", "binary", "multiclass"):
            raise ValueError(f"Unknown task: {task!r}")
        self.config = config
        self.task = task

    # -- forward -------------------------------------------------------------

    def _embed(
        self, params: TabMParams, batch: Batch
    ) -> tuple[np.ndarray, dict | None]:
        cfg = self.config
        if not (cfg.use_embeddings and cfg.n_num_features):
            parts = [a for a in (batch.x_num, batch.x_cat) if a is not None]
            if not parts:
                return np.zeros((batch.n_samples, 0), dtype=np.float32), None
            return np.concatenate(parts, axis=1), None

        x_num = batch.x_num
        assert x_num is not None and batch.x_enc is not None
        w0, b0 = params.arrays["emb_w0"], params.arrays["emb_b0"]
        linear0 = x_num[:, :, None] * w0[None] + b0[None]  # (B, F, demb)
        pl = np.zeros_like(linear0)
        offsets = np.cumsum([0] + cfg.bin_counts)
        for f, count in enumerate(cfg.bin_counts):
            wp = params.arrays[f"emb_wp_{f}"]
            pl[:, f] = batch.x_enc[:, offsets[f] : offsets[f + 1]] @ wp
        emb = linear0 + np.maximum(pl, 0.0)
        flat = emb.reshape(batch.n_samples, -1)
        if batch.x_cat is not None:
            flat = np.concatenate([flat, batch.x_cat], axis=1)
        return flat.astype(np.float32, copy=False), {"pl": pl}

    def _backbone(
        self,
        params: TabMParams,
        h: np.ndarray,
        train: bool,
        rng: np.random.RandomState | None,
    ) -> tuple[np.ndarray, list]:
        """Run the ensemble backbone; returns final activations and caches."""
        cfg = self.config
        caches: list = []
        x = h
        for i in range(cfg.n_blocks):
            prefix = f"blk{i}_"
            if cfg.arch_type == "tabm":
                w, r = params.arrays[prefix + "w"], params.arrays[prefix + "r"]
                s, b = params.arrays[prefix + "s"], params.arrays[prefix + "b"]
                if i == 0:
                    # Expand the shared representation to the k members.
                    v = x[:, None, :] * r[None]
                else:
                    v = x * r[None]
                q = v @ w.T
                u = q * s[None] + b[None]
                a = np.maximum(u, 0.0)
                cache: tuple = (x, v, q, u > 0.0)
                x = a
            elif cfg.arch_type == "tabm-mini":
                w, bias = params.arrays[prefix + "w"], params.arrays[prefix + "b"]
                block_in = x
                if i == 0:
                    x = x[:, None, :] * params.arrays["mini_r"][None]
                q = x @ w.T + bias[None, None, :]
                a = np.maximum(q, 0.0)
                cache = (x, q, q > 0.0, block_in)
                x = a
            else:  # tabm-packed
                w, bias = params.arrays[prefix + "w"], params.arrays[prefix + "b"]
                if i == 0:
                    x = np.broadcast_to(x[:, None, :], (len(x), cfg.k, x.shape[1]))
                q = np.einsum("bki,kio->bko", x, w) + bias[None]
                a = np.maximum(q, 0.0)
                cache = (x, q > 0.0)
                x = a

            if train and cfg.dropout > 0.0 and rng is not None:
                mask = (rng.random_sample(x.shape) >= cfg.dropout).astype(np.float32)
                x = x * mask / np.float32(1.0 - cfg.dropout)
                cache += (np.float32(1.0 - cfg.dropout), mask)
            else:
                cache += (None, None)
            caches.append(cache)
        return x, caches

    def forward(
        self,
        params: TabMParams,
        batch: Batch,
        train: bool = False,
        rng: np.random.RandomState | None = None,
    ) -> tuple[np.ndarray, tuple]:
        """Return predictions ``(B, k, d_out)`` and a backward cache."""
        h, emb_cache = self._embed(params, batch)
        x_final, caches = self._backbone(params, h, train, rng)
        head_w, head_b = params.arrays["head_w"], params.arrays["head_b"]
        preds = np.einsum("bki,kio->bko", x_final, head_w) + head_b[None]
        return preds, (emb_cache, caches, x_final)

    def predict(self, params: TabMParams, batch: Batch) -> np.ndarray:
        """Average the ``k`` member predictions -> ``(B, d_out)``."""
        preds, _ = self.forward(params, batch, train=False)
        return preds.mean(axis=1)

    # -- losses ----------------------------------------------------------------

    def loss_and_dpreds(
        self, preds: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray]:
        n = preds.shape[0] * preds.shape[1]
        if self.task == "regression":
            diff = preds - y[:, None, :]
            loss = float(np.mean(diff * diff))
            dpreds = (2.0 / n) * diff
        elif self.task == "binary":
            p = _sigmoid(preds)
            eps = np.float32(1e-7)
            loss = float(
                -np.mean(
                    y[:, None, :] * np.log(p + eps)
                    + (1.0 - y[:, None, :]) * np.log(1.0 - p + eps)
                )
            )
            dpreds = (p - y[:, None, :]) / n
        else:
            logp = _log_softmax(preds)
            idx = y[:, None, :].astype(np.int64)
            loss = float(-np.mean(np.take_along_axis(logp, idx, axis=-1)))
            onehot = np.eye(preds.shape[-1], dtype=np.float32)[y[:, 0].astype(np.int64)]
            dpreds = (np.exp(logp) - onehot[:, None, :]) / n
        return loss, dpreds.astype(np.float32)

    # -- backward ---------------------------------------------------------------

    def backward(
        self,
        params: TabMParams,
        batch: Batch,
        dpreds: np.ndarray,
        cache: tuple,
    ) -> dict[str, np.ndarray]:
        cfg = self.config
        grads: dict[str, np.ndarray] = {}
        emb_cache, caches, x_final = cache

        head_w = params.arrays["head_w"]
        grads["head_w"] = np.einsum("bki,bko->kio", x_final, dpreds)
        grads["head_b"] = dpreds.sum(axis=0)
        da = np.einsum("bko,kio->bki", dpreds, head_w)

        for i in reversed(range(cfg.n_blocks)):
            prefix = f"blk{i}_"
            c = caches[i]
            scale, mask = c[-2], c[-1]
            if mask is not None:
                assert scale is not None
                da = da * mask / scale
            if cfg.arch_type == "tabm":
                x_in, v, q, relu_mask = c[0], c[1], c[2], c[3]
                w, r = params.arrays[prefix + "w"], params.arrays[prefix + "r"]
                s = params.arrays[prefix + "s"]
                du = da * relu_mask
                grads[prefix + "b"] = du.sum(axis=0)
                grads[prefix + "s"] = (du * q).sum(axis=0)
                dq = du * s[None]
                grads[prefix + "w"] = np.einsum("bjo,bji->oi", dq, v)
                dv = dq @ w
                x_in_3d = x_in if x_in.ndim == 3 else x_in[:, None, :]
                grads[prefix + "r"] = np.einsum("bji,bji->ji", dv, x_in_3d)
                da = dv * r[None]
            elif cfg.arch_type == "tabm-mini":
                x_lin, _, relu_mask, block_in = c[0], c[1], c[2], c[3]
                w = params.arrays[prefix + "w"]
                du = da * relu_mask
                grads[prefix + "b"] = du.sum(axis=(0, 1))
                grads[prefix + "w"] = np.einsum("bki,bko->io", du, x_lin)
                da = np.einsum("bko,oi->bki", du, w)
                if i == 0:
                    h = block_in
                    mini_r = params.arrays["mini_r"]
                    grads["mini_r"] = np.einsum("bji,bji->ji", da, h[:, None, :])
                    da = da * mini_r[None]
            else:  # tabm-packed
                x_in, relu_mask = c[0], c[1]
                w = params.arrays[prefix + "w"]
                du = da * relu_mask
                grads[prefix + "b"] = du.sum(axis=0)
                grads[prefix + "w"] = np.einsum("bki,bko->kio", x_in, du)
                da = np.einsum("bko,kio->bki", du, w)

        if emb_cache is not None:
            # The embedding output is shared across the k members, so its
            # gradient accumulates over the member axis. Categorical blocks
            # pass through untouched, so only the numerical part is reshaped.
            x_num = batch.x_num
            x_enc = batch.x_enc
            assert x_num is not None and x_enc is not None
            dh = da.sum(axis=1)[:, : cfg.n_num_features * cfg.d_embedding]
            demb_grad = dh.reshape(dh.shape[0], cfg.n_num_features, cfg.d_embedding)
            grads["emb_w0"] = np.einsum("bf,bfo->fo", x_num, demb_grad)
            grads["emb_b0"] = demb_grad.sum(axis=0)
            # Subgradient at zero passes through so the zero-initialized
            # piecewise-linear projections can start learning.
            dpl = demb_grad * (emb_cache["pl"] >= 0.0)
            offsets = np.cumsum([0] + cfg.bin_counts)
            for f, count in enumerate(cfg.bin_counts):
                grads[f"emb_wp_{f}"] = np.einsum(
                    "bi,bo->io",
                    x_enc[:, offsets[f] : offsets[f + 1]],
                    dpl[:, f],
                )
        return grads

    # -- chunked full-batch loss + gradients -------------------------------------

    def loss_and_grads(
        self,
        params: TabMParams,
        batch: Batch,
        rng: np.random.RandomState | None = None,
        max_members: int = 16384,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Full-batch loss and gradients, accumulated over row chunks.

        ``max_members`` bounds ``chunk_size * k`` to keep activation memory
        predictable (relevant for full-batch L-BFGS).
        """
        cfg = self.config
        n = batch.n_samples
        chunk = max(1, min(n, max_members // max(1, cfg.k)))
        total_loss = 0.0
        weight_total = 0
        grads: dict[str, np.ndarray] | None = None
        for start in range(0, n, chunk):
            sub = batch.take(slice(start, min(start + chunk, n)))
            preds, cache = self.forward(params, sub, train=rng is not None, rng=rng)
            loss, dpreds = self.loss_and_dpreds(preds, sub.y)
            g = self.backward(params, sub, dpreds, cache)
            members = sub.n_samples * cfg.k
            total_loss += loss * members
            weight_total += members
            if grads is None:
                grads = g
            else:
                for key, val in g.items():
                    grads[key] += val
        assert grads is not None
        return total_loss / weight_total, grads
