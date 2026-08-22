"""Pure NumPy reference implementation of the TabICLv2 architecture.

Op-for-op mirror of :mod:`shinrin._tabicl._model_torch` so that every backend
loads the same ``.npz`` weights and produces numerically equivalent results
(within floating-point tolerance). Inference-only.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

from ._config import TabICLConfig
from ._many_classes import (
    ClassNode,
    compute_mixed_radix_bases,
    extract_mixed_radix_digit,
    fit_hierarchical_tree,
    label_encoding,
)

SKIP_VALUE = -100.0


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf-based) GELU matching ``torch.nn.GELU``."""
    return 0.5 * x * (1.0 + erf(x / np.float32(np.sqrt(2.0))))


class RotaryEmbedding:
    """Rotary positional encoding with a ``freqs`` array."""

    def __init__(
        self, dim: int, theta: float = 10000.0, interleaved: bool = True
    ) -> None:
        self.interleaved = interleaved
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.freqs = freqs

    def rotate(self, x: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        """Rotate ``x`` of shape (..., n_heads, seq_len, head_dim)."""
        seq_len = x.shape[-2]
        if positions is None:
            positions = np.arange(seq_len, dtype=np.float32)
        frq = self.freqs.astype(np.float32)
        pos = positions.astype(np.float32)
        freqs = np.outer(pos, frq)  # (T, hd/2)
        original_dtype = x.dtype
        x_f = x.astype(np.float32)
        if self.interleaved:
            freqs = np.concatenate([freqs, freqs], axis=-1)  # (T, hd)
            cos, sin = np.cos(freqs), np.sin(freqs)
            x1, x2 = x_f[..., 0::2], x_f[..., 1::2]
            rotated = np.stack((-x2, x1), axis=-1).reshape(x_f.shape)
            out = x_f * cos + rotated * sin
        else:
            cos, sin = np.cos(freqs), np.sin(freqs)  # (T, hd/2)
            half = x_f.shape[-1] // 2
            x1, x2 = x_f[..., :half], x_f[..., half:]
            out = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return out.astype(original_dtype)


class SSMax:
    """Scalable softmax with learnable per-head scale."""

    def __init__(self, num_heads: int) -> None:
        self.scales = np.ones(num_heads, dtype=np.float32)

    def forward(self, q: np.ndarray, n: int) -> np.ndarray:
        logn = np.float32(math.log(max(n, 1)))
        return q * (self.scales.reshape(1, -1, 1, 1) * logn)


class SSMaxMLP:
    """Scalable softmax with an MLP mapping log(n) to scales."""

    def __init__(
        self,
        num_heads: int,
        n_hidden: int = 64,
        elementwise: bool = False,
        head_dim: int | None = None,
    ) -> None:
        self.elementwise = elementwise
        if elementwise:
            if head_dim is None:
                raise ValueError("head_dim is required for elementwise SSMaxMLP")
            out_dim = num_heads * head_dim
        else:
            out_dim = num_heads
        self.num_heads = num_heads
        self.w1 = np.random.randn(1, n_hidden).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.w2 = np.random.randn(n_hidden, out_dim).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, q: np.ndarray, n: int) -> np.ndarray:
        logn = np.float32(math.log(max(n, 1))).reshape(1, 1)
        hidden = _gelu(logn @ self.w1 + self.b1)
        scales = hidden @ self.w2 + self.b2
        if self.elementwise:
            scales = scales.reshape(1, self.num_heads, 1, q.shape[-1])
        else:
            scales = scales.reshape(1, self.num_heads, 1, 1)
        return q * scales


class QASSMaxMLP:
    """Query-aware scalable softmax with base and query MLPs."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
        elementwise: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.elementwise = elementwise
        base_out = num_heads * head_dim if elementwise else num_heads
        query_out = head_dim if elementwise else 1
        self.base_w1 = np.random.randn(1, n_hidden).astype(np.float32)
        self.base_b1 = np.zeros(n_hidden, dtype=np.float32)
        self.base_w2 = np.random.randn(n_hidden, base_out).astype(np.float32)
        self.base_b2 = np.zeros(base_out, dtype=np.float32)
        self.query_w1 = np.random.randn(head_dim, n_hidden).astype(np.float32)
        self.query_b1 = np.zeros(n_hidden, dtype=np.float32)
        self.query_w2 = np.random.randn(n_hidden, query_out).astype(np.float32)
        self.query_b2 = np.zeros(query_out, dtype=np.float32)

    def forward(self, q: np.ndarray, n: int) -> np.ndarray:
        logn = np.float32(math.log(max(n, 1))).reshape(1, 1)
        base_hidden = _gelu(logn @ self.base_w1 + self.base_b1)
        base_scales = base_hidden @ self.base_w2 + self.base_b2
        if self.elementwise:
            base_scales = base_scales.reshape(1, self.num_heads, 1, self.head_dim)
        else:
            base_scales = base_scales.reshape(1, self.num_heads, 1, 1)
        q_f = q.astype(np.float32)
        mod_hidden = _gelu(q_f @ self.query_w1 + self.query_b1)
        modulation = 1.0 + np.tanh(mod_hidden @ self.query_w2 + self.query_b2)
        return q * (base_scales * modulation)


def create_ssmax_layer(
    ssmax_type: str | bool, num_heads: int, embed_dim: int
) -> SSMax | SSMaxMLP | QASSMaxMLP | None:
    """Factory matching the upstream ssmax variant names."""
    if ssmax_type in (None, False, "none"):
        return None
    if ssmax_type is True:
        ssmax_type = "qassmax-mlp-elementwise"
    if ssmax_type == "ssmax":
        return SSMax(num_heads)
    if ssmax_type == "ssmax-mlp":
        return SSMaxMLP(num_heads)
    if ssmax_type == "ssmax-mlp-elementwise":
        return SSMaxMLP(num_heads, head_dim=embed_dim // num_heads, elementwise=True)
    if ssmax_type == "qassmax-mlp":
        return QASSMaxMLP(num_heads, embed_dim // num_heads)
    if ssmax_type == "qassmax-mlp-elementwise":
        return QASSMaxMLP(num_heads, embed_dim // num_heads, elementwise=True)
    raise ValueError(f"Unknown ssmax type: {ssmax_type}")


class MultiheadAttention:
    """Multi-head attention with packed in-projection and optional KV cache."""

    def __init__(
        self, embed_dim: int, num_heads: int, ssmax: str | bool = False
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ssmax_layer = create_ssmax_layer(ssmax, num_heads, embed_dim)
        # Loaded from weights at init time.
        self.in_proj_weight: np.ndarray | None = None
        self.in_proj_bias: np.ndarray | None = None
        self.out_proj_w: np.ndarray | None = None
        self.out_proj_b: np.ndarray | None = None

    def __call__(
        self,
        q: np.ndarray,
        k: np.ndarray | None = None,
        v: np.ndarray | None = None,
        rope: RotaryEmbedding | None = None,
        cached_kv: tuple[np.ndarray, np.ndarray] | None = None,
        need_kv: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _attention_numpy(
            self, q, k, v, rope=rope, cached_kv=cached_kv, need_kv=need_kv
        )


def _attention_numpy(
    attn: MultiheadAttention,
    q_in: np.ndarray,
    k_in: np.ndarray | None,
    v_in: np.ndarray | None,
    rope: RotaryEmbedding | None = None,
    cached_kv: tuple[np.ndarray, np.ndarray] | None = None,
    need_kv: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-head attention (NumPy)."""
    embed_dim = q_in.shape[-1]
    num_heads, head_dim = attn.num_heads, attn.head_dim
    w = attn.in_proj_weight
    b = attn.in_proj_bias

    if cached_kv is None:
        assert k_in is not None and v_in is not None
        q = q_in @ w[:embed_dim].T + b[:embed_dim]  # ty: ignore[not-subscriptable]
        k = k_in @ w[embed_dim : 2 * embed_dim].T + b[embed_dim : 2 * embed_dim]  # ty: ignore[not-subscriptable]
        v = v_in @ w[2 * embed_dim :].T + b[2 * embed_dim :]  # ty: ignore[not-subscriptable]
        q = np.swapaxes(q.reshape(*q.shape[:-1], num_heads, head_dim), -2, -3)
        k = np.swapaxes(k.reshape(*k.shape[:-1], num_heads, head_dim), -2, -3)
        v = np.swapaxes(v.reshape(*v.shape[:-1], num_heads, head_dim), -2, -3)
        if rope is not None:
            q = rope.rotate(q)
            k = rope.rotate(k)
    else:
        k, v = cached_kv
        q = q_in @ w[:embed_dim].T + b[:embed_dim]  # ty: ignore[not-subscriptable]
        q = np.swapaxes(q.reshape(*q.shape[:-1], num_heads, head_dim), -2, -3)
        if rope is not None:
            q = rope.rotate(q)

    src_len = k.shape[-2]
    if attn.ssmax_layer is not None:
        q = attn.ssmax_layer.forward(q, src_len)

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * (head_dim**-0.5)
    attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
    out = np.matmul(attn_weights, v)
    batch_shape = q_in.shape[:-1]
    out = np.swapaxes(out, -2, -3).reshape(*batch_shape, embed_dim)
    out_proj_w = attn.out_proj_w
    out_proj_b = attn.out_proj_b
    out = out @ out_proj_w.T + out_proj_b  # ty: ignore[unresolved-attribute]

    if need_kv and cached_kv is None:
        return out, k, v
    return out


class AttentionBlock:
    """Pre-norm attention + feedforward block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        ssmax: str | bool = False,
        bias_free_ln: bool = False,
    ) -> None:
        self.attn = MultiheadAttention(d_model, nhead, ssmax)
        self.linear1_w: np.ndarray = np.zeros(
            (d_model, dim_feedforward), dtype=np.float32
        )
        self.linear1_b: np.ndarray = np.zeros(dim_feedforward, dtype=np.float32)
        self.linear2_w: np.ndarray = np.zeros(
            (dim_feedforward, d_model), dtype=np.float32
        )
        self.linear2_b: np.ndarray = np.zeros(d_model, dtype=np.float32)
        # LayerNorm affine parameters (gamma always present; beta unless
        # ``bias_free_ln``, mirroring ``nn.LayerNorm(bias=not bias_free_ln)``).
        self.norm1_w: np.ndarray = np.ones(d_model, dtype=np.float32)
        self.norm2_w: np.ndarray = np.ones(d_model, dtype=np.float32)
        self.norm1_b: np.ndarray | None = (
            None if bias_free_ln else np.zeros(d_model, dtype=np.float32)
        )
        self.norm2_b: np.ndarray | None = (
            None if bias_free_ln else np.zeros(d_model, dtype=np.float32)
        )
        self.norm_first = True
        self.bias_free_ln = bias_free_ln

    def _layer_norm(self, x: np.ndarray, which: int = 1) -> np.ndarray:
        gamma = self.norm1_w if which == 1 else self.norm2_w
        beta = self.norm1_b if which == 1 else self.norm2_b
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True, ddof=0)
        out = (x - mean) / np.sqrt(var + 1e-5)
        return out * gamma if beta is None else out * gamma + beta

    def __call__(
        self,
        q: np.ndarray,
        k: np.ndarray | None = None,
        v: np.ndarray | None = None,
        train_size: int | None = None,
        rope: RotaryEmbedding | None = None,
        cached_kv: tuple[np.ndarray, np.ndarray] | None = None,
        need_kv: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if train_size is not None:
            k = q[..., :train_size, :]
            v = k
        else:
            if k is None:
                k = q
            if v is None:
                v = k

        if cached_kv is not None:
            attn_out = self.attn(self._layer_norm(q, 1), cached_kv=cached_kv, rope=rope)
            x = q + attn_out
        elif self.norm_first:
            qn = self._layer_norm(q, 1)
            kn = qn if k is q else self._layer_norm(k, 1)
            vn = kn if v is k else self._layer_norm(v, 1)
            result = self.attn(qn, kn, vn, rope=rope, need_kv=need_kv)
            if need_kv:
                attn_out, k_proj, v_proj = result
            else:
                attn_out = result
            x = q + attn_out
        else:  # pragma: no cover - checkpoints use pre-norm
            result = self.attn(q, k, v, rope=rope, need_kv=need_kv)
            if need_kv:
                attn_out, k_proj, v_proj = result
            else:
                attn_out = result
            x = self._layer_norm(q + attn_out)

        ff = (
            _gelu(self._layer_norm(x, 2) @ self.linear1_w + self.linear1_b)
            @ self.linear2_w
            + self.linear2_b
        )
        x = x + ff
        if need_kv and cached_kv is None:
            return x, k_proj, v_proj
        return x


class InducedSelfAttentionBlock:
    """Two-stage induced self-attention (ISAB)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        ssmax: str | bool = False,
        bias_free_ln: bool = False,
    ) -> None:
        self.multihead_attn1 = AttentionBlock(
            d_model, nhead, dim_feedforward, ssmax, bias_free_ln
        )
        self.multihead_attn2 = AttentionBlock(
            d_model, nhead, dim_feedforward, False, bias_free_ln
        )
        self.num_inds = num_inds
        self.ind_vectors: np.ndarray = np.zeros((num_inds, d_model), dtype=np.float32)
        self.skip_value = SKIP_VALUE

    def induced_attention(
        self, src: np.ndarray, train_size: int | None = None
    ) -> np.ndarray:
        *batch, _, d_model = src.shape
        ind = np.broadcast_to(self.ind_vectors, (*batch, self.num_inds, d_model)).copy()
        if train_size is None:
            hidden = self.multihead_attn1(ind, src, src)
        else:
            hidden = self.multihead_attn1(
                ind, src[..., :train_size, :], src[..., :train_size, :]
            )
        return self.multihead_attn2(src, hidden, hidden)  # ty: ignore[invalid-argument-type, invalid-return-type]

    def __call__(self, src: np.ndarray, train_size: int | None = None) -> np.ndarray:
        skip_mask = (src == self.skip_value).all(axis=(-2, -1))
        if not skip_mask.any():
            return self.induced_attention(src, train_size)
        if bool(skip_mask.all()):
            return np.full_like(src, self.skip_value)
        out = np.empty_like(src)
        out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
        out[skip_mask] = self.skip_value
        return out


class SetTransformer:
    """Stack of ISAB blocks with optional KV caching."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        ssmax: str | bool = False,
        bias_free_ln: bool = False,
    ) -> None:
        self.blocks = [
            InducedSelfAttentionBlock(
                d_model, nhead, dim_feedforward, num_inds, ssmax, bias_free_ln
            )
            for _ in range(num_blocks)
        ]

    def __call__(self, src: np.ndarray, train_size: int | None = None) -> np.ndarray:
        out = src
        for block in self.blocks:
            out = block(src=out, train_size=train_size)
        return out

    def forward_with_cache(
        self,
        src: np.ndarray,
        cache_list: list,
        train_size: int,
        use_cache: bool,
    ) -> np.ndarray:
        """Cache-aware forward. Like upstream, the cache path does not exclude
        skipped (all -100) columns from stage-1 keys; skip values are restored
        between blocks."""
        skip_mask = (src == SKIP_VALUE).all(axis=(-2, -1))
        out = src
        for idx, block in enumerate(self.blocks):
            if use_cache:
                hidden_k, hidden_v = cache_list[idx]
                out = block.multihead_attn2(
                    out,  # ty: ignore[invalid-argument-type]
                    cached_kv=(hidden_k, hidden_v),
                )
            else:
                *batch, _, d_model = out.shape  # ty: ignore[unresolved-attribute]
                ind_vecs = block.ind_vectors
                n_inds = ind_vecs.shape[0]
                ind = np.broadcast_to(ind_vecs, (*batch, n_inds, d_model)).copy()
                hidden = block.multihead_attn1(
                    ind,
                    out[..., :train_size, :],  # ty: ignore[invalid-argument-type]
                    out[..., :train_size, :],  # ty: ignore[invalid-argument-type]
                )
                _, k_proj, v_proj = block.multihead_attn2(
                    out,  # ty: ignore[invalid-argument-type]
                    hidden,  # ty: ignore[invalid-argument-type]
                    hidden,  # ty: ignore[invalid-argument-type]
                    need_kv=True,
                )
                cache_list[idx] = (k_proj, v_proj)
                out = block.multihead_attn2(
                    out,  # ty: ignore[invalid-argument-type]
                    hidden,  # ty: ignore[invalid-argument-type]
                    hidden,  # ty: ignore[invalid-argument-type]
                )
            if skip_mask.any():
                out[skip_mask] = SKIP_VALUE  # ty: ignore[invalid-assignment]
        return out  # ty: ignore[invalid-return-type]


class Encoder:
    """Stack of attention blocks with optional RoPE and KV caching."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        ssmax: str | bool = False,
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: float = 100000.0,
        rope_interleaved: bool = True,
    ) -> None:
        self.rope = (
            RotaryEmbedding(
                dim=d_model // nhead,
                theta=rope_base,
                interleaved=rope_interleaved,
            )
            if use_rope
            else None
        )
        self.blocks = [
            AttentionBlock(
                d_model, nhead, dim_feedforward, ssmax=ssmax, bias_free_ln=bias_free_ln
            )
            for _ in range(num_blocks)
        ]

    def __call__(self, src: np.ndarray, train_size: int | None = None) -> np.ndarray:
        out = src
        for block in self.blocks:
            out = block(out, train_size=train_size, rope=self.rope)  # ty: ignore[invalid-argument-type]
        return out  # ty: ignore[invalid-return-type]

    def forward_with_cache(
        self,
        src: np.ndarray,
        cache_list: list,
        train_size: int,
        use_cache: bool,
    ) -> np.ndarray:
        out = src
        for idx, block in enumerate(self.blocks):
            if use_cache:
                k, v = cache_list[idx]
                out = block(out, cached_kv=(k, v), rope=self.rope)  # ty: ignore[invalid-argument-type]
            else:
                out, k_proj, v_proj = block(
                    out,  # ty: ignore[invalid-argument-type]
                    train_size=train_size,
                    rope=self.rope,
                    need_kv=True,
                )
                cache_list[idx] = (k_proj, v_proj)
        return out  # ty: ignore[invalid-return-type]


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


class SkippableLinear:
    """Linear layer preserving the skip marker for fully-masked inputs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.skip_value = SKIP_VALUE
        self.w: np.ndarray = np.zeros((in_features, out_features), dtype=np.float32)
        self.b: np.ndarray = np.zeros(out_features, dtype=np.float32)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        out = src @ self.w + self.b
        skip_mask = (src == self.skip_value).all(axis=-1)
        if skip_mask.any():
            out = np.where(
                skip_mask[..., None], np.full_like(out, self.skip_value), out
            )
        return out


class ColEmbedding:
    """Distribution-aware column-wise embedding."""

    def __init__(self, cfg: TabICLConfig) -> None:
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.feature_group_size = cfg.col_feature_group_size
        self.target_aware = cfg.col_target_aware
        self.max_classes = cfg.max_classes
        self.in_linear = SkippableLinear(cfg.col_feature_group_size, cfg.embed_dim)
        self.tf_col = SetTransformer(
            num_blocks=cfg.col_num_blocks,
            d_model=cfg.embed_dim,
            nhead=cfg.col_nhead,
            dim_feedforward=cfg.col_dim_feedforward,
            num_inds=cfg.col_num_inds,
            ssmax=cfg.col_ssmax,
            bias_free_ln=cfg.bias_free_ln,
        )
        # Regression checkpoints use nn.Linear(1, embed_dim) targets.
        y_in = cfg.max_classes if cfg.max_classes > 0 else 1
        self.y_encoder_w: np.ndarray = np.zeros((y_in, cfg.embed_dim), dtype=np.float32)
        self.y_encoder_b: np.ndarray = np.zeros(cfg.embed_dim, dtype=np.float32)

    def feature_grouping(self, X: np.ndarray) -> np.ndarray:
        size = self.feature_group_size
        _, _, H = X.shape
        idxs = np.arange(H)
        return np.stack([X[:, :, (idxs + 2**i) % H] for i in range(size)], axis=-1)

    def _compute_embeddings(
        self, features: np.ndarray, train_size: int, y_train: np.ndarray
    ) -> np.ndarray:
        src = features @ self.in_linear.w + self.in_linear.b
        if not self.target_aware:
            return self.tf_col(src, train_size=train_size)

        num_classes = int(np.max(y_train) + 1)
        needs_mixed_radix = self.max_classes > 0 and num_classes > self.max_classes
        if not needs_mixed_radix:
            if self.max_classes > 0:
                one_hot = np.eye(self.max_classes, dtype=np.float32)[
                    y_train.astype(np.int64)
                ]
                y_emb = one_hot @ self.y_encoder_w + self.y_encoder_b
            else:
                y_emb = (
                    y_train[..., None].astype(np.float32) @ self.y_encoder_w
                    + self.y_encoder_b
                )
            src[..., :train_size, :] = src[..., :train_size, :] + y_emb
            return self.tf_col(src, train_size=train_size)

        bases = compute_mixed_radix_bases(num_classes, self.max_classes)
        accum = np.zeros_like(src)
        for digit_idx in range(len(bases)):
            y_digit = extract_mixed_radix_digit(y_train, digit_idx, bases)
            one_hot = np.eye(self.max_classes, dtype=np.float32)[
                y_digit.astype(np.int64)
            ]
            y_emb = one_hot @ self.y_encoder_w + self.y_encoder_b
            src_with_y = src.copy()
            src_with_y[..., :train_size, :] = src_with_y[..., :train_size, :] + y_emb
            accum = accum + self.tf_col(src_with_y, train_size=train_size)
        return accum / len(bases)

    def __call__(
        self, X: np.ndarray, y_train: np.ndarray, train_size: int
    ) -> np.ndarray:
        """(B, T, H) x (B, train_size) -> (B, T, G+C, E)."""
        reserve_cls = self.cfg.row_num_cls
        Xg = self.feature_grouping(X)
        pad = np.full(
            (Xg.shape[0], Xg.shape[1], reserve_cls, Xg.shape[-1]),
            SKIP_VALUE,
            dtype=Xg.dtype,
        )
        Xg = np.concatenate([pad, Xg], axis=2)
        features = Xg.transpose(0, 2, 1, 3)
        # Expand labels across every feature-group token; the last dim stays
        # ``train_size`` so it aligns with ``src[..., :train_size, :]``.
        y_exp = np.broadcast_to(
            y_train[:, None, :],
            (Xg.shape[0], features.shape[1], y_train.shape[-1]),
        )
        embeddings = self._compute_embeddings(features, train_size, y_exp)
        return embeddings.transpose(0, 2, 1, 3)


class RowInteraction:
    """Row-wise transformer over column embeddings with CLS tokens."""

    def __init__(self, cfg: TabICLConfig) -> None:
        self.num_cls = cfg.row_num_cls
        self.embed_dim = cfg.embed_dim
        self.cls_tokens: np.ndarray = np.zeros(
            (cfg.row_num_cls, cfg.embed_dim), dtype=np.float32
        )
        # Final LayerNorm before CLS flattening (mirrors ``out_ln``).
        self.out_ln_w: np.ndarray = np.ones(cfg.embed_dim, dtype=np.float32)
        self.out_ln_b: np.ndarray | None = (
            None if cfg.bias_free_ln else np.zeros(cfg.embed_dim, dtype=np.float32)
        )
        self.tf_row = Encoder(
            num_blocks=cfg.row_num_blocks,
            d_model=cfg.embed_dim,
            nhead=cfg.row_nhead,
            dim_feedforward=cfg.col_dim_feedforward,
            bias_free_ln=cfg.bias_free_ln,
            use_rope=True,
            rope_base=cfg.row_rope_base,
            rope_interleaved=cfg.row_rope_interleaved,
        )

    def _aggregate(self, embeddings: np.ndarray) -> np.ndarray:
        rope = self.tf_row.rope
        for block in self.tf_row.blocks[:-1]:
            embeddings = block(embeddings, rope=rope)  # ty: ignore[invalid-assignment]
        last = self.tf_row.blocks[-1]
        cls_outputs = last(
            q=embeddings[..., : self.num_cls, :],
            k=embeddings,
            v=embeddings,
            rope=rope,
        )
        cls_outputs = np.asarray(cls_outputs)
        mean = cls_outputs.mean(axis=-1, keepdims=True)
        var = cls_outputs.var(axis=-1, keepdims=True, ddof=0)
        flat = (cls_outputs - mean) / np.sqrt(var + 1e-5)
        flat = (
            flat * self.out_ln_w
            if self.out_ln_b is None
            else (flat * self.out_ln_w + self.out_ln_b)
        )
        return flat.reshape(flat.shape[0], flat.shape[1], -1)

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        B, T = embeddings.shape[:2]
        cls_tokens = np.broadcast_to(
            self.cls_tokens, (B, T, self.num_cls, self.embed_dim)
        ).astype(embeddings.dtype)
        embeddings = embeddings.copy()
        embeddings[:, :, : self.num_cls] = cls_tokens
        return self._aggregate(embeddings)


class ICLearning:
    """Dataset-wise in-context learning."""

    def __init__(self, cfg: TabICLConfig) -> None:
        self.cfg = cfg
        self.max_classes = cfg.max_classes
        self.ln_w: np.ndarray = np.ones(cfg.icl_dim, dtype=np.float32)
        self.ln_b: np.ndarray | None = (
            None if cfg.bias_free_ln else np.zeros(cfg.icl_dim, dtype=np.float32)
        )
        y_in = cfg.max_classes if cfg.max_classes > 0 else 1
        self.y_encoder_w: np.ndarray = np.zeros((y_in, cfg.icl_dim), dtype=np.float32)
        self.y_encoder_b: np.ndarray = np.zeros(cfg.icl_dim, dtype=np.float32)
        self.decoder_w1: np.ndarray = np.zeros(
            (cfg.icl_dim, cfg.icl_dim * 2), dtype=np.float32
        )
        self.decoder_b1: np.ndarray = np.zeros(cfg.icl_dim * 2, dtype=np.float32)
        self.decoder_w2: np.ndarray = np.zeros(
            (cfg.icl_dim * 2, cfg.out_dim), dtype=np.float32
        )
        self.decoder_b2: np.ndarray = np.zeros(cfg.out_dim, dtype=np.float32)
        self.tf_icl = Encoder(
            num_blocks=cfg.icl_num_blocks,
            d_model=cfg.icl_dim,
            nhead=cfg.icl_nhead,
            dim_feedforward=cfg.icl_dim_feedforward,
            ssmax=cfg.icl_ssmax,
            bias_free_ln=cfg.bias_free_ln,
        )

    def _predict_standard(
        self, R: np.ndarray, y_train: np.ndarray, temperature: float = 0.9
    ) -> np.ndarray:
        train_size = y_train.shape[1]
        if self.max_classes > 0:
            one_hot = np.eye(self.max_classes, dtype=np.float32)[
                y_train.astype(np.int64)
            ]
            Ry = one_hot @ self.y_encoder_w + self.y_encoder_b
        else:
            Ry = (
                y_train[..., None].astype(np.float32) @ self.y_encoder_w
                + self.y_encoder_b
            )
        R = R.copy()
        R[:, :train_size] = R[:, :train_size] + Ry
        src = self.tf_icl(R, train_size=train_size)
        mean = src.mean(axis=-1, keepdims=True)
        var = src.var(axis=-1, keepdims=True, ddof=0)
        src = (src - mean) / np.sqrt(var + 1e-5)
        src = src * self.ln_w if self.ln_b is None else src * self.ln_w + self.ln_b
        h = _gelu(src @ self.decoder_w1 + self.decoder_b1)
        out = h @ self.decoder_w2 + self.decoder_b2
        return out

    def predict_standard(
        self,
        R: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        out = self._predict_standard(R, y_train, temperature)
        train_size = y_train.shape[1]
        if self.max_classes == 0:
            return out[:, train_size:]
        num_classes = int(np.unique(y_train).shape[0])
        out = out[:, train_size:, :num_classes]
        if not return_logits:
            out = _softmax(out / temperature)
        return out

    def predict_hierarchical(
        self,
        root: ClassNode,
        R_test_np: np.ndarray,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Bottom-up combination of per-node predictions."""
        assert root.classes_ is not None
        num_classes = len(root.classes_)
        test_size = R_test_np.shape[0]

        def process(node: ClassNode, r_test: np.ndarray) -> np.ndarray:
            node_r = np.concatenate([node.R, r_test], axis=0)
            if node.is_leaf:
                assert node.y is not None and node.classes_ is not None
                node_y = label_encoding(node.y)
                preds = self.predict_standard(
                    node_r[None],
                    node_y[None],
                    return_logits=False,
                    temperature=temperature,
                ).squeeze(0)
                global_preds = np.zeros((test_size, num_classes), dtype=preds.dtype)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = preds[:, local_idx]
                return global_preds

            node_y = node.group_indices
            assert node_y is not None
            group_probs = self.predict_standard(
                node_r[None],
                node_y[None],
                return_logits=False,
                temperature=temperature,
            ).squeeze(0)
            final = np.zeros((test_size, num_classes), dtype=np.float64)
            for group_idx, child in enumerate(node.child_nodes):
                final += (
                    process(child, r_test) * group_probs[:, group_idx : group_idx + 1]
                )
            return final

        probs = process(root, R_test_np)
        return probs


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #


def _softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _resolve_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map raw checkpoint tensors onto the NumPy model's attribute names.

    Accepts both the shinrin-native names used by :func:`_load_weights` and
    the upstream torch state-dict names stored in converted ``.npz``
    archives (``.weight``/``.bias`` spellings, ``attn.``, ``mlp.0.``,
    ``decoder.0.`` indirections). Linear weights are transposed where the
    NumPy attributes hold ``(in, out)`` layouts.
    """
    import re

    resolved: dict[str, np.ndarray] = {}

    def put(name: str, arr: np.ndarray, transpose: bool = False) -> None:
        resolved[name] = np.asarray(arr.T if transpose else arr, dtype=np.float32)

    for name, arr in params.items():
        if name.endswith("rope.freqs"):
            continue
        resolved.setdefault(name, np.asarray(arr, dtype=np.float32))

        # --- attention blocks: strip the `.attn` module level. ISAB blocks
        # nest as ``...multihead_attnK.attn.*`` while plain Encoder blocks
        # use ``...attn.*`` directly.
        base = None
        rest = None
        m = re.fullmatch(
            r"(?P<p>.+(?:multihead_attn\d)?)\.attn\.(?P<r>(?:in_proj|out_proj|linear|norm|ssmax).+)",
            name,
        )
        if m:
            base = m.group("p")
            rest = m.group("r")
        if base is not None:
            if rest == "in_proj_weight" or rest == "in_proj_bias":
                put(f"{base}.{rest}", arr)
            elif rest == "out_proj.weight":
                put(f"{base}.out_proj.w", arr)
            elif rest == "out_proj.bias":
                put(f"{base}.out_proj.b", arr)
            elif rest == "linear1.weight":
                put(f"{base}.linear1.w", arr, transpose=True)
            elif rest == "linear1.bias":
                put(f"{base}.linear1.b", arr)
            elif rest == "linear2.weight":
                put(f"{base}.linear2.w", arr, transpose=True)
            elif rest == "linear2.bias":
                put(f"{base}.linear2.b", arr)
            elif rest == "norm1.weight":
                put(f"{base}.norm1.w", arr)
            elif rest == "norm1.bias":
                put(f"{base}.norm1.b", arr)
            elif rest == "norm2.weight":
                put(f"{base}.norm2.w", arr)
            elif rest == "norm2.bias":
                put(f"{base}.norm2.b", arr)
            # ssmax sub-layers live under `...attn.ssmax_layer.*`
            sm = re.fullmatch(r"ssmax_layer\.(?P<s>.+)", str(rest))
            if sm:
                sname = sm.group("s")
                _put_ssmax(put, f"{base}.ssmax_layer", sname, arr)
            continue

        # --- FFN / LayerNorm tensors that live directly on the block ---
        m4 = re.fullmatch(
            r"(?P<p>.+)\.(?P<r>linear[12]|norm[12])\.(?P<t>weight|bias)", name
        )
        if m4:
            kind = m4.group("r")
            t = m4.group("t")
            if t == "bias":
                put(f"{m4.group('p')}.{kind}.b", arr)
            elif kind.startswith("linear"):
                put(f"{m4.group('p')}.{kind}.w", arr, transpose=True)
            else:
                put(f"{m4.group('p')}.{kind}.w", arr)
            continue

        # --- col embedder input projection / target encoders ---
        if name == "col_embedder.in_linear.weight":
            put("col_embedder.in_linear.w", arr, transpose=True)
        elif name == "col_embedder.in_linear.bias":
            put("col_embedder.in_linear.b", arr)
        elif name in (
            "col_embedder.y_encoder.weight",
            "icl_predictor.y_encoder.weight",
        ):
            put(name.replace(".weight", ".w"), arr, transpose=True)
        elif name in ("col_embedder.y_encoder.bias", "icl_predictor.y_encoder.bias"):
            put(name.replace(".bias", ".b"), arr)
        elif name == "row_interactor.out_ln.weight":
            put("row_interactor.out_ln.w", arr)
        elif name == "row_interactor.out_ln.bias":
            put("row_interactor.out_ln.b", arr)
        elif name == "icl_predictor.ln.weight":
            put("icl_predictor.ln.w", arr)
        elif name == "icl_predictor.ln.bias":
            put("icl_predictor.ln.b", arr)
        elif name == "icl_predictor.decoder.0.weight":
            put("icl_predictor.decoder.w1", arr, transpose=True)
        elif name == "icl_predictor.decoder.0.bias":
            put("icl_predictor.decoder.b1", arr)
        elif name == "icl_predictor.decoder.2.weight":
            put("icl_predictor.decoder.w2", arr, transpose=True)
        elif name == "icl_predictor.decoder.2.bias":
            put("icl_predictor.decoder.b2", arr)

    return resolved


def _put_ssmax(put, prefix: str, tail: str, arr: np.ndarray) -> None:
    """Resolve SSMax variant parameter names into attribute names."""
    # torch nn.Linear stores weights as (out, in); NumPy attrs are (in, out).
    if tail == "scales":
        put(f"{prefix}.scales", arr)
    elif tail == "mlp.0.weight":
        put(f"{prefix}.w1", arr, transpose=True)
    elif tail == "mlp.0.bias":
        put(f"{prefix}.b1", arr)
    elif tail == "mlp.2.weight":
        put(f"{prefix}.w2", arr, transpose=True)
    elif tail == "mlp.2.bias":
        put(f"{prefix}.b2", arr)
    elif tail == "base_mlp.0.weight":
        put(f"{prefix}.base_w1", arr, transpose=True)
    elif tail == "base_mlp.0.bias":
        put(f"{prefix}.base_b1", arr)
    elif tail == "base_mlp.2.weight":
        put(f"{prefix}.base_w2", arr, transpose=True)
    elif tail == "base_mlp.2.bias":
        put(f"{prefix}.base_b2", arr)
    elif tail == "query_mlp.0.weight":
        put(f"{prefix}.query_w1", arr, transpose=True)
    elif tail == "query_mlp.0.bias":
        put(f"{prefix}.query_b1", arr)
    elif tail == "query_mlp.2.weight":
        put(f"{prefix}.query_w2", arr, transpose=True)
    elif tail == "query_mlp.2.bias":
        put(f"{prefix}.query_b2", arr)


def _load_weights(net, params: dict[str, np.ndarray], config: TabICLConfig) -> None:
    """Load state-dict tensors into the NumPy model."""
    cfg = config
    params = _resolve_params(params)

    def load(name: str, attr: np.ndarray, shape: tuple[int, ...]) -> None:
        if name in params:
            arr = np.asarray(params[name], dtype=np.float32)
            if arr.shape != shape:
                arr = arr.reshape(shape)
            attr[:] = arr

    # ColEmbedding
    load(
        "col_embedder.in_linear.w",
        net.col_embedder.in_linear.w,
        (cfg.col_feature_group_size, cfg.embed_dim),
    )
    load("col_embedder.in_linear.b", net.col_embedder.in_linear.b, (cfg.embed_dim,))
    load(
        "col_embedder.y_encoder.w",
        net.col_embedder.y_encoder_w,
        net.col_embedder.y_encoder_w.shape,
    )
    load("col_embedder.y_encoder.b", net.col_embedder.y_encoder_b, (cfg.embed_dim,))
    _load_settransformer(net.col_embedder.tf_col, params, "col_embedder.tf_col")

    # RowInteraction
    load(
        "row_interactor.cls_tokens",
        net.row_interactor.cls_tokens,
        (cfg.row_num_cls, cfg.embed_dim),
    )
    load(
        "row_interactor.out_ln.w",
        net.row_interactor.out_ln_w,
        (cfg.embed_dim,),
    )
    if net.row_interactor.out_ln_b is not None:
        load(
            "row_interactor.out_ln.b",
            net.row_interactor.out_ln_b,
            (cfg.embed_dim,),
        )
    _load_encoder(net.row_interactor.tf_row, params, "row_interactor.tf_row")

    # ICLearning
    load("icl_predictor.ln.w", net.icl_predictor.ln_w, (cfg.icl_dim,))
    if net.icl_predictor.ln_b is not None:
        load("icl_predictor.ln.b", net.icl_predictor.ln_b, (cfg.icl_dim,))
    load(
        "icl_predictor.y_encoder.w",
        net.icl_predictor.y_encoder_w,
        net.icl_predictor.y_encoder_w.shape,
    )
    load("icl_predictor.y_encoder.b", net.icl_predictor.y_encoder_b, (cfg.icl_dim,))
    load(
        "icl_predictor.decoder.w1",
        net.icl_predictor.decoder_w1,
        (cfg.icl_dim, cfg.icl_dim * 2),
    )
    load("icl_predictor.decoder.b1", net.icl_predictor.decoder_b1, (cfg.icl_dim * 2,))
    load(
        "icl_predictor.decoder.w2",
        net.icl_predictor.decoder_w2,
        (cfg.icl_dim * 2, cfg.out_dim),
    )
    load("icl_predictor.decoder.b2", net.icl_predictor.decoder_b2, (cfg.out_dim,))
    _load_encoder(net.icl_predictor.tf_icl, params, "icl_predictor.tf_icl")


def _load_settransformer(st: SetTransformer, params: dict, prefix: str) -> None:
    for i, block in enumerate(st.blocks):
        _load_induced_block(block, params, f"{prefix}.blocks.{i}")


def _load_encoder(enc: Encoder, params: dict, prefix: str) -> None:
    for i, block in enumerate(enc.blocks):
        _load_attention_block(block, params, f"{prefix}.blocks.{i}")


def _load_induced_block(
    block: InducedSelfAttentionBlock, params: dict, prefix: str
) -> None:
    key = f"{prefix}.ind_vectors"
    if key in params:
        block.ind_vectors = np.asarray(params[key], dtype=np.float32)
    _load_attention_block(block.multihead_attn1, params, f"{prefix}.multihead_attn1")
    _load_attention_block(block.multihead_attn2, params, f"{prefix}.multihead_attn2")


def _load_attention_block(block: AttentionBlock, params: dict, prefix: str) -> None:
    attn = block.attn
    key = f"{prefix}.in_proj_weight"
    if key in params:
        attn.in_proj_weight = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.in_proj_bias"
    if key in params:
        attn.in_proj_bias = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.out_proj.w"
    if key in params:
        attn.out_proj_w = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.out_proj.b"
    if key in params:
        attn.out_proj_b = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.linear1.w"
    if key in params:
        block.linear1_w = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.linear1.b"
    if key in params:
        block.linear1_b = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.linear2.w"
    if key in params:
        block.linear2_w = np.asarray(params[key], dtype=np.float32)
    key = f"{prefix}.linear2.b"
    if key in params:
        block.linear2_b = np.asarray(params[key], dtype=np.float32)
    for norm_idx in (1, 2):
        w_key = f"{prefix}.norm{norm_idx}.w"
        if w_key in params:
            setattr(
                block,
                f"norm{norm_idx}_w",
                np.asarray(params[w_key], dtype=np.float32),
            )
        b_key = f"{prefix}.norm{norm_idx}.b"
        if b_key in params and getattr(block, f"norm{norm_idx}_b") is not None:
            setattr(
                block,
                f"norm{norm_idx}_b",
                np.asarray(params[b_key], dtype=np.float32),
            )
    _load_ssmax(attn.ssmax_layer, params, f"{prefix}.ssmax_layer")


def _load_ssmax(layer, params: dict, prefix: str) -> None:
    """Load SSMax scale parameters when present."""
    if layer is None:
        return
    simple = {
        "scales": "scales",
        "w1": "w1",
        "b1": "b1",
        "w2": "w2",
        "b2": "b2",
        "base_w1": "base_w1",
        "base_b1": "base_b1",
        "base_w2": "base_w2",
        "base_b2": "base_b2",
        "query_w1": "query_w1",
        "query_b1": "query_b1",
        "query_w2": "query_w2",
        "query_b2": "query_b2",
    }
    for suffix, attr in simple.items():
        key = f"{prefix}.{suffix}"
        if key in params:
            setattr(layer, attr, np.asarray(params[key], dtype=np.float32))


class _TabICLNet:
    """Lightweight container holding the three stages."""

    def __init__(self, cfg: TabICLConfig) -> None:
        self.col_embedder = ColEmbedding(cfg)
        self.row_interactor = RowInteraction(cfg)
        self.icl_predictor = ICLearning(cfg)


class TabICLNumPyModel:
    """Inference wrapper that loads converted weights into the NumPy architecture."""

    def __init__(self, config: TabICLConfig, params: dict[str, np.ndarray]) -> None:
        self.config = config
        self.net = _TabICLNet(config)
        _load_weights(self.net, params, config)

    def _build_col_cache(
        self,
        cfg: TabICLConfig,
        x: np.ndarray,
        y: np.ndarray,
        train_size: int,
    ) -> tuple[np.ndarray, list]:
        """Build ColEmbedding output and KV cache (uses the loaded weights)."""
        col = self.net.col_embedder
        Xg = col.feature_grouping(x)
        pad = np.full(
            (Xg.shape[0], Xg.shape[1], cfg.row_num_cls, Xg.shape[-1]),
            SKIP_VALUE,
            dtype=Xg.dtype,
        )
        Xg = np.concatenate([pad, Xg], axis=2)
        features = Xg.transpose(0, 2, 1, 3)
        y_exp = np.broadcast_to(
            y[:, None, :], (Xg.shape[0], features.shape[1], y.shape[-1])
        )
        src = features @ col.in_linear.w + col.in_linear.b
        if col.target_aware:
            if cfg.max_classes > 0:
                one_hot = np.eye(cfg.max_classes, dtype=np.float32)[
                    y_exp.astype(np.int64)
                ]
                src[..., :train_size, :] = (
                    src[..., :train_size, :]
                    + one_hot @ col.y_encoder_w
                    + col.y_encoder_b
                )
            else:
                src[..., :train_size, :] = (
                    src[..., :train_size, :]
                    + y_exp[..., None].astype(np.float32) @ col.y_encoder_w
                    + col.y_encoder_b
                )
        col_cache: list = [None] * cfg.col_num_blocks
        embeddings = col.tf_col.forward_with_cache(
            src, col_cache, train_size, use_cache=False
        )
        embeddings = embeddings.transpose(0, 2, 1, 3)
        return embeddings, col_cache

    def _build_icl_cache(
        self,
        cfg: TabICLConfig,
        R: np.ndarray,
        y: np.ndarray,
        train_size: int,
    ) -> list:
        """Build ICL K/V cache (uses the loaded weights)."""
        icl = self.net.icl_predictor
        if cfg.max_classes > 0:
            one_hot = np.eye(cfg.max_classes, dtype=np.float32)[y.astype(np.int64)]
            R = R.copy()
            R[:, :train_size] = (
                R[:, :train_size] + one_hot @ icl.y_encoder_w + icl.y_encoder_b
            )
        else:
            R = R.copy()
            R[:, :train_size] = (
                R[:, :train_size]
                + y[..., None].astype(np.float32) @ icl.y_encoder_w
                + icl.y_encoder_b
            )
        icl_cache: list = [None] * cfg.icl_num_blocks
        icl.tf_icl.forward_with_cache(R, icl_cache, train_size, use_cache=False)
        return icl_cache

    def representations(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Column embedding + row interaction. Returns (1, T, D) array."""
        x = np.ascontiguousarray(X, dtype=np.float32)[None]
        y = np.asarray(y_train, dtype=np.float32)[None]
        train_size = y.shape[1]
        emb = self.net.col_embedder(x, y, train_size)
        rep = self.net.row_interactor(emb)
        return rep

    def predict_from_representations(
        self,
        R: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Run the ICL stage on row representations."""
        cfg = self.config
        y = np.asarray(y_train, dtype=np.float32)
        if y.ndim == 1:
            y = y[None]
        train_size = y.shape[1]
        num_classes = len(np.unique(y_train)) if cfg.max_classes > 0 else 0

        if cfg.max_classes == 0 or num_classes <= cfg.max_classes:
            out = self.net.icl_predictor.predict_standard(
                R, y, return_logits=return_logits, temperature=temperature
            )
            # Match the torch wrapper, which drops the singleton batch dim.
            return out[0] if out.ndim == 3 else out

        root = fit_hierarchical_tree(R[0, :train_size], y_train, cfg.max_classes)
        probs = self.net.icl_predictor.predict_hierarchical(
            root, R[0, train_size:], temperature=temperature
        )
        if return_logits:
            probs = temperature * np.log(probs + 1e-6)
        return probs

    def build_cache(self, X: np.ndarray, y_train: np.ndarray) -> dict:
        """Pre-compute col-stage and ICL-stage K/V caches."""
        cfg = self.config
        x = np.ascontiguousarray(X, dtype=np.float32)[None]
        y = np.asarray(y_train, dtype=np.float32)[None]
        train_size = y.shape[1]

        embeddings, col_cache = self._build_col_cache(cfg, x, y, train_size)
        rep = self.net.row_interactor(embeddings)
        icl_cache = self._build_icl_cache(cfg, rep, y, train_size)
        num_classes = len(np.unique(y_train)) if cfg.max_classes > 0 else 0
        return {
            "col": col_cache,
            "icl": icl_cache,
            "train_size": train_size,
            "num_classes": num_classes,
        }

    def predict_with_cache(
        self,
        X_test: np.ndarray,
        cache: dict,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Predict using pre-computed caches."""
        cfg = self.config
        x = np.ascontiguousarray(X_test, dtype=np.float32)[None]
        col = self.net.col_embedder

        Xg = col.feature_grouping(x)
        pad = np.full(
            (Xg.shape[0], Xg.shape[1], cfg.row_num_cls, Xg.shape[-1]),
            SKIP_VALUE,
            dtype=Xg.dtype,
        )
        Xg = np.concatenate([pad, Xg], axis=2)
        features = Xg.transpose(0, 2, 1, 3)
        out = features @ col.in_linear.w + col.in_linear.b
        for idx, block in enumerate(col.tf_col.blocks):
            hidden_k, hidden_v = cache["col"][idx]
            out = block.multihead_attn2(
                out,  # ty: ignore[invalid-argument-type]
                cached_kv=(hidden_k, hidden_v),
            )
        embeddings = out.transpose(0, 2, 1, 3)  # ty: ignore[unresolved-attribute]
        rep = self.net.row_interactor(embeddings)

        icl = self.net.icl_predictor
        r_out = rep
        for idx, block in enumerate(icl.tf_icl.blocks):
            k, v = cache["icl"][idx]
            r_out = block(r_out, cached_kv=(k, v))  # ty: ignore[invalid-argument-type]
        decoded = _gelu(r_out @ icl.decoder_w1 + icl.decoder_b1)
        decoded = decoded @ icl.decoder_w2 + icl.decoder_b2
        decoded = decoded.squeeze(0)

        if cfg.max_classes == 0:
            return decoded
        num_classes = int(cache.get("num_classes", decoded.shape[-1]))
        decoded = decoded[:, :num_classes]
        if not return_logits:
            decoded = _softmax(decoded, temperature=temperature)
        return decoded

    def forward(
        self,
        X: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Full pipeline: preprocessing-free numeric forward pass."""
        R = self.representations(X, y_train)
        return self.predict_from_representations(
            R, y_train, return_logits=return_logits, temperature=temperature
        )
