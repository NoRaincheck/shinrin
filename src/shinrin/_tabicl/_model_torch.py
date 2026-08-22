"""Own PyTorch implementation of the TabICLv2 architecture.

Weights are loaded from converted checkpoints (see ``_checkpoint``) under the
original state-dict names, so this module doubles as the numeric reference
for the NumPy and Mojo backends. Inference-only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


class RotaryEmbedding(nn.Module):
    """Rotary positional encoding with a (non-trainable) ``freqs`` buffer."""

    def __init__(
        self, dim: int, theta: float = 10000.0, interleaved: bool = True
    ) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_parameter("freqs", nn.Parameter(freqs, requires_grad=False))
        self.interleaved = interleaved

    def rotate(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """Rotate ``x`` of shape (..., n_heads, seq_len, head_dim)."""
        seq_len = x.shape[-2]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        raw_freqs = self.freqs if isinstance(self.freqs, Tensor) else self.freqs.data
        pos = positions.to(torch.float32)
        frq = raw_freqs.to(torch.float32)
        freqs = torch.outer(
            pos,
            frq,  # ty: ignore[invalid-argument-type]
        )  # (T, hd/2)
        original_dtype = x.dtype
        x_f = x.to(torch.float32)
        if self.interleaved:
            freqs = torch.cat([freqs, freqs], dim=-1)  # (T, hd)
            cos, sin = freqs.cos(), freqs.sin()
            x1, x2 = x_f[..., 0::2], x_f[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            out = x_f * cos + rotated * sin
        else:
            cos, sin = freqs.cos(), freqs.sin()  # (T, hd/2)
            half = x_f.shape[-1] // 2
            x1, x2 = x_f[..., :half], x_f[..., half:]
            out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out.to(original_dtype)


class SSMax(nn.Module):
    """Scalable softmax with learnable per-head scale."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.scales = nn.Parameter(torch.ones(num_heads))

    def forward(self, q: Tensor, n: int) -> Tensor:
        logn = torch.tensor(math.log(max(n, 1)), device=q.device, dtype=q.dtype)
        return q * (self.scales.view(1, -1, 1, 1) * logn)


class SSMaxMLP(nn.Module):
    """Scalable softmax with an MLP mapping log(n) to scales."""

    def __init__(
        self,
        num_heads: int,
        n_hidden: int = 64,
        elementwise: bool = False,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.elementwise = elementwise
        if elementwise:
            if head_dim is None:
                raise ValueError("head_dim is required for elementwise SSMaxMLP")
            out_dim = num_heads * head_dim
        else:
            out_dim = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, out_dim)
        )
        self.num_heads = num_heads

    def forward(self, q: Tensor, n: int) -> Tensor:
        logn = torch.tensor(
            math.log(max(n, 1)), device=q.device, dtype=q.dtype
        ).reshape(1, 1)
        scales = self.mlp(logn)
        if self.elementwise:
            scales = scales.view(1, self.num_heads, 1, q.shape[-1])
        else:
            scales = scales.view(1, self.num_heads, 1, 1)
        return q * scales


class QASSMaxMLP(nn.Module):
    """Query-aware scalable softmax with base and query MLPs."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
        elementwise: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.elementwise = elementwise
        base_out = num_heads * head_dim if elementwise else num_heads
        query_out = head_dim if elementwise else 1
        self.base_mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, query_out)
        )

    def forward(self, q: Tensor, n: int) -> Tensor:
        logn = torch.tensor(
            math.log(max(n, 1)), device=q.device, dtype=q.dtype
        ).reshape(1, 1)
        if self.elementwise:
            base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, self.head_dim)
        else:
            base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, 1)
        modulation = 1 + torch.tanh(self.query_mlp(q))
        return q * (base_scales * modulation)


def create_ssmax_layer(
    ssmax_type: str | bool, num_heads: int, embed_dim: int
) -> nn.Module | None:
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


def _attention(
    attn: MultiheadAttention,
    q_in: Tensor,
    k_in: Tensor | None,
    v_in: Tensor | None,
    train_size: int | None = None,
    rope: RotaryEmbedding | None = None,
    cached_kv: tuple[Tensor, Tensor] | None = None,
    need_kv: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Multi-head attention with packed projection, RoPE, ssmax, and KV cache."""
    embed_dim = q_in.shape[-1]
    num_heads, head_dim = attn.num_heads, attn.head_dim
    w, b = attn.in_proj_weight, attn.in_proj_bias

    if cached_kv is None:
        assert k_in is not None and v_in is not None
        q = F.linear(q_in, w[:embed_dim], b[:embed_dim])
        k = F.linear(k_in, w[embed_dim : 2 * embed_dim], b[embed_dim : 2 * embed_dim])
        v = F.linear(v_in, w[2 * embed_dim :], b[2 * embed_dim :])
        q = q.view(*q.shape[:-1], num_heads, head_dim).transpose(-3, -2)
        k = k.view(*k.shape[:-1], num_heads, head_dim).transpose(-3, -2)
        v = v.view(*v.shape[:-1], num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate(q)
            k = rope.rotate(k)
    else:
        k, v = cached_kv
        q = F.linear(q_in, w[:embed_dim], b[:embed_dim])
        q = q.view(*q.shape[:-1], num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate(q)

    src_len = k.shape[-2]
    if attn.ssmax_layer is not None:
        q = attn.ssmax_layer(q, src_len)

    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    batch_shape = tuple(q_in.shape[:-1])
    out = out.transpose(-3, -2).contiguous().view(*batch_shape, embed_dim)
    out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)

    if need_kv and cached_kv is None:
        return out, k, v
    return out


class MultiheadAttention(nn.Module):
    """Own multi-head attention holding the packed in-projection parameters."""

    def __init__(
        self, embed_dim: int, num_heads: int, ssmax: str | bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ssmax_layer = create_ssmax_layer(ssmax, num_heads, embed_dim)

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        cached_kv: tuple[Tensor, Tensor] | None = None,
        need_kv: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        return _attention(
            self, q, k, v, rope=rope, cached_kv=cached_kv, need_kv=need_kv
        )


class AttentionBlock(nn.Module):
    """Pre-norm attention + feedforward block (norm_first architecture)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        ssmax: str | bool = False,
        bias_free_ln: bool = False,
    ) -> None:
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, ssmax)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.norm2 = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.norm_first = True

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        train_size: int | None = None,
        rope: RotaryEmbedding | None = None,
        cached_kv: tuple[Tensor, Tensor] | None = None,
        need_kv: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if train_size is not None:
            k = q[..., :train_size, :]
            v = k
        else:
            if k is None:
                k = q
            if v is None:
                v = k

        if cached_kv is not None:
            attn = self.attn(self.norm1(q), cached_kv=cached_kv, rope=rope)
            x = q + attn
        elif self.norm_first:
            qn = self.norm1(q)
            kn = qn if k is q else self.norm1(k)
            vn = kn if v is k else self.norm1(v)
            result = self.attn(qn, kn, vn, rope=rope, need_kv=need_kv)
            if need_kv:
                attn, k_proj, v_proj = result
            else:
                attn = result
            x = q + attn
        else:  # pragma: no cover - checkpoints use pre-norm
            result = self.attn(q, k, v, rope=rope, need_kv=need_kv)
            if need_kv:
                attn, k_proj, v_proj = result
            else:
                attn = result
            x = self.norm1(q + attn)

        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        if need_kv and cached_kv is None:
            return x, k_proj, v_proj
        return x


class InducedSelfAttentionBlock(nn.Module):
    """Two-stage induced self-attention (set transformer building block)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        ssmax: str | bool = False,
        bias_free_ln: bool = False,
    ) -> None:
        super().__init__()
        self.multihead_attn1 = AttentionBlock(
            d_model, nhead, dim_feedforward, ssmax, bias_free_ln
        )
        self.multihead_attn2 = AttentionBlock(
            d_model, nhead, dim_feedforward, False, bias_free_ln
        )
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        self.skip_value = SKIP_VALUE

    def induced_attention(self, src: Tensor, train_size: int | None = None) -> Tensor:
        *batch, _, d_model = src.shape
        ind = self.ind_vectors.expand(*batch, self.ind_vectors.shape[0], d_model)
        if train_size is None:
            hidden = self.multihead_attn1(ind, src, src)
        else:
            hidden = self.multihead_attn1(
                ind, src[..., :train_size, :], src[..., :train_size, :]
            )
        return self.multihead_attn2(src, hidden, hidden)

    def forward(self, src: Tensor, train_size: int | None = None) -> Tensor:
        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if not skip_mask.any():
            return self.induced_attention(src, train_size)
        if bool(skip_mask.all()):
            return torch.full_like(src, self.skip_value)
        out = torch.empty_like(src)
        out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
        out[skip_mask] = self.skip_value
        return out


class SetTransformer(nn.Module):
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
        super().__init__()
        self.blocks = nn.ModuleList(
            InducedSelfAttentionBlock(
                d_model, nhead, dim_feedforward, num_inds, ssmax, bias_free_ln
            )
            for i in range(num_blocks)
        )

    def forward(self, src: Tensor, train_size: int | None = None) -> Tensor:
        out = src
        for block in self.blocks:
            out = block(out, train_size)
        return out

    def forward_with_cache(
        self, src: Tensor, cache_list: list, train_size: int, use_cache: bool
    ) -> Tensor:
        """Cache-aware forward. Like upstream, the cache path does not exclude
        skipped (all -100) columns from stage-1 keys; skip values are restored
        between blocks."""
        skip_mask = (src == SKIP_VALUE).all(dim=(-2, -1))
        out = src
        for idx, block in enumerate(self.blocks):
            if use_cache:
                hidden_k, hidden_v = cache_list[idx]
                out = block.multihead_attn2(  # ty: ignore[call-non-callable]
                    out, cached_kv=(hidden_k, hidden_v)
                )
            else:
                *batch, _, d_model = out.shape
                ind_vecs = block.ind_vectors
                n_inds = ind_vecs.shape[0]  # ty: ignore[not-subscriptable]
                ind = ind_vecs.expand(  # ty: ignore[call-non-callable, no-matching-overload]
                    *batch, n_inds, d_model
                )
                hidden = block.multihead_attn1(  # ty: ignore[call-non-callable]
                    ind, out[..., :train_size, :], out[..., :train_size, :]
                )
                _, k_proj, v_proj = block.multihead_attn2(  # ty: ignore[call-non-callable]
                    out, hidden, hidden, need_kv=True
                )
                cache_list[idx] = (k_proj, v_proj)
                out = block.multihead_attn2(  # ty: ignore[call-non-callable]
                    out, hidden, hidden
                )
            if skip_mask.any():
                out[skip_mask] = SKIP_VALUE
        return out


class Encoder(nn.Module):
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
        super().__init__()
        self.rope = (
            RotaryEmbedding(
                dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved
            )
            if use_rope
            else None
        )
        self.blocks = nn.ModuleList(
            AttentionBlock(
                d_model, nhead, dim_feedforward, ssmax=ssmax, bias_free_ln=bias_free_ln
            )
            for i in range(num_blocks)
        )

    def forward(self, src: Tensor, train_size: int | None = None) -> Tensor:
        out = src
        for block in self.blocks:
            out = block(out, train_size=train_size, rope=self.rope)
        return out

    def forward_with_cache(
        self,
        src: Tensor,
        cache_list: list,
        train_size: int,
        use_cache: bool,
    ) -> Tensor:
        out = src
        for idx, block in enumerate(self.blocks):
            if use_cache:
                k, v = cache_list[idx]
                out = block(out, cached_kv=(k, v), rope=self.rope)
            else:
                out, k_proj, v_proj = block(
                    out, train_size=train_size, rope=self.rope, need_kv=True
                )
                cache_list[idx] = (k_proj, v_proj)
        return out


class OneHotAndLinear(nn.Linear):
    """One-hot encode integer labels and project linearly."""

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes

    def forward(self, src: Tensor) -> Tensor:  # ty: ignore[invalid-method-override]
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot.float(), self.weight, self.bias).to(src.dtype)


class SkippableLinear(nn.Linear):
    """Linear layer preserving the skip marker for fully-masked inputs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        self.skip_value = SKIP_VALUE

    def forward(self, src: Tensor) -> Tensor:  # ty: ignore[invalid-method-override]
        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out = torch.where(
                skip_mask.unsqueeze(-1), torch.full_like(out, self.skip_value), out
            )
        return out


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


class ColEmbedding(nn.Module):
    """Distribution-aware column-wise embedding (feature-grouped, target-aware)."""

    def __init__(self, cfg: TabICLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.feature_group_size = cfg.col_feature_group_size
        self.target_aware = cfg.col_target_aware
        self.max_classes = cfg.max_classes
        self.mixed_radix_ensemble = True
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
        if cfg.col_affine:
            self.out_w = SkippableLinear(cfg.embed_dim, cfg.embed_dim)
            self.out_b = SkippableLinear(cfg.embed_dim, cfg.embed_dim)
            self.ln_w = nn.LayerNorm(cfg.embed_dim)
            self.ln_b = nn.LayerNorm(cfg.embed_dim)
        if cfg.col_target_aware:
            if cfg.max_classes > 0:
                self.y_encoder = OneHotAndLinear(cfg.max_classes, cfg.embed_dim)
            else:
                self.y_encoder = nn.Linear(1, cfg.embed_dim)

    def feature_grouping(self, X: Tensor) -> Tensor:
        """(B, T, H) -> (B, T, G, group_size) via circular permutation."""
        size = self.feature_group_size
        _, _, H = X.shape
        idxs = torch.arange(H, device=X.device)
        return torch.stack([X[:, :, (idxs + 2**i) % H] for i in range(size)], dim=-1)

    def _compute_embeddings(
        self, features: Tensor, train_size: int, y_train: Tensor | None
    ) -> Tensor:
        cfg = self.cfg
        src = self.in_linear(features)
        if not self.target_aware:
            return self.tf_col(src, train_size=train_size)

        assert y_train is not None
        num_classes = int(y_train.max().item()) + 1
        needs_mixed_radix = cfg.max_classes > 0 and num_classes > cfg.max_classes

        if not needs_mixed_radix:
            if cfg.max_classes > 0:
                y_emb = self.y_encoder(y_train.float())
            else:
                y_emb = self.y_encoder(y_train.unsqueeze(-1).float())
            src[..., :train_size, :] = src[..., :train_size, :] + y_emb
            return self.tf_col(src, train_size=train_size)

        bases = compute_mixed_radix_bases(num_classes, cfg.max_classes)
        accum = torch.zeros_like(src)
        for digit_idx in range(len(bases)):
            y_digit = extract_mixed_radix_digit(y_train.cpu().numpy(), digit_idx, bases)
            y_emb = self.y_encoder(torch.from_numpy(y_digit).to(src.device).float())
            src_with_y = src.clone()
            src_with_y[..., :train_size, :] = src_with_y[..., :train_size, :] + y_emb
            accum = accum + self.tf_col(src_with_y, train_size=train_size)
        return accum / len(bases)

    def forward(self, X: Tensor, y_train: Tensor, train_size: int) -> Tensor:
        """(B, T, H) x (B, train_size) -> (B, T, G+C, E)."""
        reserve_cls = self.cfg.row_num_cls
        Xg = self.feature_grouping(X)
        pad = torch.full(
            (Xg.shape[0], Xg.shape[1], reserve_cls, Xg.shape[-1]),
            SKIP_VALUE,
            device=Xg.device,
            dtype=Xg.dtype,
        )
        Xg = torch.cat([pad, Xg], dim=2)  # (B, T, G+C, s)
        features = Xg.transpose(1, 2)  # (B, G+C, T, s)
        y_exp = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
        embeddings = self._compute_embeddings(features, train_size, y_exp)
        return embeddings.transpose(1, 2)


class RowInteraction(nn.Module):
    """Row-wise transformer over column embeddings with learnable CLS tokens."""

    def __init__(self, cfg: TabICLConfig) -> None:
        super().__init__()
        self.num_cls = cfg.row_num_cls
        self.embed_dim = cfg.embed_dim
        self.cls_tokens = nn.Parameter(torch.empty(cfg.row_num_cls, cfg.embed_dim))
        self.out_ln = nn.LayerNorm(cfg.embed_dim, bias=not cfg.bias_free_ln)
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

    def _aggregate(self, embeddings: Tensor) -> Tensor:
        rope = self.tf_row.rope
        for block in self.tf_row.blocks[:-1]:
            embeddings = block(embeddings, rope=rope)
        last = self.tf_row.blocks[-1]
        cls_outputs = last(
            q=embeddings[..., : self.num_cls, :], k=embeddings, v=embeddings, rope=rope
        )
        return self.out_ln(cls_outputs).flatten(-2)

    def forward(self, embeddings: Tensor) -> Tensor:
        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim).to(
            embeddings.dtype
        )
        embeddings[:, :, : self.num_cls] = cls_tokens
        return self._aggregate(embeddings)


class ICLearning(nn.Module):
    """Dataset-wise in-context learning with standard and hierarchical paths."""

    def __init__(self, cfg: TabICLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_classes = cfg.max_classes
        self.ln = nn.LayerNorm(cfg.icl_dim, bias=not cfg.bias_free_ln)
        if cfg.max_classes > 0:
            self.y_encoder = OneHotAndLinear(cfg.max_classes, cfg.icl_dim)
        else:
            self.y_encoder = nn.Linear(1, cfg.icl_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.icl_dim, cfg.icl_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.icl_dim * 2, cfg.out_dim),
        )
        self.tf_icl = Encoder(
            num_blocks=cfg.icl_num_blocks,
            d_model=cfg.icl_dim,
            nhead=cfg.icl_nhead,
            dim_feedforward=cfg.icl_dim_feedforward,
            ssmax=cfg.icl_ssmax,
            bias_free_ln=cfg.bias_free_ln,
        )

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        train_size = y_train.shape[1]
        if self.max_classes > 0:
            Ry = self.y_encoder(y_train.float())
        else:
            Ry = self.y_encoder(y_train.unsqueeze(-1))
        R = R.clone()
        R[:, :train_size] = R[:, :train_size] + Ry
        src = self.tf_icl(R, train_size=train_size)
        src = self.ln(src)
        return self.decoder(src)

    def predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        temperature: float = 0.9,
    ) -> Tensor:
        out = self._icl_predictions(R, y_train)
        train_size = y_train.shape[1]
        if self.max_classes == 0:
            return out[:, train_size:]
        num_classes = len(torch.unique(y_train[0]))
        out = out[:, train_size:, :num_classes]
        if not return_logits:
            out = torch.softmax(out / temperature, dim=-1)
        return out

    def predict_hierarchical(
        self,
        root: ClassNode,
        R_test_np: np.ndarray,
        device: torch.device,
        temperature: float = 0.9,
    ) -> Tensor:
        """Bottom-up combination of per-node predictions (single member)."""
        assert root.classes_ is not None
        num_classes = len(root.classes_)
        test_size = R_test_np.shape[0]

        def process(node: ClassNode, r_test: np.ndarray) -> np.ndarray:
            node_r = np.concatenate([node.R, r_test], axis=0)
            r_tensor = (
                torch.from_numpy(node_r.astype(np.float32)).unsqueeze(0).to(device)
            )
            if node.is_leaf:
                assert node.y is not None and node.classes_ is not None
                node_y = label_encoding(node.y)
                preds = self.predict_standard(
                    r_tensor,
                    torch.from_numpy(node_y).unsqueeze(0).to(device),
                    temperature=temperature,
                )
                preds = preds.squeeze(0).cpu().numpy()
                global_preds = np.zeros((test_size, num_classes), dtype=preds.dtype)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = preds[:, local_idx]
                return global_preds

            node_y = node.group_indices
            group_probs = (
                self.predict_standard(
                    r_tensor,
                    torch.from_numpy(node_y).unsqueeze(0).to(device),
                    temperature=temperature,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
            final = np.zeros((test_size, num_classes), dtype=np.float64)
            for group_idx, child in enumerate(node.child_nodes):
                final += (
                    process(child, r_test) * group_probs[:, group_idx : group_idx + 1]
                )
            return final

        probs = process(root, R_test_np)
        return torch.from_numpy(probs.astype(np.float32)).to(device)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #


class TabICLNet(nn.Module):
    """The three-stage TabICL network with parameter names matching checkpoints."""

    def __init__(self, cfg: TabICLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.col_embedder = ColEmbedding(cfg)
        self.row_interactor = RowInteraction(cfg)
        self.icl_predictor = ICLearning(cfg)


class TabICLTorchModel:
    """Inference wrapper around :class:`TabICLNet` loading converted weights."""

    def __init__(self, config: TabICLConfig, params: dict[str, np.ndarray]) -> None:
        self.config = config
        self.net = TabICLNet(config)
        state = {
            name: torch.from_numpy(np.asarray(arr, dtype=np.float32))
            for name, arr in params.items()
        }
        self.net.load_state_dict(state, strict=True)
        self.net.eval()

    @torch.no_grad()
    def representations(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Column embedding + row interaction. Returns (1, T, D) array."""
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(0)
        y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).unsqueeze(0)
        train_size = y.shape[1]
        emb = self.net.col_embedder(x, y, train_size)
        rep = self.net.row_interactor(emb)
        return rep.numpy()

    @torch.no_grad()
    def predict_from_representations(
        self,
        R: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Run the ICL stage on row representations.

        Returns (test_size, num_classes) probabilities/logits, or
        (test_size, num_quantiles) raw quantiles for regression.
        """
        cfg = self.config
        r = torch.from_numpy(np.ascontiguousarray(R, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
        if y.ndim == 1:
            y = y.unsqueeze(0)
        train_size = y.shape[1]
        num_classes = len(np.unique(y_train)) if cfg.max_classes > 0 else 0

        if cfg.max_classes == 0 or num_classes <= cfg.max_classes:
            out = self.net.icl_predictor.predict_standard(
                r, y, return_logits=return_logits, temperature=temperature
            ).squeeze(0)
            return out.numpy()

        root = fit_hierarchical_tree(R[0, :train_size], y_train, cfg.max_classes)
        probs = self.net.icl_predictor.predict_hierarchical(
            root, R[0, train_size:], r.device, temperature
        )
        if return_logits:
            probs = temperature * torch.log(probs + 1e-6)
        return probs.numpy()

    @torch.no_grad()
    def build_cache(self, X: np.ndarray, y_train: np.ndarray) -> dict:
        """Pre-compute col-stage and ICL-stage K/V caches for repeated predicts.

        Returns a dict with col K/V per ISAB block, row representations of the
        training data (post target-addition), and ICL K/V per layer.
        """
        cfg = self.config
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(0)
        y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).unsqueeze(0)
        train_size = y.shape[1]

        net = self.net
        col_cache: list = [None] * cfg.col_num_blocks
        col = net.col_embedder
        Xg = col.feature_grouping(x)
        pad = torch.full(
            (Xg.shape[0], Xg.shape[1], cfg.row_num_cls, Xg.shape[-1]),
            SKIP_VALUE,
            device=Xg.device,
            dtype=Xg.dtype,
        )
        Xg = torch.cat([pad, Xg], dim=2)
        features = Xg.transpose(1, 2)
        y_exp = y.unsqueeze(1).expand(-1, features.shape[1], -1)
        src = col.in_linear(features)
        if col.target_aware:
            if cfg.max_classes > 0:
                src[..., :train_size, :] = src[..., :train_size, :] + col.y_encoder(
                    y_exp.float()
                )
            else:
                src[..., :train_size, :] = src[..., :train_size, :] + col.y_encoder(
                    y_exp.unsqueeze(-1)
                )
        src = col.tf_col.forward_with_cache(src, col_cache, train_size, use_cache=False)
        embeddings = src.transpose(1, 2)

        rep = net.row_interactor(embeddings)  # (1, T, D)

        # ICL cache: add target embedding to train rows, then store K/V per layer.
        R = rep.clone()
        if cfg.max_classes > 0:
            R[:, :train_size] = R[:, :train_size] + net.icl_predictor.y_encoder(
                y.float()
            )
        else:
            R[:, :train_size] = R[:, :train_size] + net.icl_predictor.y_encoder(
                y.unsqueeze(-1)
            )
        icl_cache: list = [None] * cfg.icl_num_blocks
        net.icl_predictor.tf_icl.forward_with_cache(
            R, icl_cache, train_size, use_cache=False
        )
        num_classes = len(np.unique(y_train)) if cfg.max_classes > 0 else 0
        return {
            "col": col_cache,
            "icl": icl_cache,
            "train_size": train_size,
            "num_classes": num_classes,
        }

    @torch.no_grad()
    def predict_with_cache(
        self,
        X_test: np.ndarray,
        cache: dict,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Predict using pre-computed caches; ``X_test`` holds test rows only.

        Only supported for standard classification/regression (labels fixed at
        fit time); many-class hierarchical prediction bypasses the cache.
        """
        cfg = self.config
        x = torch.from_numpy(np.ascontiguousarray(X_test, dtype=np.float32)).unsqueeze(
            0
        )
        net = self.net
        col = net.col_embedder

        Xg = col.feature_grouping(x)
        pad = torch.full(
            (Xg.shape[0], Xg.shape[1], cfg.row_num_cls, Xg.shape[-1]),
            SKIP_VALUE,
            device=Xg.device,
            dtype=Xg.dtype,
        )
        Xg = torch.cat([pad, Xg], dim=2)
        features = Xg.transpose(1, 2)
        out = col.in_linear(features)
        for idx, block in enumerate(col.tf_col.blocks):
            hidden_k, hidden_v = cache["col"][idx]
            out = block.multihead_attn2(  # ty: ignore[call-non-callable]
                out, cached_kv=(hidden_k, hidden_v)
            )
        embeddings = out.transpose(1, 2)

        rep = net.row_interactor(embeddings)  # (1, T_test, D); positions restart at 0

        icl = net.icl_predictor
        r_out = rep
        for idx, block in enumerate(icl.tf_icl.blocks):
            k, v = cache["icl"][idx]
            r_out = block(r_out, cached_kv=(k, v))
        r_out = icl.ln(r_out)
        decoded = icl.decoder(r_out).squeeze(0)

        if cfg.max_classes == 0:
            return decoded.numpy()
        num_classes = int(cache.get("num_classes", decoded.shape[-1]))
        decoded = decoded[:, :num_classes]
        if not return_logits:
            decoded = torch.softmax(decoded / temperature, dim=-1)
        return decoded.numpy()

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
