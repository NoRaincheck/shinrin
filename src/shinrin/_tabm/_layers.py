"""Parameter containers and TabM-style initialization (NumPy).

Initialization follows yandex-research/tabm (Apache-2.0, see NOTICE):

- shared weights/biases: uniform in ``[-d**-0.5, d**-0.5]``
- scaling adapters ``r``/``s``: ``random-signs`` or ``normal``; under
  ``tabm_init`` only the very first adapter is random and all later ones
  start at exactly 1
- with ``start_scaling_init_chunks`` (one chunk per input feature) all
  values within one feature block share a single sampled value
"""

from __future__ import annotations

import numpy as np


def _rsqrt_uniform(rng: np.random.RandomState, size: tuple[int, ...], d: int):
    return rng.uniform(-(d**-0.5), d**-0.5, size=size).astype(np.float32)


def _init_scaling(
    rng: np.random.RandomState,
    shape: tuple[int, ...],
    distribution: str,
    chunks: list[int] | None = None,
) -> np.ndarray:
    if distribution == "ones":
        out = np.ones(shape, dtype=np.float32)
    elif distribution == "normal":
        out = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    elif distribution == "random-signs":
        out = (rng.randint(0, 2, size=shape) * 2 - 1).astype(np.float32)
    else:
        raise ValueError(f"Unknown scaling init distribution: {distribution!r}")
    if chunks is not None:
        if sum(chunks) != shape[-1]:
            raise ValueError("chunks do not match the last dimension")
        expanded = np.empty(shape, dtype=np.float32)
        start = 0
        for size in chunks:
            expanded[..., start : start + size] = out[..., start, None]
            start += size
        out = expanded
    return out


class TabMConfig:
    """Static description of a TabM architecture (no parameters).

    The flat representation seen by the backbone has width
    ``n_num_features * d_embedding + sum(cat_cardinalities)`` when
    embeddings are enabled, otherwise ``n_features``.
    """

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_out: int,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        arch_type: str = "tabm",
        use_embeddings: bool = True,
        bins: list[np.ndarray] | None = None,
        d_embedding: int = 8,
    ) -> None:
        if arch_type not in ("tabm", "tabm-mini", "tabm-packed"):
            raise ValueError(f"Unknown arch_type: {arch_type!r}")
        self.n_num_features = n_num_features
        self.cat_cardinalities = list(cat_cardinalities)
        self.d_out = d_out
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.arch_type = arch_type
        self.use_embeddings = use_embeddings
        self.bins = bins
        self.d_embedding = d_embedding

    @property
    def n_cat_features(self) -> int:
        return len(self.cat_cardinalities)

    @property
    def bin_counts(self) -> list[int]:
        if self.bins is None:
            return []
        return [len(b) - 1 for b in self.bins]

    @property
    def d_enc(self) -> int:
        """Width of the piecewise-linear encoding of the numerical features."""
        return sum(self.bin_counts)

    @property
    def d_in(self) -> int:
        """Backbone input width."""
        if self.use_embeddings and self.n_num_features:
            num = self.n_num_features * self.d_embedding
        else:
            num = self.n_num_features
        return num + sum(self.cat_cardinalities)

    @property
    def feature_chunks(self) -> list[int]:
        """Per-feature widths of the flat representation (for chunked init)."""
        if self.use_embeddings and self.n_num_features:
            num = [self.d_embedding] * self.n_num_features
        else:
            num = [1] * self.n_num_features
        return num + self.cat_cardinalities


class TabMParams:
    """Trainable parameters of a TabM model held as float32 arrays.

    Layout (also used for the flat vector handed to the Mojo kernels):

    - embeddings (when enabled): ``emb_w0 (F, demb)``, ``emb_b0 (F, demb)``,
      then one zero-initialized ``(bins_f, demb)`` projection per feature
    - ``arch_type='tabm'``, block ``i``: ``W (d_block, d_in_i)``,
      ``r (k, d_in_i)``, ``s (k, d_block)``, ``b (k, d_block)``
    - ``arch_type='tabm-mini'``: shared ``W (d_block, d_in_i)`` /
      ``b (d_block,)`` per block plus a single ``mini_r (k, d_in)``
    - ``arch_type='tabm-packed'``, block ``i``: ``W (k, d_in_i, d_block)``,
      ``b (k, d_block)``
    - head: ``head_w (k, d_block, d_out)``, ``head_b (k, d_out)``
    """

    def __init__(self, config: TabMConfig, arrays: dict[str, np.ndarray]) -> None:
        self.config = config
        self.arrays = arrays

    @classmethod
    def init(cls, config: TabMConfig, seed: int | None = None) -> TabMParams:
        rng = np.random.RandomState(seed)
        cfg = config
        arrays: dict[str, np.ndarray] = {}
        chunks = cfg.feature_chunks

        if cfg.use_embeddings and cfg.n_num_features:
            demb = cfg.d_embedding
            arrays["emb_w0"] = _rsqrt_uniform(rng, (cfg.n_num_features, demb), demb)
            arrays["emb_b0"] = _rsqrt_uniform(rng, (cfg.n_num_features, demb), demb)
            for f, count in enumerate(cfg.bin_counts):
                # Small random init (instead of the upstream exact-zero init):
                # keeps the piecewise component learnable from the first step
                # and avoids a field of ReLU kinks that breaks quasi-Newton
                # curvature estimates.
                arrays[f"emb_wp_{f}"] = (0.01 * rng.randn(count, demb)).astype(
                    np.float32
                )

        first_scaling = None if cfg.arch_type == "tabm-packed" else "random-signs"
        d_in = cfg.d_in
        for i in range(cfg.n_blocks):
            b_in = d_in if i == 0 else cfg.d_block
            prefix = f"blk{i}_"
            if cfg.arch_type == "tabm":
                arrays[prefix + "w"] = _rsqrt_uniform(rng, (cfg.d_block, b_in), b_in)
                if i == 0:
                    arrays[prefix + "r"] = _init_scaling(
                        rng, (cfg.k, b_in), first_scaling or "random-signs", chunks
                    )
                    arrays[prefix + "s"] = _init_scaling(
                        rng, (cfg.k, cfg.d_block), "ones"
                    )
                else:
                    arrays[prefix + "r"] = _init_scaling(rng, (cfg.k, b_in), "ones")
                    arrays[prefix + "s"] = _init_scaling(
                        rng, (cfg.k, cfg.d_block), "ones"
                    )
                bias = _rsqrt_uniform(rng, (cfg.d_block,), b_in)
                arrays[prefix + "b"] = np.broadcast_to(
                    bias, (cfg.k, cfg.d_block)
                ).copy()
            elif cfg.arch_type == "tabm-mini":
                arrays[prefix + "w"] = _rsqrt_uniform(rng, (cfg.d_block, b_in), b_in)
                arrays[prefix + "b"] = _rsqrt_uniform(rng, (cfg.d_block,), b_in)
                if i == 0:
                    arrays["mini_r"] = _init_scaling(
                        rng, (cfg.k, d_in), first_scaling or "random-signs", chunks
                    )
            else:  # tabm-packed
                arrays[prefix + "w"] = _rsqrt_uniform(
                    rng, (cfg.k, b_in, cfg.d_block), b_in
                )
                arrays[prefix + "b"] = _rsqrt_uniform(rng, (cfg.k, cfg.d_block), b_in)

        arrays["head_w"] = _rsqrt_uniform(
            rng, (cfg.k, cfg.d_block, cfg.d_out), cfg.d_block
        )
        arrays["head_b"] = _rsqrt_uniform(rng, (cfg.k, cfg.d_out), cfg.d_block)
        return cls(config, arrays)

    # -- flat vector view --------------------------------------------------

    def names(self) -> list[str]:
        return list(self.arrays.keys())

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.arrays[n].ravel() for n in self.arrays])

    @classmethod
    def unflatten(cls, config: TabMConfig, theta: np.ndarray) -> TabMParams:
        params = cls.init(config, seed=None)
        offset = 0
        for name, arr in params.arrays.items():
            size = arr.size
            params.arrays[name] = (
                theta[offset : offset + size].reshape(arr.shape).astype(np.float32)
            )
            offset += size
        if offset != theta.size:
            raise ValueError("theta size does not match the architecture")
        return params

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: arr.shape for name, arr in self.arrays.items()}
