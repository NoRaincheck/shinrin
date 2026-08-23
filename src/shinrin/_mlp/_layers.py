"""Parameter containers and initialization for the plain MLP.

Weights use scikit-learn's MLP initialization (Glorot uniform:
``uniform(-limit, limit)`` with ``limit = sqrt(6 / (fan_in + fan_out))``)
so both implementations start from the same distribution. When the PLE
embedding is enabled the numerical-feature parameters mirror
:mod:`shinrin._tabm._layers` (per-feature linear map plus a small random
piecewise-linear projection).

Flat layout (mirrored exactly by ``shinrin._mlp_kernels.mojo``):

    emb_w0, emb_b0, emb_wp_0..F-1,      (when embeddings enabled)
    l{i}_w, l{i}_b,                     for i in 0..n_layers-1

Layer ``i`` maps ``layer_sizes[i] -> layer_sizes[i+1]``; the last entry of
``layer_sizes`` is the output width. Weights are stored row-major
``(d_out_i, d_in_i)`` so forward passes evaluate ``x @ W.T + b``.
"""

from __future__ import annotations

import numpy as np

from shinrin._quant import GRANULARITIES, QUANTIZATION_NONE, validate_quantization

ACTIVATIONS = ("identity", "logistic", "tanh", "relu")


class MLPConfig:
    """Static description of an MLP architecture (no parameters)."""

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_out: int,
        layer_sizes: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_embeddings: bool = False,
        bins: list[np.ndarray] | None = None,
        d_embedding: int = 8,
        quantization: str = QUANTIZATION_NONE,
        quantization_granularity: str = "per_row",
        quantize_output: bool = False,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation!r}")
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least input and output widths")
        validate_quantization(quantization, quantization_granularity)
        self.n_num_features = n_num_features
        self.cat_cardinalities = list(cat_cardinalities)
        self.d_out = d_out
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.dropout = dropout
        self.use_embeddings = use_embeddings
        self.bins = bins
        self.d_embedding = d_embedding
        self.quantization = quantization
        self.quantization_granularity = (
            quantization_granularity
            if quantization != QUANTIZATION_NONE
            else GRANULARITIES[0]
        )
        self.quantize_output = bool(quantize_output)

    @property
    def n_layers(self) -> int:
        """Number of weight matrices (hidden layers + output layer)."""
        return len(self.layer_sizes) - 1

    @property
    def bin_counts(self) -> list[int]:
        if self.bins is None:
            return []
        return [len(b) - 1 for b in self.bins]

    def layer_is_quantized(self, i: int) -> bool:
        """Whether backbone weight matrix ``i`` is ternary-quantized.

        The output layer is kept at full precision unless explicitly
        requested via ``quantize_output``.
        """
        if self.quantization == QUANTIZATION_NONE:
            return False
        if i == self.n_layers - 1:
            return self.quantize_output
        return True

    @property
    def d_enc(self) -> int:
        return sum(self.bin_counts)

    @property
    def d_in(self) -> int:
        """Backbone input width."""
        if self.use_embeddings and self.n_num_features:
            num = self.n_num_features * self.d_embedding
        else:
            num = self.n_num_features
        return num + sum(self.cat_cardinalities)


def _glorot_uniform(
    rng: np.random.RandomState,
    fan_in: int,
    fan_out: int,
    factor: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and biases like sklearn's ``_init_coef`` (Glorot uniform).

    Biases share the weight limit and are drawn right after the weights
    from the same stream (matching scikit-learn exactly).
    """
    init_bound = np.sqrt(factor / (fan_in + fan_out))
    w = rng.uniform(-init_bound, init_bound, size=(fan_in, fan_out))
    b = rng.uniform(-init_bound, init_bound, size=fan_out)
    return (
        w.astype(np.float32),
        b.astype(np.float32),
    )


class MLPParams:
    """Trainable parameters of a plain MLP held as float32 arrays."""

    def __init__(self, config: MLPConfig, arrays: dict[str, np.ndarray]) -> None:
        self.config = config
        self.arrays = arrays

    @classmethod
    def init(cls, config: MLPConfig, seed: int | None = None) -> MLPParams:
        rng = np.random.RandomState(seed)
        cfg = config
        arrays: dict[str, np.ndarray] = {}

        if cfg.use_embeddings and cfg.n_num_features:
            demb = cfg.d_embedding
            limit = np.sqrt(6.0 / demb)
            arrays["emb_w0"] = rng.uniform(
                -limit, limit, size=(cfg.n_num_features, demb)
            ).astype(np.float32)
            arrays["emb_b0"] = rng.uniform(
                -limit, limit, size=(cfg.n_num_features, demb)
            ).astype(np.float32)
            for f, count in enumerate(cfg.bin_counts):
                # Small nonzero init keeps the piecewise component learnable
                # from the first step and avoids a field of ReLU kinks.
                arrays[f"emb_wp_{f}"] = (0.01 * rng.randn(count, demb)).astype(
                    np.float32
                )

        factor = 2.0 if cfg.activation == "logistic" else 6.0
        for i in range(cfg.n_layers):
            w, b = _glorot_uniform(
                rng, cfg.layer_sizes[i], cfg.layer_sizes[i + 1], factor
            )
            # Stored transposed (d_out, d_in) for x @ W.T forward passes.
            arrays[f"l{i}_w"] = np.ascontiguousarray(w.T)
            arrays[f"l{i}_b"] = b
        return cls(config, arrays)

    # -- flat vector view --------------------------------------------------

    def names(self) -> list[str]:
        return list(self.arrays.keys())

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.arrays[n].ravel() for n in self.arrays])

    @classmethod
    def unflatten(cls, config: MLPConfig, theta: np.ndarray) -> MLPParams:
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

    # -- sklearn-compatible views -------------------------------------------

    def coefs_(self) -> list[np.ndarray]:
        return [self.arrays[f"l{i}_w"].T.copy() for i in range(self.config.n_layers)]

    def intercepts_(self) -> list[np.ndarray]:
        return [self.arrays[f"l{i}_b"].copy() for i in range(self.config.n_layers)]
