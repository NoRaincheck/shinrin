"""Ternary weight quantization (BitNet-style "BitLinear") primitives.

Shared by the MLP and TabM trainers. The scheme follows BitNet b1.58:
latent full-precision weights are kept as the trainable parameters while
the forward pass uses their ternary approximation

    gamma = mean(|W|)                      # absmean scale
    W_eff = round(clip(W / gamma, -1, 1)) * gamma

so every effective weight lies in ``{-1, 0, +1} * gamma`` (~1.58 bits of
information per weight). Gradients flow through the quantization as if it
were the identity (straight-through estimator), which is exact for the
scale factor and a faithful approximation for the rounding step.

Rounding is half-to-even to match ``numpy.round`` exactly; the Mojo
kernels implement the same rule so both backends stay in parity.

Granularity:

- ``"per_row"``     — one scale per output row of ``W`` (shape ``(d_out,)``);
  the default and generally the more accurate choice.
- ``"per_tensor"``  — one scale per weight matrix.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QUANTIZATION_NONE",
    "QUANTIZATION_TERNARY",
    "QUANTIZATIONS",
    "GRANULARITIES",
    "validate_quantization",
    "ternary_scales",
    "ternary_quantize_dequantize",
]

QUANTIZATION_NONE = "none"
QUANTIZATION_TERNARY = "ternary"
QUANTIZATIONS = (QUANTIZATION_NONE, QUANTIZATION_TERNARY)
GRANULARITIES = ("per_row", "per_tensor")

# Weights whose scale collapses to zero (an all-zero matrix/row) would
# divide by zero; any positive stand-in works because W is all-zero there.
_TINY = np.float32(1e-12)


def validate_quantization(quantization: str, granularity: str) -> None:
    """Validate the estimator-level quantization parameters."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(
            f"quantization must be one of {QUANTIZATIONS}, got {quantization!r}"
        )
    if quantization == QUANTIZATION_NONE:
        return
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"quantization_granularity must be one of {GRANULARITIES}, "
            f"got {granularity!r}"
        )


def ternary_scales(w: np.ndarray, granularity: str = "per_row") -> np.ndarray:
    """Absmean scales for ``w`` shaped for broadcasting over ``(d_out, d_in)``.

    Returns shape ``(d_out, 1)`` for ``"per_row"`` or ``()`` (a scalar
    array) for ``"per_tensor"`` so callers can simply multiply/divide.
    """
    w2 = np.asarray(w, dtype=np.float32)
    if granularity == "per_row":
        return np.maximum(
            np.mean(np.abs(w2), axis=1, keepdims=True).astype(np.float32), _TINY
        )
    if granularity == "per_tensor":
        return np.maximum(np.float32(np.mean(np.abs(w2))), _TINY)
    raise ValueError(f"Unknown granularity: {granularity!r}")


def ternary_quantize_dequantize(w: np.ndarray, granularity: str = "per_row") -> np.ndarray:
    """Return the ternary approximation of ``w`` used by the forward pass.

    The result has the same shape/dtype as ``w`` but every element is one
    of ``{-gamma, 0, +gamma}`` with ``gamma`` the absmean scale(s).
    """
    s = ternary_scales(w, granularity)
    return np.round(np.clip(w / s, -1.0, 1.0)) * s
