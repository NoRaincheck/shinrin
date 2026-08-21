"""Quantile-to-distribution postprocessing for TabICL regression (NumPy).

Converts the 999 predicted quantiles into a proper distribution with
exponential tail extrapolation, mirroring ``tabicl._model.quantile_dist``:

- crossing is fixed by sorting (default),
- tail scales are estimated by log-space linear regression over the most
  extreme quantiles,
- statistics (mean / variance / quantiles via inverse CDF) have closed forms.

Only what inference needs is implemented; GPD tails are supported for
parameter parity but the regressor defaults to exponential tails.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

TOL = 1e-6
MIN_BETA = 0.01
MAX_BETA = 100.0
MIN_ETA = -0.49
MAX_ETA = 0.49
ETA_TOLERANCE = 0.01
MAX_LOG_RATIO = 15.0
MAX_EXPONENT = 15.0
TAIL_QUANTILES_FOR_ESTIMATION = 20


def enforce_monotonicity(
    quantiles: np.ndarray,
    method: Literal["sort", "isotonic", "cummax"] = "sort",
) -> np.ndarray:
    """Fix quantile crossing so values are non-decreasing in level."""
    if method == "sort":
        return np.sort(quantiles, axis=-1)
    if method == "cummax":
        return np.maximum.accumulate(quantiles, axis=-1)
    raise ValueError(f"Unknown method: {method}. Use 'isotonic', 'cummax', or 'sort'.")


def estimate_exp_tail_params(
    quantiles: np.ndarray,
    alpha_levels: np.ndarray,
    num_tail_quantiles: int = TAIL_QUANTILES_FOR_ESTIMATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate exponential tail scales via log-space linear regression."""
    n = quantiles.shape[-1]
    k = min(num_tail_quantiles, n // 4)

    ln_alpha_left = np.log(np.clip(alpha_levels[:k], TOL, None))
    q_left = quantiles[..., :k]
    centered_a = ln_alpha_left - ln_alpha_left.mean()
    centered_q = q_left - q_left.mean(axis=-1, keepdims=True)
    cov = (centered_q * centered_a).mean(axis=-1)
    var = (centered_a**2).mean()
    beta_l = np.abs(cov / max(var, TOL))
    beta_l = np.clip(beta_l, MIN_BETA, MAX_BETA)

    ln_1ma_right = np.log(np.clip(1 - alpha_levels[-k:], TOL, None))
    q_right = quantiles[..., -k:]
    centered_a_r = ln_1ma_right - ln_1ma_right.mean()
    centered_q_r = q_right - q_right.mean(axis=-1, keepdims=True)
    cov_r = (centered_q_r * centered_a_r).mean(axis=-1)
    var_r = (centered_a_r**2).mean()
    beta_r = np.abs(-cov_r / max(var_r, TOL))
    beta_r = np.clip(beta_r, MIN_BETA, MAX_BETA)

    return beta_l.astype(quantiles.dtype), beta_r.astype(quantiles.dtype)


def default_alpha_levels(num_quantiles: int) -> np.ndarray:
    """Default levels: interior points of an equal-spaced grid on [0, 1]."""
    return np.linspace(0.0, 1.0, num_quantiles + 2)[1:-1].astype(np.float32)


class QuantileDistribution:
    """Distribution built from predicted quantiles with exponential tails."""

    def __init__(
        self,
        quantiles: np.ndarray,
        alpha_levels: np.ndarray | None = None,
        fix_crossing: bool = True,
        crossing_method: Literal["sort", "isotonic", "cummax"] = "sort",
    ) -> None:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        self.num_quantiles = quantiles.shape[-1]
        self.alpha_levels = (
            default_alpha_levels(self.num_quantiles).astype(np.float64)
            if alpha_levels is None
            else np.asarray(alpha_levels, dtype=np.float64)
        )
        if fix_crossing:
            quantiles = enforce_monotonicity(quantiles, method=crossing_method)
        self.quantiles = quantiles

        # Spline segments between consecutive knots.
        alpha = self.alpha_levels
        self.alpha_lo, self.alpha_hi = alpha[:-1], alpha[1:]
        self.q_lo, self.q_hi = quantiles[..., :-1], quantiles[..., 1:]
        self.delta_alpha = self.alpha_hi - self.alpha_lo
        self.num_segments = self.num_quantiles - 1

        # Boundary values.
        self.alpha_l = float(alpha[0])
        self.alpha_r = float(alpha[-1])
        self.q_l = quantiles[..., 0]
        self.q_r = quantiles[..., -1]

        # Exponential tails: Q(a) = a_l*ln(alpha) + b_l ; Q(a) = a_r*ln(1-alpha) + b_r.
        self.beta_l, self.beta_r = estimate_exp_tail_params(
            quantiles, self.alpha_levels
        )
        alpha_l_safe = max(self.alpha_l, TOL)
        alpha_r_safe = min(self.alpha_r, 1 - TOL)
        self.tail_a_l = self.beta_l
        self.tail_b_l = self.q_l - self.tail_a_l * math.log(alpha_l_safe)
        self.tail_a_r = -self.beta_r
        self.tail_b_r = self.q_r - self.tail_a_r * math.log(1 - alpha_r_safe)

    def icdf(self, alpha: float | np.ndarray) -> np.ndarray:
        """Inverse CDF at probability levels ``alpha`` (broadcastable)."""
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        scalar = alpha_arr.ndim == 0
        a = np.atleast_1d(alpha_arr)

        q_left = (
            self.tail_a_l[..., None] * np.log(np.clip(a, TOL, None))
            + self.tail_b_l[..., None]
        )
        q_right = (
            self.tail_a_r[..., None] * np.log(np.clip(1 - a, TOL, None))
            + self.tail_b_r[..., None]
        )

        seg_idx = np.clip(
            np.searchsorted(self.alpha_lo, a, side="right") - 1,
            0,
            self.num_segments - 1,
        )
        q_lo_g = self.q_lo[..., seg_idx]
        q_hi_g = self.q_hi[..., seg_idx]
        a_lo_g = self.alpha_lo[seg_idx]
        a_hi_g = self.alpha_hi[seg_idx]
        t = (a - a_lo_g) / np.clip(a_hi_g - a_lo_g, TOL, None)
        q_spline = q_lo_g + np.clip(t, 0.0, 1.0) * (q_hi_g - q_lo_g)
        q_spline = np.where(a >= self.alpha_r, self.q_r[..., None], q_spline)

        result = np.where(
            a < self.alpha_l, q_left, np.where(a > self.alpha_r, q_right, q_spline)
        )
        return result[..., 0] if scalar else result

    def mean(self) -> np.ndarray:
        """Analytical mean with exponential tails."""
        left = self.alpha_l * (self.q_l - self.tail_a_l)
        spline = (self.delta_alpha * (self.q_lo + self.q_hi) / 2).sum(axis=-1)
        right = (1 - self.alpha_r) * (self.q_r - self.tail_a_r)
        return left + spline + right

    def variance(self) -> np.ndarray:
        """Analytical variance with exponential tails."""
        a_l, a_r = self.tail_a_l, self.tail_a_r
        e_z2_left = self.alpha_l * (self.q_l**2 - 2 * a_l * self.q_l + 2 * a_l**2)
        e_z2_spline = (
            self.delta_alpha * (self.q_lo**2 + self.q_lo * self.q_hi + self.q_hi**2) / 3
        ).sum(axis=-1)
        e_z2_right = (1 - self.alpha_r) * (
            self.q_r**2 - 2 * a_r * self.q_r + 2 * a_r**2
        )
        e_z2 = e_z2_left + e_z2_spline + e_z2_right
        return np.clip(e_z2 - self.mean() ** 2, 0.0, None)


class QuantileToDistribution:
    """Wrapper converting raw quantile predictions into a distribution."""

    def __init__(
        self,
        num_quantiles: int = 999,
        fix_crossing: bool = True,
        crossing_method: Literal["sort", "isotonic", "cummax"] = "sort",
    ) -> None:
        self.num_quantiles = num_quantiles
        self.fix_crossing = fix_crossing
        self.crossing_method = crossing_method
        self.alpha_levels = default_alpha_levels(num_quantiles)

    def forward(self, quantiles: np.ndarray) -> QuantileDistribution:
        """Build a :class:`QuantileDistribution` from raw quantile predictions."""
        return QuantileDistribution(
            quantiles,
            alpha_levels=self.alpha_levels,
            fix_crossing=self.fix_crossing,
            crossing_method=self.crossing_method,
        )

    def mean(self, quantiles: np.ndarray) -> np.ndarray:
        """Mean of the distribution per row."""
        return self.forward(quantiles).mean()

    def median(self, quantiles: np.ndarray) -> np.ndarray:
        """Median of the distribution per row."""
        return self.forward(quantiles).icdf(0.5)

    def quantiles(self, quantiles: np.ndarray, alphas: list[float]) -> np.ndarray:
        """Quantiles of the distribution at levels ``alphas``."""
        return self.forward(quantiles).icdf(np.asarray(alphas, dtype=np.float64))

    def variance(self, quantiles: np.ndarray) -> np.ndarray:
        """Variance of the distribution per row."""
        return self.forward(quantiles).variance()

    def stats(
        self,
        quantiles: np.ndarray,
        output_type: str | list[str],
        alphas: list[float] | None = None,
    ):
        """Compute one or several statistics from raw quantile outputs.

        Returns a single array when ``output_type`` is a string, else a dict.
        """
        keys = [output_type] if isinstance(output_type, str) else list(output_type)
        dist = None
        results: dict[str, np.ndarray] = {}
        for key in keys:
            if key == "raw_quantiles":
                results[key] = quantiles
                continue
            if dist is None:
                dist = self.forward(quantiles)
            if key == "mean":
                results[key] = dist.mean()
            elif key == "median":
                results[key] = dist.icdf(0.5)
            elif key == "variance":
                results[key] = dist.variance()
            elif key == "quantiles":
                levels = (
                    alphas
                    if alphas is not None
                    else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
                results[key] = dist.icdf(np.asarray(levels, dtype=np.float64))
            else:
                raise ValueError(f"Unknown output_type '{key}'.")
        if isinstance(output_type, str):
            return results[output_type]
        return results
