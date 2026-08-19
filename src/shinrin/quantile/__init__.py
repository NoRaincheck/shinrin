"""Quantile regression trees and forests (vendored from scikit-garden)."""

from shinrin._skgarden.quantile.ensemble import (
    ExtraTreesQuantileRegressor,
    RandomForestQuantileRegressor,
)
from shinrin._skgarden.quantile.tree import (
    DecisionTreeQuantileRegressor,
    ExtraTreeQuantileRegressor,
)

__all__ = [
    "DecisionTreeQuantileRegressor",
    "ExtraTreeQuantileRegressor",
    "ExtraTreesQuantileRegressor",
    "RandomForestQuantileRegressor",
]
