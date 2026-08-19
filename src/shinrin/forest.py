"""Random forest variants with an unbiased feature importances estimator.

Vendored and adapted from scikit-garden.
"""

from shinrin._skgarden.forest import ExtraTreesRegressor, RandomForestRegressor

__all__ = ["ExtraTreesRegressor", "RandomForestRegressor"]
