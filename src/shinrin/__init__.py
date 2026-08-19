"""shinrin: random forests, quantile regression and rule extraction.

This package vendors scikit-garden (BSD-3) and skope-rules (BSD-3) so they
can be used without being installed as dependencies.
"""

from shinrin._skgarden.forest import ExtraTreesRegressor, RandomForestRegressor
from shinrin._skrules.rule import Rule, replace_feature_name
from shinrin._skrules.skope_rules import SkopeRules
from shinrin.mondrian import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    MondrianTreeClassifier,
    MondrianTreeRegressor,
)
from shinrin.quantile import (
    DecisionTreeQuantileRegressor,
    ExtraTreeQuantileRegressor,
    ExtraTreesQuantileRegressor,
    RandomForestQuantileRegressor,
)

__version__ = "0.1.0"

__all__ = [
    "DecisionTreeQuantileRegressor",
    "ExtraTreeQuantileRegressor",
    "ExtraTreesQuantileRegressor",
    "ExtraTreesRegressor",
    "MondrianForestClassifier",
    "MondrianForestRegressor",
    "MondrianTreeClassifier",
    "MondrianTreeRegressor",
    "RandomForestQuantileRegressor",
    "RandomForestRegressor",
    "Rule",
    "SkopeRules",
    "replace_feature_name",
]
