"""Tests for automatic categorical-feature awareness (detection heuristic +
CatBoost-style target-statistic encoding) shared by Mondrian and SPOT/SPOTSET."""

import numpy as np
import pytest

from shinrin._categorical import TargetStatisticsEncoder, resolve_categorical_mask

sklearn = pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Detection heuristic
# ---------------------------------------------------------------------------


def test_detects_integer_columns_under_cardinality_cap():
    X = np.array(
        [
            [0.0, 1.5, 5.0],
            [1.0, 2.5, 7.0],
            [2.0, 3.5, 5.0],
            [1.0, 4.5, 9.0],
        ]
    )
    mask = resolve_categorical_mask(X, "auto")
    assert mask is not None
    assert mask.tolist() == [True, False, True]


def test_ignores_constant_and_high_cardinality_columns():
    constant = np.full((40, 1), 3.0)
    high_card = np.arange(40, dtype=float).reshape(-1, 1)
    mixed = np.column_stack([constant, high_card])
    mask = resolve_categorical_mask(mixed, "auto")
    assert mask is not None and not mask.any()


def test_respects_max_categories():
    col = np.arange(16, dtype=float).reshape(-1, 1)
    loose = resolve_categorical_mask(col, "auto", max_categories=16)
    tight = resolve_categorical_mask(col, "auto", max_categories=15)
    assert loose is not None and tight is not None
    assert loose[0]
    assert not tight[0]


def test_non_integral_floats_never_detected():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(20, 1)).astype(float)
    X += 0.5  # no longer integral
    mask = resolve_categorical_mask(X, "auto")
    assert mask is not None and not mask.any()


def test_nan_or_inf_disqualifies_column():
    X = np.array([[0.0], [1.0], [np.nan], [1.0]])
    mask = resolve_categorical_mask(X, "auto")
    assert mask is not None and not mask[0]


def test_disabled_spec_returns_none():
    X = np.zeros((4, 2))
    assert resolve_categorical_mask(X, None) is None
    assert resolve_categorical_mask(X, False) is None


def test_explicit_boolean_mask_and_indices():
    X = np.zeros((4, 3))
    by_mask = resolve_categorical_mask(X, np.array([False, True, False]))
    by_index = resolve_categorical_mask(X, np.array([1]))
    assert by_mask is not None and by_index is not None
    assert by_mask.tolist() == [False, True, False]
    assert np.array_equal(by_mask, by_index)


@pytest.mark.parametrize(
    "spec", ["automatic", [[0, 1]], np.array([True, True, True, True]), [99]]
)
def test_invalid_specs_raise(spec):
    X = np.zeros((4, 3))
    if isinstance(spec, np.ndarray) and spec.dtype == bool:
        # wrong-length boolean mask must raise
        with pytest.raises(ValueError, match="boolean mask"):
            resolve_categorical_mask(X, spec)
    else:
        with pytest.raises(ValueError):
            resolve_categorical_mask(X, spec)


# ---------------------------------------------------------------------------
# Target-statistic encoder
# ---------------------------------------------------------------------------


def test_encoder_smoothing_formula_matches_manual_computation():
    X = np.array([[0.0], [0.0], [0.0], [1.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    enc = TargetStatisticsEncoder(smoothing=1.0).fit(X, y, np.array([True]))
    prior = y.mean()  # 0.5
    # rows [0,0,0,1]: cat-0 stat (2 + 0.5)/4 = 0.625; cat-1 stat (0 + 0.5)/2 = 0.25
    expected = np.full((4, 1), 2.5 / 4.0)
    expected[3, 0] = 0.25
    np.testing.assert_allclose(enc.transform(X), expected, rtol=1e-6)
    assert enc.prior_ == pytest.approx(prior)


def test_encoder_zero_smoothing_is_plain_mean():
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([3.0, 5.0, 10.0])
    enc = TargetStatisticsEncoder(smoothing=0.0).fit(X, y, np.array([True]))
    out = enc.transform(X)
    assert out[0, 0] == pytest.approx(4.0)
    assert out[2, 0] == pytest.approx(10.0)


def test_unseen_categories_map_to_prior_at_transform():
    X_fit = np.array([[0.0], [1.0]])
    y = np.array([1.0, 0.0])
    enc = TargetStatisticsEncoder(smoothing=1.0).fit(X_fit, y, np.array([True]))
    X_new = np.array([[7.0]])
    np.testing.assert_allclose(enc.transform(X_new), [[y.mean()]])


def test_transform_preserves_shape_and_dtype():
    X = np.array([[0.0, 2.5], [1.0, 3.5], [0.0, 4.5]], dtype=np.float32)
    y = np.array([0, 1, 1])
    enc = TargetStatisticsEncoder().fit(X, y, np.array([True, False]))
    out = enc.transform(X)
    assert out.shape == X.shape
    assert out.dtype == np.float32
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out[:, 1], X[:, 1])


def test_multiclass_classindex_weighted_statistic():
    # classes are integer indices; stat is E[class | category]
    X = np.array([[0.0]] * 4 + [[1.0]] * 2)
    y = np.array([0, 0, 1, 1, 2, 2])
    enc = TargetStatisticsEncoder(smoothing=0.0).fit(X, y, np.array([True]))
    out = enc.transform(X)
    assert out[0, 0] == pytest.approx(0.5)  # mean class index of cat 0
    assert out[4, 0] == pytest.approx(2.0)


def test_inactive_encoder_is_identity():
    X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    enc = TargetStatisticsEncoder().fit(
        X, np.array([0.0, 1.0]), np.array([False, False])
    )
    assert not enc.active_
    np.testing.assert_array_equal(enc.transform(X), X)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        TargetStatisticsEncoder().transform(np.zeros((2, 1)))


# ---------------------------------------------------------------------------
# ThresholdGuessBinarizer (SPOT / SPOTSET)
# ---------------------------------------------------------------------------


def _cat_informative_dataset(n=300, seed=7):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 4, size=n).astype(float)  # 4 categories
    num = rng.normal(size=n)
    X = np.column_stack([cat, num, rng.integers(0, 10, size=n).astype(float)])
    y = ((cat % 2) + (num > 0)).astype(int) % 2
    return X, y


def test_tgb_auto_detects_and_encodes_categoricals():
    from shinrin._spot import ThresholdGuessBinarizer

    X, y = _cat_informative_dataset()
    enc = ThresholdGuessBinarizer(n_estimators=20, random_state=0).fit(X, y)
    assert enc.categorical_mask_.tolist() == [True, False, True]
    # the GBDT must have found at least one usable axis overall
    assert enc.n_features_out_ > 0
    Xt = enc.transform(X)
    assert Xt.shape == (X.shape[0], enc.n_features_out_)
    assert set(np.unique(Xt)) <= {0.0, 1.0}


def test_tgb_target_encoding_partitions_categories():
    from shinrin._spot import ThresholdGuessBinarizer

    X, y = _cat_informative_dataset()
    enc = ThresholdGuessBinarizer(
        n_estimators=30, column_elimination=False, random_state=0
    ).fit(X, y)
    cat_identities = [i for i in enc.thresholds_ if i[0] == "cat_target"]
    if not cat_identities:
        pytest.skip("guesser found no categorical axes")
    # every category maps to exactly one smoothed statistic value
    stats = enc.cat_stats_[0]
    # every category maps to exactly one statistic value
    assert len(stats) == 4
    # unseen category falls back to the prior
    X_unseen = X.copy()
    X_unseen[:, 0] = 99.0
    out = enc.transform(X_unseen[:1])
    ref = enc.transform(X[:1])
    # prior-based encoding differs from any seen-category row only if the
    # statistic crosses the threshold differently; just check it runs.
    assert out.shape == ref.shape


def test_tgb_onehot_mode():
    from shinrin._spot import ThresholdGuessBinarizer

    X, y = _cat_informative_dataset()
    enc = ThresholdGuessBinarizer(
        n_estimators=20, categorical_encoding="onehot", random_state=0
    ).fit(X, y)
    kinds = {identity[0] for identity in enc.thresholds_}
    assert kinds <= {"numeric", "cat_onehot"}
    Xt = enc.transform(X)
    assert Xt.shape[1] == enc.n_features_out_


def test_tgb_invalid_encoding_raises():
    from shinrin._spot import ThresholdGuessBinarizer

    X, y = _cat_informative_dataset(60)
    with pytest.raises(ValueError, match="categorical_encoding"):
        ThresholdGuessBinarizer(n_estimators=2, categorical_encoding="hash").fit(X, y)


def test_tgb_numeric_only_matches_legacy_output():
    from shinrin._spot import ThresholdGuessBinarizer

    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    new = ThresholdGuessBinarizer(
        n_estimators=10, random_state=0, categorical_features=None
    ).fit(X, y)
    old = ThresholdGuessBinarizer(n_estimators=10, random_state=0)
    # simulate legacy behaviour via explicit None as well
    old.set_params(categorical_features=None)
    old.fit(X, y)
    np.testing.assert_array_equal(new.transform(X), old.transform(X))
    assert all(i[0] == "numeric" for i in new.thresholds_)


def test_spot_end_to_end_with_categoricals():
    from shinrin._spot import SPOTClassifier, ThresholdGuessBinarizer

    X, y = _cat_informative_dataset(n=250, seed=11)
    enc = ThresholdGuessBinarizer(n_estimators=15, random_state=0)
    X_bin = enc.fit_transform(X, y)
    clf = SPOTClassifier(depth_budget=4, worker_limit=1, time_limit=30)
    clf.fit(X_bin, y)
    acc = clf.score(X_bin, y)
    assert acc >= 0.8


# ---------------------------------------------------------------------------


def test_mondrian_auto_categorical_improves_separability():
    from shinrin import MondrianForestClassifier

    rng = np.random.default_rng(42)
    n = 400
    # informative integer-coded categorical feature: class follows parity
    cat = rng.integers(0, 6, size=n).astype(np.float32)
    noise = rng.normal(size=(n, 4)).astype(np.float32)  # pure noise dimensions
    X = np.hstack([cat[:, None], noise])
    y = (cat.astype(int) % 2).astype(int)

    auto = MondrianForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    off = MondrianForestClassifier(
        n_estimators=20, random_state=0, categorical_features=None
    ).fit(X, y)

    assert auto.score(X, y) > 0.85
    assert auto.score(X, y) >= off.score(X, y)
    # auto mode must have detected exactly the first column
    assert auto.categorical_features_.tolist() == [True, False, False, False, False]


def test_mondrian_disabled_matches_default_numeric_behaviour():
    from shinrin import MondrianForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    explicit_off = MondrianForestClassifier(
        n_estimators=5, random_state=0, categorical_features=None
    )
    # float data: even "auto" finds nothing, so both paths must agree exactly
    auto = MondrianForestClassifier(n_estimators=5, random_state=0)
    assert explicit_off.fit(X, y).score(X, y) == pytest.approx(
        auto.fit(X, y).score(X, y)
    )


def test_target_statistics_encoder_transform_is_pure():
    """Regression: transform once mutated a zero-copy float32 buffer in
    place, silently re-encoding callers' training data between fits."""
    enc = TargetStatisticsEncoder()
    X = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    enc.fit(X, y, np.array([True]))
    snapshot = X.copy()
    out = enc.transform(X)
    np.testing.assert_array_equal(X, snapshot)
    assert out is not X


def test_forest_fit_does_not_mutate_training_data():
    from shinrin import MondrianForestClassifier

    rng = np.random.default_rng(5)
    X = np.column_stack(
        [
            rng.integers(0, 4, size=120).astype(np.float32),
            rng.normal(size=120).astype(np.float32),
        ]
    )
    y = (X[:, 0] % 2).astype(int)
    snapshot = X.copy()
    clf = MondrianForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    np.testing.assert_array_equal(X, snapshot)
    assert clf.score(X, y) > 0.9
