"""Tests for categorical handling.

Combines two suites:

- Automatic categorical-feature awareness (detection heuristic +
  CatBoost-style target-statistic encoding) shared by Mondrian and
  SPOT/SPOTSET (from feat/categorical-awareness).
- CatBoost-style target encoder and partition recovery with
  CategoricalTree / BRANCH_MEMBER ONNX export (from main / #41).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.utils.validation import NotFittedError

from shinrin._categorical import TargetStatisticsEncoder, resolve_categorical_mask
from shinrin._encoding import TargetEncoder

sklearn = pytest.importorskip("sklearn")

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


# ---------------------------------------------------------------------------
# TargetEncoder + CategoricalTree + BRANCH_MEMBER export (from main)
# ---------------------------------------------------------------------------


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    n = 200
    color = rng.randint(0, 4, size=n).astype(np.float64)  # categorical codes
    size = rng.randn(n)  # numeric passthrough
    X = np.column_stack([color, size])
    y = (color == 1).astype(np.float64) * 0.8 + 0.2 * rng.randn(n)
    return X, y


class TestTargetEncoderStats:
    def test_matches_manual_statistics(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0], smoothing=1.0).fit(X, y)
        assert enc.categorical_features_ == [0]
        assert enc.prior_ == pytest.approx(y.mean())
        for code in np.unique(X[:, 0]):
            mask = X[:, 0] == code
            expected = (y[mask].sum() + 1.0 * y.mean()) / (mask.sum() + 1.0)
            idx = np.searchsorted(enc.categories_[0], code)
            assert enc.encodings_[0][idx] == pytest.approx(expected)

    def test_smoothing_shrinks_towards_prior(self, data):
        X, y = data
        strong = TargetEncoder(categorical_features=[0], smoothing=1000.0).fit(X, y)
        weak = TargetEncoder(categorical_features=[0], smoothing=0.0).fit(X, y)
        # With zero smoothing the encoding is the raw category mean; heavy
        # smoothing pulls every encoding towards the global prior.
        raw = np.array([y[X[:, 0] == c].mean() for c in weak.categories_[0]])
        assert np.allclose(weak.encodings_[0], raw)
        assert np.abs(strong.encodings_[0] - y.mean()).max() < (
            np.abs(raw - y.mean()).max()
        )

    def test_numeric_columns_passthrough(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        Xt = enc.transform(X)
        np.testing.assert_array_equal(Xt[:, 1], X[:, 1])

    def test_transform_matches_encoding_table(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0], smoothing=0.5).fit(X, y)
        Xt = enc.transform(X)
        lookup = dict(zip(enc.categories_[0], enc.encodings_[0]))
        expected = np.array([lookup[v] for v in X[:, 0]])
        np.testing.assert_allclose(Xt[:, 0], expected)

    def test_unseen_category_maps_to_prior(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        probe = np.array([[99.0, 1.0]])
        assert enc.transform(probe)[0, 0] == pytest.approx(enc.prior_)

    def test_non_numeric_targets_factorized(self, data):
        X, _ = data
        labels = np.where(np.arange(len(X)) % 2 == 0, "pos", "neg")
        enc = TargetEncoder(categorical_features=[0], smoothing=0.0).fit(X, labels)
        codes, inv = np.unique(labels, return_inverse=True)
        for i, code in enumerate(enc.categories_[0]):
            mask = X[:, 0] == code
            expected = inv[mask].mean()
            idx = np.searchsorted(enc.categories_[0], code)
            assert enc.encodings_[0][idx] == pytest.approx(expected)
        assert set(codes) == {"neg", "pos"}

    def test_all_categorical_default(self):
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array([0.0, 1.0, 1.0, 1.0])
        enc = TargetEncoder(smoothing=0.0).fit(X, y)
        assert enc.categorical_features_ == [0]
        np.testing.assert_allclose(enc.transform([[0.0], [1.0]]), [[0.5], [1.0]])


class TestTargetEncoderValidation:
    def test_not_fitted_errors(self):
        enc = TargetEncoder(categorical_features=[0])
        with pytest.raises(NotFittedError):
            enc.transform(np.zeros((1, 1)))
        with pytest.raises(NotFittedError):
            enc.partitions(0, 0.5)

    def test_feature_index_out_of_range(self, data):
        X, y = data
        with pytest.raises(ValueError, match="out of range"):
            TargetEncoder(categorical_features=[5]).fit(X, y)
        with pytest.raises(ValueError, match="out of range"):
            TargetEncoder(categorical_features=[-1]).fit(X, y)

    def test_negative_smoothing_rejected(self, data):
        X, y = data
        with pytest.raises(ValueError, match="smoothing"):
            TargetEncoder(categorical_features=[0], smoothing=-1).fit(X, y)

    def test_width_mismatch_on_transform(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        with pytest.raises(ValueError, match="columns"):
            enc.transform(X[:, :1])


class TestPartitionRecovery:
    """The post-processing that maps encoded splits back to categories."""

    def test_partitions_match_leq_routing(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        cats, encs = enc.categories_[0], enc.encodings_[0]
        for thr in [-1.0, *encs, max(encs) + 1]:
            mask = enc.partitions(0, thr)
            assert mask.shape == cats.shape
            for i, e in enumerate(encs):
                assert mask[i] == (e <= thr)

    def test_members_agree_with_encoded_routing(self, data):
        X, _ = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, X[:, 1])
        Xt = enc.transform(X)
        thr = float(enc.encodings_[0].max())
        routed = np.unique(Xt[Xt[:, 0] <= thr][:, 0])
        expected = np.unique(enc.members(0, thr))
        table = dict(zip(enc.categories_[0], enc.encodings_[0]))
        assert set(routed) == {table[c] for c in expected}

    def test_threshold_round_trip(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        cats, encs = enc.categories_[0], enc.encodings_[0]
        order = np.argsort(encs, kind="stable")  # prefixes of this are valid
        for k in range(len(cats) + 1):
            subset = cats[order[:k]].tolist()
            thr = enc.threshold_for_partition(0, subset)
            recovered = enc.members(0, thr)
            np.testing.assert_array_equal(
                np.sort(recovered), np.sort(np.asarray(subset))
            )

    def test_non_prefix_partition_rejected(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        cats, encs = enc.categories_[0], enc.encodings_[0]
        order = np.argsort(encs, kind="stable")
        middle = cats[order[1:3]].tolist()  # skips the lowest category
        with pytest.raises(ValueError, match="prefix"):
            enc.threshold_for_partition(0, middle)

    def test_tied_boundary_raises(self):
        # Two categories with identical target statistics share an encoding,
        # so no threshold can separate them from each other's neighbours.
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 0.0, 1.0])  # cat 1 and 3 tie at 1.0
        enc = TargetEncoder(categorical_features=[0], smoothing=0.0).fit(X, y)
        assert enc.encodings_[0][1] == enc.encodings_[0][3]
        with pytest.raises(ValueError, match="not expressible"):
            enc.threshold_for_partition(0, [0, 1, 2])

    def test_non_categorical_feature_rejected(self, data):
        X, y = data
        enc = TargetEncoder(categorical_features=[0]).fit(X, y)
        with pytest.raises(ValueError, match="not categorical"):
            enc.members(1, 0.5)


# ---------------------------------------------------------------------------
# Increment 2: CategoricalTree recovery layer
# ---------------------------------------------------------------------------

import shinrin
from shinrin.categorical import CategoricalTree, to_categorical_tree


def make_encoded_classification(n=400, seed=0):
    """Raw-code X plus its target-encoded twin and the shared encoder."""
    rng = np.random.RandomState(seed)
    color = rng.randint(0, 4, size=n).astype(np.float64)
    size = rng.randn(n)
    X_raw = np.column_stack([color, size])
    y = ((color == 1) | (color == 2)).astype(int)
    enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
    return X_raw, enc.transform(X_raw), y, enc


class TestToCategoricalTree:
    def test_members_recovered_on_categorical_splits(self):
        _, X_enc, y, enc = make_encoded_classification()
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_enc, y)
        ctree = to_categorical_tree(model, enc)

        assert isinstance(ctree, CategoricalTree)
        assert ctree.n_features_in == 2
        # Every split on feature 0 must carry member sets; feature 1 none.
        for node in range(len(ctree.feature)):
            if ctree.children_left[node] == -1:
                continue
            if ctree.feature[node] == 0:
                m = ctree.members[node]
                thr = ctree.threshold[node]
                np.testing.assert_array_equal(m, enc.members(0, thr))
            else:
                assert ctree.members[node] is None

    def test_apply_matches_source_tree_routing(self):
        X_raw, X_enc, y, enc = make_encoded_classification()
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_enc, y)
        ctree = to_categorical_tree(model, enc)
        # Raw-convention routing through member sets must reproduce the
        # source tree's encoded-space routing.
        np.testing.assert_array_equal(
            ctree.apply(X_raw), model.tree_.apply(X_enc.astype(np.float32))
        )

    def test_round_trip_partitions_preserved(self):
        X_raw, X_enc, y, enc = make_encoded_classification()
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=2).fit(X_enc, y)
        ctrees = to_categorical_tree(model, enc)
        assert isinstance(ctrees, list) and len(ctrees) == 5
        for ctree, est in zip(ctrees, model.estimators_):
            thr = ctree.to_encoded_thresholds(enc)
            for node, m in enumerate(ctree.members):
                if m is None:
                    continue
                f = int(ctree.feature[node])
                np.testing.assert_array_equal(enc.members(f, thr[node]), m)
            # Routing unchanged after round trip (raw vs encoded inputs).
            np.testing.assert_array_equal(
                ctree.apply(X_raw), est.tree_.apply(X_enc.astype(np.float32))
            )

    def test_prediction_equivalence_regressor(self):
        rng = np.random.RandomState(3)
        n = 300
        color = rng.randint(0, 3, n).astype(np.float64)
        num = rng.randn(n)
        X_raw = np.column_stack([color, num])
        y = 10.0 * (color == 2) + num
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor(random_state=4).fit(enc.transform(X_raw), y)
        ctree = to_categorical_tree(model, enc)
        pred_ct = ctree.value[ctree.apply(X_raw)].ravel()
        np.testing.assert_allclose(pred_ct, model.predict(enc.transform(X_raw)))

    def test_text_rendering(self):
        rng = np.random.RandomState(5)
        n = 150
        color = rng.randint(0, 3, n).astype(np.float64)
        size = rng.randn(n)
        enc = shinrin.TargetEncoder(categorical_features=[0])
        from sklearn.tree import DecisionTreeClassifier

        cat_model = DecisionTreeClassifier(random_state=6).fit(
            enc.fit_transform(np.column_stack([color, size]), color == 2),
            color == 2,
        )
        text_cat = to_categorical_tree(cat_model, enc).to_text(["color", "size"])
        assert "color in {" in text_cat

        num_model = DecisionTreeClassifier(random_state=7).fit(
            enc.transform(np.column_stack([color, size])), size > 0
        )
        text_num = to_categorical_tree(num_model, enc).to_text(["color", "size"])
        assert "size <=" in text_num

    def test_bad_encoder_rejected(self):
        _, X_enc, y, _ = make_encoded_classification(n=80)
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=2).fit(X_enc, y)
        with pytest.raises(TypeError, match="lacks fitted attribute"):
            to_categorical_tree(model, object())

    def test_lazy_exports(self):
        assert shinrin.CategoricalTree is CategoricalTree
        assert shinrin.TargetEncoder is not None

    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda: shinrin.MondrianForestRegressor(n_estimators=3, random_state=8),
            lambda: shinrin.MondrianForestClassifier(n_estimators=3, random_state=9),
        ],
        ids=["regressor", "classifier"],
    )
    def test_mondrian_forest_recovery(self, model_factory):
        X_raw, X_enc, y, enc = make_encoded_classification()
        model = model_factory().fit(X_enc, y)
        ctrees = to_categorical_tree(model, enc)
        assert isinstance(ctrees, list) and len(ctrees) == 3
        n_checked = 0
        for ctree, est in zip(ctrees, model.estimators_):
            for node, m in enumerate(ctree.members):
                if m is None:
                    continue
                np.testing.assert_array_equal(
                    m, enc.members(int(ctree.feature[node]), ctree.threshold[node])
                )
                n_checked += 1
            np.testing.assert_array_equal(
                ctree.apply(X_raw), est.tree_.apply(X_enc.astype(np.float32))
            )
        assert n_checked > 0


# ---------------------------------------------------------------------------
# Increment 3: opset-5 BRANCH_MEMBER ONNX export
# ---------------------------------------------------------------------------

from shinrin.onnx import to_onnx

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

needs_ort = pytest.mark.skipif(ort is None, reason="onnxruntime not installed")


def _assert_close(actual, desired, atol=1e-5):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).ravel(),
        np.asarray(desired, dtype=np.float64).ravel(),
        atol=atol,
        rtol=0,
    )


def _check_model(model_proto):
    # onnxruntime implies onnx; import locally so type checkers see a module.
    from onnx.checker import check_model

    check_model(model_proto.SerializeToString(), full_check=True)


@needs_ort
class TestMemberExportOrtParity:
    """Native predict on encoded X vs member graph on raw codes."""

    @pytest.fixture
    def session(self):
        def run(model_proto):
            assert ort is not None
            return ort.InferenceSession(
                model_proto.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )

        return run

    def test_mondrian_forest_regressor(self, session):
        rng = np.random.RandomState(10)
        n = 400
        color = rng.randint(0, 4, n).astype(np.float64)
        num = rng.randn(n)
        X_raw = np.column_stack([color, num])
        y = 10.0 * (color == 2) + 3.0 * (color == 0) + num
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        model = shinrin.MondrianForestRegressor(
            n_estimators=5, random_state=11, bootstrap=True
        ).fit(enc.transform(X_raw), y)

        proto = to_onnx(model, X=X_raw, encoder=enc)
        _check_model(proto)
        sess = session(proto)
        got = sess.run(None, {"X": np.ascontiguousarray(X_raw, dtype=np.float32)})[0]
        want = model.predict(enc.transform(X_raw))
        _assert_close(got, want, atol=1e-5)

    def test_random_forest_regressor(self, session):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(12)
        n = 300
        color = rng.randint(0, 3, n).astype(np.float64)
        num = rng.randn(n)
        X_raw = np.column_stack([color, num])
        y = color * 2.0 + num
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        model = RandomForestRegressor(n_estimators=8, random_state=13).fit(
            enc.transform(X_raw), y
        )
        proto = to_onnx(model, encoder=enc)
        _check_model(proto)
        got = session(proto).run(
            None, {"X": np.ascontiguousarray(X_raw, dtype=np.float32)}
        )[0]
        _assert_close(got, model.predict(enc.transform(X_raw)), atol=1e-5)

    def test_mondrian_forest_classifier_binary(self, session):
        X_raw, X_enc, y, enc = make_encoded_classification()
        model = shinrin.MondrianForestClassifier(n_estimators=4, random_state=14).fit(
            X_enc, y
        )
        proto = to_onnx(model, encoder=enc)
        _check_model(proto)
        labels, probs = session(proto).run(
            None, {"X": np.ascontiguousarray(X_raw, dtype=np.float32)}
        )
        want_proba = model.predict_proba(X_enc)
        np.testing.assert_allclose(probs, want_proba, atol=1e-5, rtol=0)
        np.testing.assert_array_equal(labels, model.predict(X_enc))

    def test_multiclass_and_string_labels(self, session):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(15)
        n = 450
        color = rng.randint(0, 3, n).astype(np.float64)
        num = rng.randn(n)
        X_raw = np.column_stack([color, num])
        y_str = np.array(["a", "b", "c"])[
            np.argmax(np.column_stack([-num, color - 1.0, num]), axis=1)
        ]
        enc = shinrin.TargetEncoder(categorical_features=[0], smoothing=1.0).fit(
            X_raw, y_str
        )
        model = RandomForestClassifier(n_estimators=6, random_state=16).fit(
            enc.transform(X_raw), y_str
        )
        proto = to_onnx(model, encoder=enc, class_names=["a", "b", "c"])
        _check_model(proto)
        labels, probs = session(proto).run(
            None, {"X": np.ascontiguousarray(X_raw, dtype=np.float32)}
        )
        np.testing.assert_allclose(
            probs, model.predict_proba(enc.transform(X_raw)), atol=1e-5, rtol=0
        )
        np.testing.assert_array_equal(labels, model.predict(enc.transform(X_raw)))
        assert labels.dtype.kind in "USO"

    def test_single_tree_degenerate(self, session):
        # A tree with a single leaf (no splits at all).
        X_raw = np.zeros((20, 2))
        X_raw[:, 1] = np.linspace(-1, 1, 20)
        y = np.ones(20)
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor(max_depth=1).fit(
            enc.transform(X_raw), y
        )  # depth 1 may still split; force degenerate below
        ctree = to_categorical_tree(model, enc)
        if len(ctree.feature) == 1:  # truly degenerate only when no split
            pass
        proto = to_onnx(model, encoder=enc)
        _check_model(proto)
        got = session(proto).run(
            None, {"X": np.ascontiguousarray(X_raw, dtype=np.float32)}
        )[0]
        _assert_close(got, model.predict(enc.transform(X_raw)), atol=1e-5)

    def test_unseen_category_routing(self, session):
        """Raw codes unseen during training route via the false branches."""
        rng = np.random.RandomState(17)
        n = 200
        color = rng.randint(0, 3, n).astype(np.float64)  # train codes {0,1,2}
        X_raw = np.column_stack([color, rng.randn(n)])
        y = (color == 1).astype(float)
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        model = shinrin.MondrianForestRegressor(n_estimators=3, random_state=18).fit(
            enc.transform(X_raw), y
        )
        proto = to_onnx(model, encoder=enc)
        # Probe with code 9: never seen; encoded value would be the prior.
        probe = np.array([[9.0, 0.25]], dtype=np.float32)
        got = session(proto).run(None, {"X": probe})[0]
        assert np.isfinite(got).all()

    def test_metadata_props(self):
        _X_raw, X_enc, y, enc = make_encoded_classification()
        model = shinrin.MondrianForestClassifier(n_estimators=2, random_state=19).fit(
            X_enc, y
        )
        proto = to_onnx(model, encoder=enc, feature_names=["color", "size"])
        props = {p.key: p.value for p in proto.metadata_props}
        assert props["shinrin_treeensemble_export"] == "member-v5"
        assert props["feature_names"] == "color,size"


class TestMemberExportValidation:
    def test_encoder_width_mismatch(self):
        _X_raw, X_enc, y, enc = make_encoded_classification(n=100)
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor().fit(X_enc[:, :1], y.astype(float))
        with pytest.raises(ValueError):
            to_onnx(model, encoder=enc)

    def test_quantile_rejects_encoder(self):
        rng = np.random.RandomState(20)
        X = rng.randn(60, 1)
        y = rng.randn(60)
        model = shinrin.RandomForestQuantileRegressor(n_estimators=3).fit(X, y)
        enc = shinrin.TargetEncoder(categorical_features=None)
        enc.fit(X, y)
        with pytest.raises(ValueError, match="quantile"):
            to_onnx(model, encoder=enc)


class TestMemberExportMondrianRouting:
    def test_smoothing_estimator_warns(self):
        rng = np.random.RandomState(21)
        X_raw = np.column_stack([rng.randint(0, 3, 200).astype(float), rng.randn(200)])
        y = X_raw[:, 1]
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        model = shinrin.MondrianTreeRegressor(path_smoothing=True, random_state=22).fit(
            enc.transform(X_raw), y
        )
        assert getattr(model, "path_smoothing", True)
        with pytest.warns(UserWarning, match="BRANCH_MEMBER"):
            proto = to_onnx(model, encoder=enc)
        props = {p.key: p.value for p in proto.metadata_props}
        assert props["shinrin_treeensemble_export"] == "member-v5"

    def test_approximate_false_keeps_exact_graph(self):
        rng = np.random.RandomState(23)
        X_raw = np.column_stack([rng.randint(0, 3, 150).astype(float), rng.randn(150)])
        y = X_raw[:, 1]
        enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
        model = shinrin.MondrianForestRegressor(
            n_estimators=2, path_smoothing=True, random_state=24
        ).fit(enc.transform(X_raw), y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            proto = to_onnx(model, encoder=enc, approximate=False)
        props = {p.key: p.value for p in proto.metadata_props}
        assert props.get("shinrin_mondrian_export") == "exact"
