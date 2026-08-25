"""Tests for the CatBoost-style target encoder and partition recovery."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.validation import NotFittedError

from shinrin._encoding import TargetEncoder


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
