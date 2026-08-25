# Categorical Features & Target Encoding

shinrin handles categorical columns via **CatBoost-style ordered-less
target encoding**: categories are replaced by smoothed per-category target
statistics, and trees train on ordinary numeric thresholds. Because
encoded-threshold splits always separate *prefixes* of the
encoding-sorted categories, the original categorical split can be
recovered exactly afterwards — for interpretability (`CategoricalTree`)
and for deployment (`BRANCH_MEMBER` ONNX export).

## TargetEncoder

```python
import numpy as np
import shinrin

# column 0 holds integer category codes, column 1 is numeric
enc = shinrin.TargetEncoder(categorical_features=[0], smoothing=1.0)
X_enc = enc.fit_transform(X_raw, y)

model = shinrin.MondrianForestRegressor(n_estimators=20).fit(X_enc, y)
```

Each category `c` of feature `f` is encoded as

```
enc(c) = (sum_y(c) + smoothing * prior) / (count(c) + smoothing)
```

where `prior` is the global target mean. Rare categories shrink toward
the prior; unseen categories at transform time map to the prior.

Attributes after fitting:

| Attribute | Meaning |
|---|---|
| `categorical_features_` | Indices of encoded columns |
| `categories_` | Per-column sorted category values |
| `encodings_` | Per-column encoded values aligned with `categories_` |
| `prior_` | Global target mean used for smoothing |

## Recovering categorical splits

Splits on an encoded column partition the categories by their encoded
values. Since only prefixes of the encoding-sorted categories are
representable as a single threshold, every trained split corresponds to
exactly one membership set. `to_categorical_tree()` recovers it:

```python
ctree = shinrin.to_categorical_tree(model, enc)

# render human-readable rules over RAW inputs
print(ctree.to_text(feature_names=["color", "size"]))
#   x0 in {0.0, 2.0}
#   ├─ x1 <= 1.32  →  10.2
#   └─ ...

# apply() consumes raw pre-encoding samples
leaf = ctree.apply(X_raw)
```

For forests, `to_categorical_tree(model, enc)` returns a list (one
`CategoricalTree` per estimator). Each tree also round-trips back to the
encoded representation with `ctree.to_encoded_thresholds(enc)`, so both
views stay interchangeable.

## Exporting to ONNX without the encoder

The recovered partitions let the ONNX export use `ai.onnx.ml` opset-5
`TreeEnsemble` `BRANCH_MEMBER` splits over raw category codes, removing
the encoder from the deployed graph entirely. See
[Categorical features & BRANCH_MEMBER](onnx-export.md#categorical-features-branch-member-opset-5).
