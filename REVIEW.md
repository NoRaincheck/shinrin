# Review: TabM ONNX Export

## Summary

Adds ONNX export support for the vendored TabM estimators (`TabMClassifier`,
`TabMRegressor`). A fitted TabM model is now converted to a single,
self-contained ONNX graph that reproduces the complete inference pipeline —
preprocessing, PLE embeddings, BatchEnsemble backbone, and ensemble-averaged
head — so deployment requires nothing but the `.onnx` file and raw feature
vectors.

```
TabMRegressor/TabMClassifier ──> shinrin.onnx.to_onnx / save_onnx ──> .onnx ──> onnxruntime
```

## Changes

| File | Change |
|---|---|
| `src/shinrin/_tabm_onnx.py` | **New.** Graph builder for TabM models (~550 lines). |
| `src/shinrin/onnx.py` | Routes TabM estimators to the new exporter; docstring updates. |
| `pyproject.toml` | New `onnx` optional extra (`pip install shinrin[onnx]`, previously advertised in the README but missing); `onnx`, `onnxruntime`, `scikit-learn` added to the dev group so the full test suite runs out of the box. |
| `tests/test_tabm_onnx.py` | **New.** 25 tests: onnxruntime parity across architectures/tasks/configurations, edge cases, API behavior. |
| `src/shinrin/_backend.py`, `_mlp/_backend.py`, `_tabm/_backend.py` | `import mojo.importer` → `importlib.import_module("mojo.importer")`. Behavior identical; resolves pre-existing static-check failures without depending on whether Mojo is installed. |
| `src/shinrin/onnx_import.py`, `tests/test_onnx_*.py`, `src/shinrin/onnx.py` | Removed stale `# ty: ignore[unresolved-import]` directives (unused since onnx is now a dev dependency) and fixed one latent typing issue in `make_graph` output list. |
| `README.md`, `docs/features/onnx-export.md` | Document TabM export, outputs, and limitations. |

## Design decisions

**Self-contained graph with preprocessing baked in.** The alternative —
requiring callers to run `_Preprocessor.transform()` themselves — would leak
private API into deployment code. All fitted state (quantile boundaries,
scaler mean/std, bin edges, category tables) is captured as initializers.

**Standard-domain ops only, opset 15** (matches the existing tree exporter's
default). No `ai.onnx.ml` ops, and **no `Einsum`**: member-batched matmuls
are expressed as `Transpose` + broadcasted `MatMul` pairs, which every
runtime we know of supports.

**Vectorized PLE encoding.** Features can have different bin counts; instead
of emitting per-feature node chains, all features are encoded in one padded
computation `(B, F, E_max)` with rule masks (first/interior/last/single-bin)
and zero-padded weight rows. Graph size is independent of feature count.

**Categorical encoding via bucketization**, not float equality (`Equal`
didn't support floats before opset 19): index = `Σ[v_j ≤ x] − 1` clamped to
`[0, C−1]`, then `OneHot`.

## Subtleties worth reviewing

1. **Raw vs transformed numeric inputs** (`_Preprocessor.transform` returns
   *raw* `x_num` next to encodings of its *transformed* copy). Consequently
   the linear embedding term consumes raw values while the piecewise-linear
   branch consumes quantile/asinh/scaler-transformed values. An earlier
   draft fed transformed values to both and drifted ~0.07 absolute; the
   exporter now mirrors this split explicitly.
2. **`head_b` needs an explicit member axis** — `(k, d_out)` would silently
   broadcast against batch `B`; it is unsqueezed to `(k, 1, d_out)`.
3. **Single-output regression is squeezed** to `(n_samples,)` in-graph to
   match the estimator's sklearn-compatible `predict` shape.
4. **Binary classification** stacks `[1 − p, p]` via sigmoid of the single
   logit (d_out = 1), matching `predict_proba`'s column order.

## Verification

- **Parity vs NumPy reference** (onnxruntime CPU): max absolute deviation
  ≈ **5e-7** on regression outputs and ≈ **1e-7** on probabilities —
  float32 round-off level. Verified across `tabm` / `tabm-mini` /
  `tabm-packed`, binary/multiclass/regression, embeddings on/off,
  preprocessing on/off, mixed categorical features, constant features
  (single-bin PLE), and dynamic batch sizes 1…128.
- `tests/test_tabm_onnx.py`: **25 passed** (skipped cleanly when
  onnx/onnxruntime/sklearn are absent).
- Full suite: **178 passed, 33 skipped** (`tests/test_mlp.py` and
  `tests/test_skrules.py` fail identically on `main` in this environment:
  a stale locally-built `_native_mlp.so` and a missing pandas extra — both
  unrelated to this change).
- `just format` and `just lint` pass with zero diagnostics (clippy, ruff,
  ty). This includes fixing four small pre-existing ty failures so the
  branch is lint-clean end to end.

## Known limitations

- Input tensor is float32; float64 inputs must be cast by the caller
  (`validate_data` already does this inside shinrin).
- NaN handling matches training-time semantics only loosely: unseen/out-of-range
  categorical values clamp into the nearest bucket rather than falling back to
  index 0 exactly as the Python dict lookup does for high outliers.
- Dropout is inference-inactive and not represented in the graph.
- `labels` are int64 indices into `classes_` unless `class_names` is passed,
  in which case string labels are gathered in-graph.
