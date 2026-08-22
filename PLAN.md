# Plan: TabICL support in shinrin

Add [TabICLv2](https://github.com/soda-inria/tabicl) (tabular foundation model,
ICML 2026) to shinrin following the established TabM pattern: an optional torch
backend, a pure-NumPy reference backend, and Mojo kernels, all loading the same
pretrained weights, with numeric parity tests and careful benchmarks.

## Design decisions (agreed)

| Decision | Choice |
|---|---|
| Torch backend | **Own implementation** of the architecture (~600 lines), weights loaded from the HF checkpoint; only needs `torch` + `huggingface-hub`. Doubles as the numeric reference for parity testing. |
| Feature scope | **Full parity**: many-class support (mixed-radix + hierarchical classification), KV cache in every backend, quantile→distribution regression postprocessing, normalization ensemble. |
| Weight loading | **One-time conversion with torch**: `.ckpt` → `.npz` cached beside it; NumPy/Mojo backends load `.npz`. Without torch installed, raise an informative error explaining how to convert. |

## Background: TabICLv2 architecture

Inference-only model (fit stores data; predict = single forward pass). No
backward pass or optimizers needed — much simpler than the TabM trainer.
Runtime complexity O(n² + n·m²) in train rows n and features m.

Checkpoints (~110MB Lightning `.ckpt`) on Hugging Face repo **`jingang/TabICL`**:

| File | Purpose |
|---|---|
| `tabicl-classifier-v2-20260212.ckpt` | default classifier (v2) |
| `tabicl-regressor-v2-20260212.ckpt` | default regressor (v2) |
| `tabicl-classifier-v1.1-20250506.ckpt` | v1.1 (classification only) |
| `tabicl-classifier-v1-20250208.ckpt` | v1 (classification only) |

Three sequential stages (dims from v2 defaults; actual values read from ckpt
`hyper_parameters`):

1. **ColEmbedding** — per-dataset standardization (train-row stats); circular
   feature grouping (`feature_group_size=3`) → linear embed to `embed_dim=128`;
   target-aware embedding added to training rows (one-hot × learned matrix for
   classes, linear for regression); 3 induced self-attention blocks (ISAB):
   128 inducing vectors attend over *training rows only*, then rows attend over
   inducing vectors. First attention layer of each block uses SSMax
   (`qassmax-mlp-elementwise`).
2. **RowInteraction** — 4 learnable CLS tokens prepended per row → 3 pre-norm
   transformer blocks attending across feature groups within each row, with
   RoPE (`row_rope_base=100000`; interleaved rotation = v1 variant,
   non-interleaved = v2; flag stored in ckpt config). Last block computes CLS
   outputs only. LayerNorm + flatten 4×128 → `icl_dim=512` token per row.
3. **ICLearning** — add class/quantile embeddings again (512-dim one-hot ×
   matrix); 12 pre-norm self-attention blocks (8 heads) where test rows attend
   to training rows (+ themselves); final block runs test queries only.
   Output MLP → logits (`max_classes`) or 999 quantiles.

SSMax variants to port (checkpoint-dependent): `none`, `ssmax`,
`ssmax-mlp`, `ssmax-mlp-elementwise`, `qassmax-mlp`,
`qassmax-mlp-elementwise`.

### Ensembling & postprocessing

- `n_estimators=8` members: per-member feature shuffling (`latin`
  hypercube-style / permutation) and class shifting (`shift`); predictions
  averaged as **logits** by default (`average_logits=True`), else probabilities.
- Classification softmax temperature 0.9.
- **Many classes (>10)**: mixed-radix decomposition of class indices into
  ≤10-class subproblems during column embedding; hierarchical classification
  combines digit distributions at output (`support_many_classes=True`).
- **Regression**: model predicts 999 quantiles → enforce monotonicity (sort /
  isotonic PAVA) → tail estimation (exponential / GPD) → `QuantileToDistribution`
  → mean/median/quantile predictions.

### Preprocessing (upstream `_sklearn/preprocessing.py` semantics)

Categorical detection + ordinal encoding (string/object/category/bool;
integer columns treated as numerical), separate NaN category for categoricals,
mean imputation for numerical NaNs, outlier detection/clipping
(z-score threshold 4.0), normalization ensemble across members
(`norm_methods=None` → `["none", "power"]`; methods: `none`, `power`
Yeo-Johnson, `quantile`, `quantile_rtdl`, `robust`), feature shuffling,
class shifting. Reuse shinrin's existing `QuantileTransform` where possible.

### KV cache

Cache K/V projections of training data at:
1. ColEmbedding: K/V of the second attention layer of each ISAB block
2. ICLearning: K/V from training data at each layer

Built during `fit()` when enabled; reused across `predict()` calls. Must exist
in all three backends (natively allocated in Mojo).

## File layout

```
src/shinrin/tabicl.py               # public TabICLClassifier / TabICLRegressor
src/shinrin/_tabicl/
    __init__.py                     # keep minimal per AGENTS.md
    _backend.py                     # SHINRIN_TABICL_BACKEND=auto|numpy|torch|mojo
    _config.py                      # TabICLConfig dataclass parsed from ckpt hyper_parameters
    _checkpoint.py                  # hf_hub_download("jingang/TabICL"), .ckpt→.npz converter, npz loader
    _preprocess.py                  # normalization ensemble, outlier clip, shuffles, class shift
    _model_torch.py                 # own torch implementation (stages, ssmax variants, rope, kv cache)
    _model_numpy.py                 # NumPy mirror of every op (reference backend)
    _many_classes.py                # mixed-radix encode + hierarchical combine
    _quantile_dist.py               # PAVA isotonic, monotonicity, exp/GPD tails, QuantileToDistribution
    _kv_cache.py                    # backend-agnostic cache structures
    _mojo_backend.py                # Python adapter over native kernels
src/shinrin/_tabicl_kernels.mojo    # native inference kernels
scripts/benchmarks/bench_tabicl.py  # benchmark harness
scripts/benchmarks/TABICL_BENCHMARK.md
tests/test_tabicl.py                # behavior tests w/ tiny random synthetic weights (no network)
tests/test_tabicl_parity.py         # torch ↔ numpy ↔ mojo numeric parity
```

Mojo kernel notes (reuse `_tabm_kernels.mojo` patterns — see
`.agents/skills/tabm-mojo-kernels/SKILL.md`): GEMM register tiling
(`gemm_nt`/`gemm_nn`/`gemm_tn_acc`, SIMDW=8, f64 accumulation for reductions),
fused scaled-dot-product attention with SSMax scaling + RoPE, LayerNorm,
GELU-MLP, single workspace slab with computed offsets, `TaskGroup` threading
over row batches × ensemble members, zero-copy numpy interop via
`__array_interface__`, `PyInit__native_tabicl` export, `+16` buffer padding,
explicit alloc/free discipline.

## Packaging & build

- `pyproject.toml`: optional group `tabicl-bench = ["torch>=2.0", "huggingface-hub", "tabicl"]`
  (upstream package used only for benchmark comparison); runtime optional dep
  `huggingface-hub` for checkpoint download; weight conversion additionally
  requires torch.
- `Justfile`: `build-tabicl-mojo`, `test-tabicl-mojo` targets mirroring tabm ones.
- Lazy exports `TabICLClassifier` / `TabICLRegressor` in `shinrin/__init__.py`.
- Public API mirrors upstream defaults: `n_estimators=8`, `norm_methods=None`,
  `feat_shuffle_method="latin"`, `class_shuffle_method="shift"`,
  `outlier_threshold=4.0`, `softmax_temperature=0.9`, `average_logits=True`,
  `support_many_classes=True`, `batch_size=8`, `kv_cache=False`, `model_path`,
  `allow_auto_download=True`, `checkpoint_version`, `device=` (torch only),
  `random_state=42`.

## Implementation phases

1. **Skeleton + checkpoints** — `_backend`, `_config`, `_checkpoint`;
   pyproject groups; conversion utility (requires torch, caches `.npz`);
   Justfile targets; clear no-torch error path.
2. **Torch implementation** — full forward incl. all ssmax variants, both RoPE
   modes, KV cache; unit-tested with tiny random weights (no download).
3. **Preprocessing + estimator wrappers** — preprocessing port; classifier /
   regressor sklearn API; ensembling orchestration (shuffle/class-shift/logit
   average); many-class path; quantile postprocessing.
4. **NumPy backend** — op-for-op mirror of the torch model; parity tests.
5. **Mojo kernels** — CPU multithreaded inference; parity numpy↔mojo; native
   KV cache buffers.
6. **Benchmarks + docs** — harness, runs, results doc, README/docs updates.

## Testing & validation

Validation chain: upstream `tabicl` (optional) → our torch → our numpy → our
mojo; each link covered by parity tests (fp32 tolerances ~rtol 1e-3–1e-2).

- `tests/test_tabicl.py`: fit/predict API, shapes, determinism, many-class
  mapping, kv-cache reuse correctness — synthetic tiny weights, no network.
- `tests/test_tabicl_parity.py`: identical weights through all backends;
  classifier (≤10 and >10 classes) + regressor paths; kv-cache vs no-cache
  equivalence; skips when deps/kernels missing (mirrors `test_tabm_parity.py`).
- Optional slow/network-marked integration test against real checkpoint +
  upstream accuracy sanity check.

All commits must pass `just lint` (ruff + ty + clippy, zero warnings);
run `just format` before committing (AGENTS.md). Conventional commits
(e.g. `feat(tabicl): add checkpoint download and conversion`).

## Benchmark methodology (bench_tabicl.py)

- Grid: {300, 1000, 5000, 20000} samples × {10, 100} features,
  classification + regression, plus one mixed categorical case.
- Backends: `numpy`, `mojo`, `torch`; optional `--with-upstream` compares
  against pip `tabicl` (needs `tabicl-bench` extras).
- Timing covers predict forward pass separately from fit/preprocessing;
  warmup run then ≥3 timed repeats reporting mean ± std; report peak RSS;
  record thread counts / hardware in TABICL_BENCHMARK.md.
- Accuracy sanity: scores must be non-degenerate (e.g. > majority baseline)
  for each backend — guards against silent weight-layout bugs.

## Explicitly out of scope

Forecaster (`TabICLForecaster`), fine-tuning, SHAP module, pre-training/prior
code generation, GPU offloading / disk-offload machinery (numpy/mojo are
CPU-resident; torch backend exposes `device=`), FlashAttention-3 (torch SDPA
instead).
