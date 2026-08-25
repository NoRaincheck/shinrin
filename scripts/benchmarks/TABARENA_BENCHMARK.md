# TabArena Benchmark

Wall-clock performance of the runnable shinrin algorithms on a core
subset of [TabArena](https://arxiv.org/abs/2506.16791)-v0.1, the living
tabular ML benchmark ([OpenML suite 457](https://www.openml.org/s/457)).
The core subset spans regression, binary and multiclass classification
as well as numeric-only and categorical-heavy feature spaces; every
dataset has at most ~5k rows. Produced by
`scripts/benchmarks/bench_tabarena.py`; regenerate locally with:

```bash
uv run --extra pandas python scripts/benchmarks/bench_tabarena.py
```

## Environment

| | |
|---|---|
| Date (UTC) | 2026-08-25 |
| OS | Darwin 25.6.0 |
| CPU | Apple M1 Max |
| Cores | 10 |
| Python | 3.14.3 |
| shinrin | 0.2.0 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.0 |
| Backends | Rust (Mondrian); NumPy reference (MLP, TabM) |

## Datasets

| Dataset | Task | Type | Rows | Features | % categorical |
|---|---|---|---|---|---|
| fish-toxicity | regression | regression | 908 | 6 | 0.0 |
| concrete-strength | regression | regression | 1,030 | 8 | 0.0 |
| insurance-expenses | regression | regression | 1,338 | 6 | 42.9 |
| airfoil-noise | regression | regression | 1,503 | 5 | 16.7 |
| fiat-500 | regression | regression | 1,538 | 7 | 12.5 |
| blood-transfusion | classification | binary | 748 | 4 | 20.0 |
| pima-diabetes | classification | binary | 768 | 8 | 11.1 |
| credit-g | classification | binary | 1,000 | 20 | 66.7 |
| qsar-biodeg | classification | binary | 1,054 | 41 | 14.3 |
| seismic-bumps | classification | binary | 2,584 | 15 | 25.0 |
| churn | classification | binary | 5,000 | 19 | 25.0 |
| anneal | classification | multiclass | 898 | 38 | 84.6 |
| maternal-health | classification | multiclass | 1,014 | 6 | 14.3 |
| website-phishing | classification | multiclass | 1,353 | 9 | 100.0 |

## Methodology

- Datasets are fetched from OpenML by their TabArena dataset id (cached under `~/scikit_learn_data`); categorical features are ordinal-encoded and numeric features median-imputed using training-fold statistics only. Test categories unseen during training map to a reserved trailing code.
- Single stratified 80/20 train/test split (`random_state=0`). TabArena itself uses 3 outer folds x 10 repeats with tuned models, so scores here are **not** comparable with the [public leaderboard](https://tabarena.ai).
- Fit time: best of 3 fits (all core-subset datasets are below the 5,000-row best-of-N threshold).
- Predict time: mean over 10 predictions of the held-out test set; also normalized per 1,000 samples.
- Score: R^2 on the test set for regression, accuracy for classification.
- All algorithms run single-threaded on identical splits. MLP and TabM use their pure-NumPy reference backends; Mondrian models use the Rust backend.
- Model configurations match `bench_all.py`: MondrianTree depth 16; MondrianForest 20 trees, depth 16; RandomForest / ExtraTrees 100 trees; quantile forests 50 trees; MLP (128, 64) hidden units, 100 Adam epochs; TabM (256, 256) hidden units, 60 Adam epochs.
- Scores reflect these fixed budgets, not tuned optima.
- GOSDT runs behind the threshold-guessing binarization pipeline (`depth_budget=4`, 60 s search limit); CORELS on quantile one-hot binarized features (`max_card=1`); both binary-classification only.
- Not included: SkopeRules (requires optional `pandas` at fit time), TabICL (requires torch plus a downloaded checkpoint).

Times are seconds unless stated otherwise; predict columns are
milliseconds per full test-set call / per 1,000 samples.

## Regression

*Fit time (seconds).*

| Dataset | MondrianTree | MondrianForest | RandomForest | ExtraTrees | RF-Quantile | ET-Quantile | MLP | TabM |
|---|---|---|---|---|---|---|---|---|
| fish-toxicity | 0.000498 | 0.00961 | 0.129 | 0.0841 | 0.202 | 0.177 | 0.0764 | 18.7 |
| concrete-strength | 0.000593 | 0.0117 | 0.163 | 0.109 | 0.25 | 0.22 | 0.0918 | 22.5 |
| insurance-expenses | 0.000819 | 0.0141 | 0.141 | 0.111 | 0.317 | 0.294 | 0.109 | 27.1 |
| airfoil-noise | 0.000753 | 0.0156 | 0.144 | 0.118 | 0.359 | 0.337 | 0.135 | 32.6 |
| fiat-500 | 0.00084 | 0.0158 | 0.191 | 0.133 | 0.339 | 0.316 | 0.125 | 32.2 |

*Predict: ms per full test-set call / ms per 1k samples.*

| Dataset | MondrianTree | MondrianForest | RandomForest | ExtraTrees | RF-Quantile | ET-Quantile | MLP | TabM |
|---|---|---|---|---|---|---|---|---|
| fish-toxicity | 0.0705 / 0.388 | 1.41 / 7.77 | 5.14 / 28.2 | 5.38 / 29.5 | 2.52 / 13.8 | 2.58 / 14.2 | 0.174 / 0.956 | 5.92 / 32.5 |
| concrete-strength | 0.0809 / 0.393 | 1.65 / 8.03 | 5.29 / 25.7 | 5.66 / 27.5 | 2.76 / 13.4 | 2.9 / 14.1 | 0.189 / 0.918 | 7.51 / 36.5 |
| insurance-expenses | 0.102 / 0.382 | 2.21 / 8.24 | 6.06 / 22.6 | 6.6 / 24.6 | 3.12 / 11.6 | 3.15 / 11.7 | 0.302 / 1.13 | 8.77 / 32.7 |
| airfoil-noise | 0.111 / 0.369 | 2.51 / 8.35 | 6.48 / 21.5 | 7.02 / 23.3 | 3.31 / 11 | 3.46 / 11.5 | 0.331 / 1.1 | 10.3 / 34.2 |
| fiat-500 | 0.104 / 0.339 | 2.34 / 7.61 | 6.38 / 20.7 | 7.25 / 23.5 | 3.33 / 10.8 | 3.34 / 10.8 | 0.3 / 0.976 | 10.4 / 33.6 |

*R^2 on the held-out test set.*

| Dataset | MondrianTree | MondrianForest | RandomForest | ExtraTrees | RF-Quantile | ET-Quantile | MLP | TabM |
|---|---|---|---|---|---|---|---|---|
| fish-toxicity | 0.1246 | 0.5593 | 0.5936 | 0.5568 | 0.6031 | 0.5902 | 0.5059 | 0.5134 |
| concrete-strength | 0.6306 | 0.8422 | 0.9037 | 0.9208 | 0.9030 | 0.9090 | 0.3573 | 0.3688 |
| insurance-expenses | -0.6839 | 0.4749 | 0.8340 | 0.8222 | 0.8346 | 0.8385 | 0.1570 | 0.1527 |
| airfoil-noise | 0.0323 | 0.3817 | 0.9444 | 0.9605 | 0.9431 | 0.9457 | 0.6299 | 0.2392 |
| fiat-500 | 0.7514 | 0.8147 | 0.8460 | 0.8388 | 0.8450 | 0.8413 | -5.3630 | 0.6191 |

## Classification

*Fit time (seconds).*

| Dataset | MondrianTree | MondrianForest | MLP | TabM | GOSDT | CORELS |
|---|---|---|---|---|---|---|
| blood-transfusion | 0.000424 | 0.00872 | 0.0785 | 16.9 | 0.00112 | 0.00115 |
| pima-diabetes | 0.000536 | 0.0103 | 0.0517 | 17 | 0.00523 | 0.0151 |
| credit-g | 0.000724 | 0.0157 | 0.105 | 23.2 | 0.0408 | 0.0106 |
| qsar-biodeg | 0.00106 | 0.0212 | 0.15 | 38.4 | 0.0414 | 0.0134 |
| seismic-bumps | 0.00118 | 0.0252 | 0.252 | 59.3 | 0.00233 | 0.00214 |
| churn | 0.00293 | 0.0548 | 0.376 | 137 | 0.0037 | 0.0457 |
| anneal | 0.000793 | 0.0183 | 0.0722 | 31.8 | - | - |
| maternal-health | 0.000607 | 0.0108 | 0.112 | 32.5 | - | - |
| website-phishing | 0.000646 | 0.0132 | 0.12 | 39.3 | - | - |

*Predict: ms per full test-set call / ms per 1k samples.*

| Dataset | MondrianTree | MondrianForest | MLP | TabM | GOSDT | CORELS |
|---|---|---|---|---|---|---|
| blood-transfusion | 0.0822 / 0.548 | 0.875 / 5.83 | 0.302 / 2.01 | 5.33 / 35.5 | 0.0708 / 0.472 | 0.00989 / 0.0659 |
| pima-diabetes | 0.0898 / 0.583 | 0.906 / 5.88 | 0.235 / 1.53 | 5.52 / 35.8 | 0.0897 / 0.582 | 0.011 / 0.0714 |
| credit-g | 0.123 / 0.617 | 1.29 / 6.44 | 0.671 / 3.36 | 7.45 / 37.3 | 0.0911 / 0.455 | 0.0134 / 0.0669 |
| qsar-biodeg | 0.0891 / 0.422 | 0.916 / 4.34 | 0.863 / 4.09 | 11.4 / 54.1 | 0.11 / 0.52 | 0.0224 / 0.106 |
| seismic-bumps | 0.158 / 0.305 | 1.98 / 3.83 | 0.968 / 1.87 | 21 / 40.6 | 0.15 / 0.29 | 0.0187 / 0.0362 |
| churn | 0.324 / 0.324 | 3.88 / 3.88 | 0.998 / 0.998 | 44.4 / 44.4 | 0.291 / 0.291 | 0.0401 / 0.0401 |
| anneal | 0.106 / 0.589 | 1.36 / 7.54 | 0.963 / 5.35 | 16.3 / 90.4 | - | - |
| maternal-health | 0.0986 / 0.486 | 1.13 / 5.59 | 0.362 / 1.79 | 20.6 / 101 | - | - |
| website-phishing | 0.112 / 0.412 | 1.38 / 5.11 | 0.555 / 2.05 | 25.9 / 95.6 | - | - |

*Accuracy on the held-out test set.*

| Dataset | MondrianTree | MondrianForest | MLP | TabM | GOSDT | CORELS |
|---|---|---|---|---|---|---|
| blood-transfusion | 0.6933 | 0.6800 | 0.7467 | 0.7533 | 0.7600 | 0.7600 |
| pima-diabetes | 0.7468 | 0.7403 | 0.6429 | 0.7338 | 0.7727 | 0.7662 |
| credit-g | 0.6400 | 0.6750 | 0.7000 | 0.7100 | 0.7000 | 0.7000 |
| qsar-biodeg | 0.7820 | 0.8341 | 0.8483 | 0.8389 | 0.7393 | 0.8246 |
| seismic-bumps | 0.9033 | 0.9207 | 0.9342 | 0.9342 | 0.9342 | 0.9342 |
| churn | 0.8540 | 0.8800 | 0.8770 | 0.9040 | 0.8590 | 0.8590 |
| anneal | 0.8278 | 0.8611 | 0.8556 | 0.9222 | - | - |
| maternal-health | 0.7586 | 0.8177 | 0.7586 | 0.7438 | - | - |
| website-phishing | 0.8450 | 0.8745 | 0.8819 | 0.8635 | - | - |

## Skipped / failed cells

- `anneal` x `GOSDT`: skipped (binary targets only)
- `anneal` x `CORELS`: skipped (binary targets only)
- `maternal-health` x `GOSDT`: skipped (binary targets only)
- `maternal-health` x `CORELS`: skipped (binary targets only)
- `website-phishing` x `GOSDT`: skipped (binary targets only)
- `website-phishing` x `CORELS`: skipped (binary targets only)

