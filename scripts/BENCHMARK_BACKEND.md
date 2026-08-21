# Benchmark: Rust vs Mojo Backends

Apple Silicon (aarch64 macOS), Mojo 1.0.0, best of 5 repeats, identical data & seeds.

## Results

| case | rust | mojo | mojo vs rust |
|---|---|---|---|
| **small** — n=1,000, f=10 | | | |
| fit tree | 0.37 ms | 0.69 ms | 0.54x |
| predict ×5 | 0.26 ms | 0.36 ms | 0.72x |
| partial_fit (4 chunks) | 1.55 ms | 1.70 ms | 0.91x |
| **medium** — n=10,000, f=20 | | | |
| fit tree | 4.69 ms | 5.97 ms | 0.79x |
| predict ×5 | 3.86 ms | 7.22 ms | 0.53x |
| partial_fit (4 chunks) | 24.09 ms | 33.69 ms | 0.72x |
| **large** — n=50,000, f=50 | | | |
| fit tree | 53.82 ms | 62.30 ms | 0.86x |
| predict ×5 | 41.77 ms | 64.98 ms | 0.64x |
| partial_fit (4 chunks) | 282.52 ms | 469.28 ms | 0.60x |

## Takeaways

- **Rust is currently ~1.2–1.9x faster across the board**; the gap narrows as dataset size grows for `fit` (0.54x → 0.86x) but stays roughly constant for `predict`.
- **`partial_fit` was the worst offender (0.12x)** until scratch-buffer reuse landed — now 0.6–0.9x. The remaining cost is per-node bounds copies in `extend()` and unvectorized inner loops.
- **Prediction gap is dominated by scalar traversal + per-node bound checks**, which should vectorize well with Mojo SIMD — the most promising optimization target.
- These are first-pass numbers for an alpha compiler; no attempt was made beyond scratch-buffer reuse to optimize the Mojo code yet.

## Reproduction

Run benchmarks with:

```bash
just bench-backends
```

> Note: `speedup > 1.0` would mean Mojo is faster.
