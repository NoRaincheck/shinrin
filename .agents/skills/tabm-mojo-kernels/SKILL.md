---
name: tabm-mojo-kernels
description: Key learnings from building the TabM Mojo training kernels on feat/tabm (src/shinrin/_tabm_kernels.mojo + _tabm/_mojo_trainer.py), not present in main. Use when modifying, extending, debugging, or benchmarking the native TabM trainer, its Python bindings, build setup, or parity tests.
---

# TabM Mojo kernels — implementation learnings

Single file `src/shinrin/_tabm_kernels.mojo` compiles to `shinrin/_native_tabm.so`
(`just build-tabm-mojo` → `uv run mojo build ... --emit shared-lib`). Python side
adapter is `src/shinrin/_tabm/_mojo_trainer.py`; backend selection in
`_tabm/_backend.py` (`SHINRIN_TABM_BACKEND=auto|numpy|mojo`, auto = use `.so`
when present).

## Module structure

- Entry point must be `@export def PyInit__native_tabm() abi("C") -> PythonObject`;
  module name in `PythonModuleBuilder("_native_tabm")` must match the `PyInit`
  suffix and the built `.so` name.
- Bound type needs `(ImplicitlyCopyable, Movable, Writable)`; `Writable` forces
  a `write_to(writer)` method.
- Methods are registered via
  `PythonModuleBuilder(...).add_type[T](...).def_py_init[T.py_init]().def_method[T.method]("name")`.
- Bound methods have the shape
  `def m(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject`
  — self arrives as a pointer; copy with `self_ptr[]`.

## NumPy interop

| Rule | Detail |
|---|---|
| Zero-copy arrays | Read raw pointers from `arr.__array_interface__["data"][0]`; wrap via `Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)`. No Tensor/LayoutTensor anywhere. |
| Shapes | Read from `__array_interface__["shape"]`, never assume 2-D (`iface_dim(arr, axis)`) |
| Scalars | Convert with `Int(py=x)`, `Float64(py=x)`; return via `Python.tuple(...)`, `Python.none()` |
| Contiguity/dtype | Python side must pass `np.ascontiguousarray(x, dtype=np.float32)` |
| None inputs | Replace with 1-element dummy arrays before crossing (`_or_dummy`) — kernels never see `None` |
| Outputs | Preallocate in Python and pass in (e.g. `forward_avg` writes into `out`); allocate-and-return only for small results (grad arrays via `np.empty` created inside Mojo) |

Parameter layout mirrors `_layers.TabMParams.flatten` exactly:
`emb_w0, emb_b0, emb_wp_0..F-1, blk{i}_w/r/s/b, head_w, head_b`. Offsets are
computed once in the constructor into an `offs` table. If you change flatten
order, both sides break silently — parity tests catch it.

## Data-parallel design (perf rewrite)

- One `ChunkWorker` per thread, each owning a private `Workspace` + gradient
  buffer; runs forward → loss/dpreds → backward on a private contiguous row
  slice of each chunk.
- Threads come from `std.runtime.asyncrt.TaskGroup`: `tg.create_task(w.run())`
  where `run` is `async def`; then `tg.wait()`, then SIMD-reduce per-worker
  grads into one buffer; Adam updates only on the master thread afterwards.
- Worker count defaults to `num_performance_cores()`; `SHINRIN_TABM_THREADS`
  env override. Workers are allocated once per epoch and reused across rounds;
  free with an explicit `release()` after the loop.
- Loss denominators must be global to the round (pass `denom_b` down), not
  per-worker, so row-splitting doesn't change member-mean semantics. Return
  per-slice means; caller re-weights by slice size.

## Determinism

- Results are deterministic for fixed seed **and** thread count (not across
  thread counts).
- Per-worker stream derived as `mix_u64(seed ^ mix_u64(round_id * GOLDEN + t))`
  using splitmix64 finalizer; epoch shuffle is Fisher-Yates on the same master
  stream as the serial NumPy path so parity holds.
- Dropout packs 8 keep/drop decisions per 64-bit draw via byte thresholding
  (`keep_mask_from_bits`, P(drop) = thresh/256) instead of one draw per scalar.

## SIMD patterns that mattered (~8x speedup)

- `comptime SIMDW = 8` for f32; body loop `ptr.unsafe_load[width=SIMDW](i)` +
  `reduce_add()`, then scalar tail loop for `n % SIMDW`.
- Accumulate reductions (dot products, loss sums) in `Float64`, lanes in f32,
  to match NumPy within test tolerance (`rtol=1e-2, atol=1e-3`).
- GEMM register tiling: 4 output rows × SIMDW columns, four independent
  accumulators to break the FMA dependency chain; B row segment loaded once
  per 4 rows. Three layouts needed: `gemm_nt`, `gemm_nn`, `gemm_tn_acc`
  (backward accumulates INTO C).
- `fast_exp`: computes `exp` in Float64 and casts down to f32 — the sigmoid/
  softplus paths rely on this for accuracy.
- Every workspace buffer padded `+16` elements so vector loads/stores can
  overshoot safely.

## Workspace slab

Per-block scratch lives in one `alloc[Float32]` slab with computed offsets
(`vs_off`/`qs_off`/...) and pointer accessors: fewer allocations, better
locality, plain-pointer fields stay coroutine-friendly for async tasks.
Gotcha: the shared `tmp` buffer doubles as the PLE encoding snapshot, so it
must size to the widest bin count (`denc`), not just backbone widths.

## Hard-won gotchas

- Guard bin-count reads: embeddings-disabled configs send a zero-length bins
  array; index `f < nbins` or read OOB.
- Validate `sum(bin_counts) == d_enc` at construction — mismatch otherwise
  corrupts memory far away.
- Manual memory discipline: `alloc[T]` / `ptr.unsafe_free()` pairs everywhere;
  missing frees leak per call, double frees crash under threading.
- Env parsing happens in Mojo: `getenv("SHINRIN_TABM_THREADS")` returns
  `StringSlice`; check truthiness then `Int(String(env))`.

## Verification workflow

```
just build-tabm-mojo && just lint && uv run pytest tests/test_tabm_parity.py tests/test_tabm.py
```

Parity tests compare against the NumPy reference (`loss_grad`, L2 variant,
categorical variant, `forward_avg`); estimator-level determinism is covered by
`test_random_state_determinism`. Benchmarks live in
`scripts/benchmarks/TABM_BENCHMARK.md` (5k samples: ~27–50s serial → ~3.5–6s
parallel, vs torch ~6–7s).
