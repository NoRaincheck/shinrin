"""Core primitives for the native TabICL kernels.

Internal module consumed by ``shinrin._tabicl_kernels`` (build with
``just build-tabicl-mojo``). Contains the NumPy boundary helpers, scalar
math utilities and the GEMM micro-kernels. Code moved verbatim from
``_tabicl_kernels.mojo`` — keep behavior-preserving edits separate from
moves so regressions stay attributable.

The public ``gemm_nt`` / ``gemm_nn`` dispatchers split output rows across
pthread workers for large problems (bit-exact vs the serial row kernels,
since each C element is written by exactly one thread); small GEMMs run
inline. GELU uses the native float32 SIMD erf path.
"""

from std.ffi import external_call
from std.math import erf, exp, log, max, sqrt
from std.memory import alloc
from std.python import Python, PythonObject

comptime SIMDW = 8


# =============================================================================
# numpy interop helpers
# =============================================================================


def ptr_f32(arr: PythonObject) raises -> Pointer[Float32, MutUntrackedOrigin]:
    var addr = Int(py=arr.__array_interface__["data"][0])
    return Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def ptr_i64(arr: PythonObject) raises -> Pointer[Int, MutUntrackedOrigin]:
    var addr = Int(py=arr.__array_interface__["data"][0])
    return Pointer[Int, MutUntrackedOrigin](unsafe_from_address=addr)


def ptr_f64(arr: PythonObject) raises -> Pointer[Float64, MutUntrackedOrigin]:
    var addr = Int(py=arr.__array_interface__["data"][0])
    return Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def iface_dim(arr: PythonObject, axis: Int) raises -> Int:
    var shape = arr.__array_interface__["shape"]
    return Int(py=shape[axis])


def np_module() raises -> PythonObject:
    return Python.import_module("numpy")


# =============================================================================
# Math helpers
# =============================================================================


@always_inline
def fast_exp(x: Float32) -> Float32:
    return exp(Float64(x)).cast[DType.float32]()


@always_inline
def fast_log(x: Float32) -> Float32:
    return log(Float64(x)).cast[DType.float32]()


@always_inline
def gelu_scalar(x: Float32) -> Float32:
    # Exact GELU (torch nn.GELU default): 0.5*x*(1+erf(x/sqrt(2)))
    var cx = Float64(x)
    return (0.5 * cx * (1.0 + erf(cx * 0.7071067811865476))).cast[DType.float32]()


@always_inline
def gelu8(x: SIMD[DType.float32, SIMDW]) -> SIMD[DType.float32, SIMDW]:
    # Exact GELU (torch nn.GELU default): 0.5*x*(1+erf(x/sqrt(2))) using the
    # native float32 SIMD erf (elementwise, so per-lane results match a
    # width-1 call; ~1 ulp-level deviation from the float64 path only).
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))


# =============================================================================
# GEMM kernels
# =============================================================================


def gemm_nt_rows(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin], lo: Int, hi: Int):
    # rows [lo, hi) of C (m,n) = A (m,kk) @ B (n,kk)^T -- overwrites those rows.
    # Deterministic per-row: results identical regardless of row partitioning.
    #
    # The full-block case processes output elements in 4-row x 2-column
    # register tiles: eight independent FMA chains instead of four, which
    # hides multiply-add latency without changing any element's accumulation
    # order (sequential over kk in SIMDW steps, then the same scalar tail),
    # so results stay bit-identical to the historical single-column kernel.
    var it = lo
    while it < hi:
        if it + 4 <= hi:
            var ap0 = a + it * kk
            var ap1 = ap0 + kk
            var ap2 = ap0 + 2 * kk
            var ap3 = ap0 + 3 * kk
            var j = 0
            while j + 2 <= n:
                var bp0 = b + j * kk
                var bp1 = bp0 + kk
                var acc00 = SIMD[DType.float32, SIMDW](0.0)
                var acc01 = SIMD[DType.float32, SIMDW](0.0)
                var acc10 = SIMD[DType.float32, SIMDW](0.0)
                var acc11 = SIMD[DType.float32, SIMDW](0.0)
                var acc20 = SIMD[DType.float32, SIMDW](0.0)
                var acc21 = SIMD[DType.float32, SIMDW](0.0)
                var acc30 = SIMD[DType.float32, SIMDW](0.0)
                var acc31 = SIMD[DType.float32, SIMDW](0.0)
                var t = 0
                while t + SIMDW <= kk:
                    var b0 = bp0.unsafe_load[width=SIMDW](t)
                    var b1 = bp1.unsafe_load[width=SIMDW](t)
                    var a0 = ap0.unsafe_load[width=SIMDW](t)
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    var a1 = ap1.unsafe_load[width=SIMDW](t)
                    acc10 += a1 * b0
                    acc11 += a1 * b1
                    var a2 = ap2.unsafe_load[width=SIMDW](t)
                    acc20 += a2 * b0
                    acc21 += a2 * b1
                    var a3 = ap3.unsafe_load[width=SIMDW](t)
                    acc30 += a3 * b0
                    acc31 += a3 * b1
                    t += SIMDW
                var s00 = acc00.reduce_add()
                var s01 = acc01.reduce_add()
                var s10 = acc10.reduce_add()
                var s11 = acc11.reduce_add()
                var s20 = acc20.reduce_add()
                var s21 = acc21.reduce_add()
                var s30 = acc30.reduce_add()
                var s31 = acc31.reduce_add()
                while t < kk:
                    var b0 = bp0[t]
                    var b1 = bp1[t]
                    s00 += ap0[t] * b0
                    s01 += ap0[t] * b1
                    s10 += ap1[t] * b0
                    s11 += ap1[t] * b1
                    s20 += ap2[t] * b0
                    s21 += ap2[t] * b1
                    s30 += ap3[t] * b0
                    s31 += ap3[t] * b1
                    t += 1
                c[it * n + j] = s00
                c[it * n + j + 1] = s01
                c[(it + 1) * n + j] = s10
                c[(it + 1) * n + j + 1] = s11
                c[(it + 2) * n + j] = s20
                c[(it + 2) * n + j + 1] = s21
                c[(it + 3) * n + j] = s30
                c[(it + 3) * n + j + 1] = s31
                j += 2
            while j < n:
                var bp = b + j * kk
                var acc0 = SIMD[DType.float32, SIMDW](0.0)
                var acc1 = SIMD[DType.float32, SIMDW](0.0)
                var acc2 = SIMD[DType.float32, SIMDW](0.0)
                var acc3 = SIMD[DType.float32, SIMDW](0.0)
                var t = 0
                while t + SIMDW <= kk:
                    var bv = bp.unsafe_load[width=SIMDW](t)
                    acc0 += ap0.unsafe_load[width=SIMDW](t) * bv
                    acc1 += ap1.unsafe_load[width=SIMDW](t) * bv
                    acc2 += ap2.unsafe_load[width=SIMDW](t) * bv
                    acc3 += ap3.unsafe_load[width=SIMDW](t) * bv
                    t += SIMDW
                var s0 = acc0.reduce_add()
                var s1 = acc1.reduce_add()
                var s2 = acc2.reduce_add()
                var s3 = acc3.reduce_add()
                while t < kk:
                    s0 += ap0[t] * bp[t]
                    s1 += ap1[t] * bp[t]
                    s2 += ap2[t] * bp[t]
                    s3 += ap3[t] * bp[t]
                    t += 1
                c[it * n + j] = s0
                c[(it + 1) * n + j] = s1
                c[(it + 2) * n + j] = s2
                c[(it + 3) * n + j] = s3
                j += 1
        else:
            var r = it
            while r < hi:
                var ap = a + r * kk
                var j = 0
                while j < n:
                    var bp = b + j * kk
                    var acc = SIMD[DType.float32, SIMDW](0.0)
                    var t = 0
                    while t + SIMDW <= kk:
                        acc += ap.unsafe_load[width=SIMDW](t) * bp.unsafe_load[width=SIMDW](t)
                        t += SIMDW
                    var s = acc.reduce_add()
                    while t < kk:
                        s += ap[t] * bp[t]
                        t += 1
                    c[r * n + j] = s
                    j += 1
                r += 1
        it += 4


@always_inline
def gemm_nn_rows(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin], lo: Int, hi: Int):
    # rows [lo, hi) of C (m,n) = A (m,kk) @ B (kk,n) -- overwrites those rows.
    var it = lo
    while it < hi:
        if it + 4 <= hi:
            var j = 0
            while j + 2 * SIMDW <= n:
                # Two independent column blocks: eight FMA chains instead of
                # four; each element's accumulation order is unchanged.
                var acc0 = SIMD[DType.float32, SIMDW](0.0)
                var acc1 = SIMD[DType.float32, SIMDW](0.0)
                var acc2 = SIMD[DType.float32, SIMDW](0.0)
                var acc3 = SIMD[DType.float32, SIMDW](0.0)
                var acb0 = SIMD[DType.float32, SIMDW](0.0)
                var acb1 = SIMD[DType.float32, SIMDW](0.0)
                var acb2 = SIMD[DType.float32, SIMDW](0.0)
                var acb3 = SIMD[DType.float32, SIMDW](0.0)
                var t = 0
                while t < kk:
                    var a0 = SIMD[DType.float32, SIMDW](a[it * kk + t])
                    var bva = b.unsafe_load[width=SIMDW](t * n + j)
                    var bvb = b.unsafe_load[width=SIMDW](t * n + j + SIMDW)
                    acc0 += a0 * bva
                    acb0 += a0 * bvb
                    var a1 = SIMD[DType.float32, SIMDW](a[(it + 1) * kk + t])
                    acc1 += a1 * bva
                    acb1 += a1 * bvb
                    var a2 = SIMD[DType.float32, SIMDW](a[(it + 2) * kk + t])
                    acc2 += a2 * bva
                    acb2 += a2 * bvb
                    var a3 = SIMD[DType.float32, SIMDW](a[(it + 3) * kk + t])
                    acc3 += a3 * bva
                    acb3 += a3 * bvb
                    t += 1
                var base = it * n + j
                c.unsafe_store[width=SIMDW](base, acc0)
                c.unsafe_store[width=SIMDW](base + n, acc1)
                c.unsafe_store[width=SIMDW](base + 2 * n, acc2)
                c.unsafe_store[width=SIMDW](base + 3 * n, acc3)
                c.unsafe_store[width=SIMDW](base + SIMDW, acb0)
                c.unsafe_store[width=SIMDW](base + n + SIMDW, acb1)
                c.unsafe_store[width=SIMDW](base + 2 * n + SIMDW, acb2)
                c.unsafe_store[width=SIMDW](base + 3 * n + SIMDW, acb3)
                j += 2 * SIMDW
            while j + SIMDW <= n:
                var acc0 = SIMD[DType.float32, SIMDW](0.0)
                var acc1 = SIMD[DType.float32, SIMDW](0.0)
                var acc2 = SIMD[DType.float32, SIMDW](0.0)
                var acc3 = SIMD[DType.float32, SIMDW](0.0)
                var t = 0
                while t < kk:
                    var bv = b.unsafe_load[width=SIMDW](t * n + j)
                    acc0 += SIMD[DType.float32, SIMDW](a[it * kk + t]) * bv
                    acc1 += SIMD[DType.float32, SIMDW](a[(it + 1) * kk + t]) * bv
                    acc2 += SIMD[DType.float32, SIMDW](a[(it + 2) * kk + t]) * bv
                    acc3 += SIMD[DType.float32, SIMDW](a[(it + 3) * kk + t]) * bv
                    t += 1
                var base = it * n + j
                c.unsafe_store[width=SIMDW](base, acc0)
                c.unsafe_store[width=SIMDW](base + n, acc1)
                c.unsafe_store[width=SIMDW](base + 2 * n, acc2)
                c.unsafe_store[width=SIMDW](base + 3 * n, acc3)
                j += SIMDW
            while j < n:
                var r = 0
                while r < 4:
                    var acc: Float32 = 0.0
                    var t = 0
                    while t < kk:
                        acc += a[(it + r) * kk + t] * b[t * n + j]
                        t += 1
                    c[(it + r) * n + j] = acc
                    r += 1
                j += 1
        else:
            var r = it
            while r < hi:
                var j = 0
                while j + SIMDW <= n:
                    var accr = SIMD[DType.float32, SIMDW](0.0)
                    var t = 0
                    while t < kk:
                        accr += (
                            SIMD[DType.float32, SIMDW](a[r * kk + t])
                            * b.unsafe_load[width=SIMDW](t * n + j)
                        )
                        t += 1
                    c.unsafe_store[width=SIMDW](r * n + j, accr)
                    j += SIMDW
                while j < n:
                    var acc: Float32 = 0.0
                    var t = 0
                    while t < kk:
                        acc += a[r * kk + t] * b[t * n + j]
                        t += 1
                    c[r * n + j] = acc
                    j += 1
                r += 1
        it += 4


@always_inline
def gemm_tn_acc(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) += A (kk,m)^T @ B (kk,n)  -- accumulates INTO C
    var it = 0
    while it + 4 <= m:
        var j = 0
        while j + SIMDW <= n:
            var acc0 = SIMD[DType.float32, SIMDW](0.0)
            var acc1 = SIMD[DType.float32, SIMDW](0.0)
            var acc2 = SIMD[DType.float32, SIMDW](0.0)
            var acc3 = SIMD[DType.float32, SIMDW](0.0)
            var t = 0
            while t < kk:
                var bv = b.unsafe_load[width=SIMDW](t * n + j)
                acc0 += SIMD[DType.float32, SIMDW](a[t * m + it]) * bv
                acc1 += SIMD[DType.float32, SIMDW](a[t * m + it + 1]) * bv
                acc2 += SIMD[DType.float32, SIMDW](a[t * m + it + 2]) * bv
                acc3 += SIMD[DType.float32, SIMDW](a[t * m + it + 3]) * bv
                t += 1
            var base = it * n + j
            var cur0 = c.unsafe_load[width=SIMDW](base)
            var cur1 = c.unsafe_load[width=SIMDW](base + n)
            var cur2 = c.unsafe_load[width=SIMDW](base + 2 * n)
            var cur3 = c.unsafe_load[width=SIMDW](base + 3 * n)
            c.unsafe_store[width=SIMDW](base, cur0 + acc0)
            c.unsafe_store[width=SIMDW](base + n, cur1 + acc1)
            c.unsafe_store[width=SIMDW](base + 2 * n, cur2 + acc2)
            c.unsafe_store[width=SIMDW](base + 3 * n, cur3 + acc3)
            j += SIMDW
        while j < n:
            var r = 0
            while r < 4:
                var acc: Float32 = 0.0
                var t = 0
                while t < kk:
                    acc += a[t * m + it + r] * b[t * n + j]
                    t += 1
                c[(it + r) * n + j] += acc
                r += 1
            j += 1
        it += 4
    while it < m:
        var j2 = 0
        while j2 < n:
            var acc: Float32 = 0.0
            var t = 0
            while t < kk:
                acc += a[t * m + it] * b[t * n + j2]
                t += 1
            c[it * n + j2] += acc
            j2 += 1
        it += 1


# =============================================================================
# Threaded GEMM dispatch (pthreads; output rows split across worker threads)
# =============================================================================

comptime P_U8 = Pointer[UInt8, MutUntrackedOrigin]
comptime GEMM_THREAD_MIN_FLOPS = 1 << 19  # ~0.5M MACs: below this threading overhead wins
comptime GEMM_THREAD_MIN_PANEL = 1 << 17  # B panel (n*kk) must exceed L1 so threads hide memory latency
comptime SYSCONF_NPROCESSORS_ONLN = 58  # Darwin value of _SC_NPROCESSORS_ONLN


struct GemmJob(Movable):
    var a: Pointer[Float32, MutUntrackedOrigin]
    var b: Pointer[Float32, MutUntrackedOrigin]
    var c: Pointer[Float32, MutUntrackedOrigin]
    var m: Int
    var n: Int
    var kk: Int
    var lo: Int
    var hi: Int
    var kind: Int  # 0 = nt (A@B^T), 1 = nn (A@B)

    def __init__(
        out self,
        a: Pointer[Float32, MutUntrackedOrigin],
        b: Pointer[Float32, MutUntrackedOrigin],
        c: Pointer[Float32, MutUntrackedOrigin],
        m: Int,
        n: Int,
        kk: Int,
        lo: Int,
        hi: Int,
        kind: Int,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.n = n
        self.kk = kk
        self.lo = lo
        self.hi = hi
        self.kind = kind


def hardware_threads() -> Int:
    # Number of online CPUs via sysconf(_SC_NPROCESSORS_ONLN). Cheap syscall;
    # globals are unsupported in Mojo so no caching.
    var nthreads = Int(external_call["sysconf", Int32](SYSCONF_NPROCESSORS_ONLN))
    if nthreads < 1:
        nthreads = 1
    return nthreads


@export("shinrin_gemm_worker")
def gemm_worker(raw: P_U8) abi("C") -> None:
    var jp = raw.unsafe_bitcast[GemmJob]()
    if jp[].kind == 0:
        gemm_nt_rows(jp[].m, jp[].n, jp[].kk, jp[].a, jp[].b, jp[].c, jp[].lo, jp[].hi)
    else:
        gemm_nn_rows(jp[].m, jp[].n, jp[].kk, jp[].a, jp[].b, jp[].c, jp[].lo, jp[].hi)


comptime GemmWorkerFn = def(P_U8) thin abi("C") -> None


def _gemm_dispatch(pool: UnsafePointer[ThreadPool, MutUntrackedOrigin], kind: Int, m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    var threads = hardware_threads()
    if (
        m < 24
        or threads < 2
        or m * n * kk < GEMM_THREAD_MIN_FLOPS
        or n * kk < GEMM_THREAD_MIN_PANEL
    ):
        if kind == 0:
            gemm_nt_rows(m, n, kk, a, b, c, 0, m)
        else:
            gemm_nn_rows(m, n, kk, a, b, c, 0, m)
        return

    var jobs = Pointer[GemmJob, MutUntrackedOrigin](alloc[GemmJob](threads))
    var chunk = (m + threads - 1) // threads
    var spawned = 0
    var lo = 0
    while spawned < threads and lo < m:
        var hi = lo + chunk
        if hi > m:
            hi = m
        jobs.unsafe_offset(spawned)[] = GemmJob(a, b, c, m, n, kk, lo, hi, kind)
        lo = hi
        spawned += 1

    comptime WorkerFn = GemmWorkerFn
    var fp: WorkerFn = gemm_worker
    run_partitioned_range(pool, jobs, spawned, fp)


@always_inline
def gemm_nt(pool: UnsafePointer[ThreadPool, MutUntrackedOrigin], m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) = A (m,kk) @ B (n,kk)^T -- overwrites C (threaded for large sizes)
    _gemm_dispatch(pool, 0, m, n, kk, a, b, c)


@always_inline
def gemm_nn(pool: UnsafePointer[ThreadPool, MutUntrackedOrigin], m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) = A (m,kk) @ B (kk,n) -- overwrites C (threaded for large sizes)
    _gemm_dispatch(pool, 1, m, n, kk, a, b, c)


# =============================================================================
# Generic partitioned-range runner (pthreads over job records)
# =============================================================================


def run_partitioned_range[
    T: AnyType
](
    pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
    jobs: Pointer[T, MutUntrackedOrigin],
    n_parts: Int,
    worker_fp: GemmWorkerFn,
):
    """Run ``worker_fp`` over ``n_parts`` consecutive job records.

    Record ``t`` lives at ``jobs.unsafe_offset(t)``; the worker reads its own
    work range out of the record. Partition 0 always runs on the calling
    thread. With a live pool, partitions 1..n_parts-1 are executed by the
    pool's parked workers (no thread creation); otherwise one pthread is
    spawned per remaining partition and joined, with a failed spawn running
    that partition synchronously so results stay correct regardless of
    thread availability.
    """
    if n_parts < 2:
        worker_fp(jobs.unsafe_bitcast[UInt8]())
        return

    if pool[].workers > 0:
        pool_run(pool, jobs, n_parts, worker_fp)
        return

    var threads = hardware_threads()
    if threads < 2:
        worker_fp(jobs.unsafe_bitcast[UInt8]())
        return

    var tids = alloc[P_U8](n_parts - 1)
    var spawned = 0  # successful spawns; also the next free tid slot
    var t = 1
    while t < n_parts and t < threads:
        var rc = external_call[
            "pthread_create",
            Int32,
            Pointer[P_U8, MutUntrackedOrigin],  # pthread_t *
            Int,                                # const pthread_attr_t * (NULL)
            GemmWorkerFn,                       # start routine
            P_U8,                               # void *arg
        ](
            tids.unsafe_offset(spawned),
            0,
            worker_fp,
            jobs.unsafe_offset(t).unsafe_bitcast[UInt8](),
        )
        if rc != 0:
            # Spawn failed: run this partition synchronously so results stay correct.
            worker_fp(jobs.unsafe_offset(t).unsafe_bitcast[UInt8]())
        else:
            spawned += 1
        t += 1

    worker_fp(jobs.unsafe_bitcast[UInt8]())

    t = 0
    while t < spawned:
        _ = external_call[
            "pthread_join",
            Int32,
            P_U8,  # pthread_t
            Int,   # void **retval (NULL)
        ](tids.unsafe_offset(t)[], 0)
        t += 1


# =============================================================================
# Persistent worker pool (scoped to one top-level entry-point call)
# =============================================================================
# A stack-anchored pool created once per exported entry-point invocation
# (predict / fit / cache build) and shut down at scope exit. Workers park on
# a condition variable between parallel regions, eliminating the per-region
# pthread_create/join cost; lifecycle stays deterministic and no global
# state is required. The Python GIL serializes entry points, so at most one
# pool executes work at any time even when several models are alive.

comptime _POOL_MUTEX_BYTES = 96
comptime _POOL_COND_BYTES = 96

comptime _PTHREAD_MUTEX_TIMED_NP = 0


struct PoolWorkerArg(Copyable, Movable):
    var pool: UnsafePointer[ThreadPool, MutUntrackedOrigin]
    var index: Int
    var job: P_U8  # rewritten by the dispatcher before every wake

    def __init__(
        out self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        index: Int,
    ):
        self.pool = pool
        self.index = index
        self.job = pool[].mu  # placeholder until first dispatch


struct ThreadPool(Movable):
    var mu: P_U8
    var work_cv: P_U8
    var done_cv: P_U8
    var tids: Pointer[P_U8, MutUntrackedOrigin]
    var args: Pointer[PoolWorkerArg, MutUntrackedOrigin]
    var seen: Pointer[Int, MutUntrackedOrigin]
    var workers: Int  # spawned workers; partitions beyond this run inline
    var gen: Int  # dispatch sequence; workers wake when it advances
    var done: Int  # workers that finished the current dispatch
    var shutdown: Bool
    var job_base: P_U8
    var parts: Int
    var worker_fn: GemmWorkerFn


def _pm_init(mut m: P_U8) -> None:
    _ = external_call["pthread_mutex_init", Int32](m, 0)


def _pm_lock(mut m: P_U8) -> None:
    _ = external_call["pthread_mutex_lock", Int32](m)


def _pm_unlock(mut m: P_U8) -> None:
    _ = external_call["pthread_mutex_unlock", Int32](m)


def _pm_destroy(mut m: P_U8) -> None:
    _ = external_call["pthread_mutex_destroy", Int32](m)


def _pcv_init(mut c: P_U8) -> None:
    _ = external_call["pthread_cond_init", Int32](c, 0)


def _pcv_wait(mut c: P_U8, mut m: P_U8) -> None:
    _ = external_call["pthread_cond_wait", Int32](c, m)


def _pcv_signal(mut c: P_U8) -> None:
    _ = external_call["pthread_cond_signal", Int32](c)


def _pcv_broadcast(mut c: P_U8) -> None:
    _ = external_call["pthread_cond_broadcast", Int32](c)


def _pcv_destroy(mut c: P_U8) -> None:
    _ = external_call["pthread_cond_destroy", Int32](c)


@export("shinrin_pool_worker")
def pool_worker(raw: P_U8) abi("C") -> None:
    var arg = raw.unsafe_bitcast[PoolWorkerArg]()
    var pool = arg[].pool
    var idx = arg[].index
    _pm_lock(pool[].mu)
    while True:
        while pool[].gen == pool[].seen.unsafe_offset(idx)[] and not pool[].shutdown:
            _pcv_wait(pool[].work_cv, pool[].mu)
        if pool[].shutdown:
            _pm_unlock(pool[].mu)
            return
        pool[].seen.unsafe_offset(idx)[] = pool[].gen
        # Snapshot this worker's job record while holding the lock: the next
        # dispatch cannot begin until every worker has counted done, so the
        # snapshot stays valid for the whole job. The dispatcher wrote our
        # partition's record pointer into args[idx].job before waking us.
        var my_job = arg[].job
        var parts_now = pool[].parts
        var fn_now = pool[].worker_fn
        _pm_unlock(pool[].mu)
        # Partition idx+1 (partition 0 runs on the dispatching thread).
        if idx + 1 < parts_now:
            fn_now(my_job)
        _pm_lock(pool[].mu)
        pool[].done += 1
        _pcv_signal(pool[].done_cv)


def pool_noop_worker(raw: P_U8) abi("C") -> None:
    pass


def pool_init(pool: UnsafePointer[ThreadPool, MutUntrackedOrigin], requested: Int) -> None:
    """Initialize ``pool`` and spawn up to ``requested - 1`` parked workers.

    ``requested <= 0`` means use every online CPU. With fewer than two CPUs
    the pool stays empty and every region runs inline.
    """
    var threads = hardware_threads()
    var n = threads - 1
    if requested > 0 and requested - 1 < n:
        n = requested - 1
    if n < 0:
        n = 0
    pool[].mu = alloc[Byte](_POOL_MUTEX_BYTES)
    pool[].work_cv = alloc[Byte](_POOL_COND_BYTES)
    pool[].done_cv = alloc[Byte](_POOL_COND_BYTES)
    pool[].tids = alloc[P_U8](n)
    pool[].args = alloc[PoolWorkerArg](n)
    pool[].seen = alloc[Int](n)
    pool[].gen = 0
    pool[].done = 0
    pool[].shutdown = False
    pool[].job_base = pool[].mu  # placeholder; unused by workers
    pool[].parts = 0
    pool[].worker_fn = pool_noop_worker
    pool[].workers = 0
    _pm_init(pool[].mu)
    _pcv_init(pool[].work_cv)
    _pcv_init(pool[].done_cv)
    var handle: GemmWorkerFn = pool_worker
    var i = 0
    while i < n:
        pool[].seen.unsafe_offset(i)[] = 0
        pool[].args.unsafe_offset(i)[] = PoolWorkerArg(pool, i)
        i += 1
    while pool[].workers < n:
        var rc = external_call[
            "pthread_create",
            Int32,
            Pointer[P_U8, MutUntrackedOrigin],
            Int,
            GemmWorkerFn,
            P_U8,
        ](
            pool[].tids.unsafe_offset(pool[].workers),
            0,
            handle,
            pool[].args.unsafe_offset(pool[].workers).unsafe_bitcast[UInt8](),
        )
        if rc != 0:
            return  # keep whatever workers spawned; regions degrade inline
        pool[].workers += 1


def pool_run[
    T: AnyType
](
    pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
    jobs: Pointer[T, MutUntrackedOrigin],
    parts: Int,
    worker_fn: GemmWorkerFn,
) -> None:
    """Execute partitions 0..parts-1; partition 0 runs on the caller.

    Per-partition record pointers are published through ``args[i].job``.
    """
    _pm_lock(pool[].mu)
    var w = 1
    while w <= pool[].workers and w < parts:
        pool[].args.unsafe_offset(w - 1)[].job = jobs.unsafe_offset(w).unsafe_bitcast[
            UInt8
        ]()
        w += 1
    pool[].parts = parts
    pool[].worker_fn = worker_fn
    pool[].done = 0
    pool[].gen += 1
    _pcv_broadcast(pool[].work_cv)
    _pm_unlock(pool[].mu)

    worker_fn(jobs.unsafe_bitcast[UInt8]())

    _pm_lock(pool[].mu)
    while pool[].done < pool[].workers:
        _pcv_wait(pool[].done_cv, pool[].mu)
    _pm_unlock(pool[].mu)


def pool_shutdown(pool: UnsafePointer[ThreadPool, MutUntrackedOrigin]) -> None:
    """Stop and join all workers, then release pool resources."""
    _pm_lock(pool[].mu)
    pool[].shutdown = True
    _pcv_broadcast(pool[].work_cv)
    _pm_unlock(pool[].mu)
    var i = 0
    while i < pool[].workers:
        _ = external_call[
            "pthread_join",
            Int32,
            P_U8,
            Int,
        ](pool[].tids.unsafe_offset(i)[], 0)
        i += 1
    _pm_destroy(pool[].mu)
    _pcv_destroy(pool[].work_cv)
    _pcv_destroy(pool[].done_cv)
    pool[].mu.unsafe_free()
    pool[].work_cv.unsafe_free()
    pool[].done_cv.unsafe_free()
    pool[].tids.unsafe_free()
    pool[].args.unsafe_free()
    pool[].seen.unsafe_free()
