"""Mojo training kernels for the vendored TabM model (arch_type='tabm').

Compiles to the ``shinrin._native_tabm`` Python extension module (build
with ``just build-tabm-mojo``). Exposes a ``TabMTrainer`` bound type whose
methods run complete training steps without returning to Python:

- ``adam_epoch(parts)``: one shuffled minibatch Adam epoch (dropout included)
- ``lbfgs_minimize(parts)``: full-batch L-BFGS with backtracking line search
- ``loss_grad(parts)``: full-batch loss + gradient (parity testing)
- ``forward_avg(parts)``: k-member-averaged predictions into a preallocated array

All arrays cross the boundary as NumPy buffers accessed through raw
pointers (float32 data, int64 dims). The parameter layout matches
``shinrin._tabm._layers.TabMParams.flatten`` exactly:

    emb_w0, emb_b0, emb_wp_0..F-1,               (when embeddings enabled)
    blk{i}_w, blk{i}_r, blk{i}_s, blk{i}_b,      for i in 0..n_blocks-1
    head_w, head_b

Loss conventions mirror ``shinrin._tabm._model.TabMCore.loss_and_grads``:
regression loss is the mean over B*k*d_out elements; binary/multiclass
losses are means over B*k members. L2 adds alpha*theta to gradients and
0.5*alpha*||theta||^2 to the loss.
"""

from std.os import abort, getenv
from std.math import abs, exp, log, pow, sqrt
from std.memory import alloc
from std.io import Writer
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime.asyncrt import TaskGroup
from std.sys.info import num_performance_cores

comptime SIMDW = 8


# =============================================================================
# numpy interop helpers (same pattern as _native_mojo.mojo)
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


def np_empty_1d(np: PythonObject, n: Int, dtype: String) raises -> PythonObject:
    return np.empty(Python.tuple(Int(n)), dtype)


# =============================================================================
# small math helpers
# =============================================================================


@always_inline
def fast_exp(x: Float32) -> Float32:
    return exp(Float64(x)).cast[DType.float32]()


@always_inline
def fast_log(x: Float32) -> Float32:
    return log(Float64(x)).cast[DType.float32]()


@always_inline
def sigmoid(x: Float32) -> Float32:
    if x >= 0:
        return 1.0 / (1.0 + fast_exp(-x))
    var e = fast_exp(x)
    return e / (1.0 + e)


@always_inline
def softplus(x: Float32) -> Float32:
    if x > 20.0:
        return x
    if x < -20.0:
        return fast_exp(x)
    return fast_log(1.0 + fast_exp(x))


@always_inline
def mix_u64(x: UInt64) -> UInt64:
    """splitmix64 finalizer: derives independent per-worker RNG streams."""
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def resolve_threads() raises -> Int:
    """Worker count for data-parallel kernels.

    Defaults to the number of performance cores; override with the
    SHINRIN_TABM_THREADS environment variable.
    """
    var t = num_performance_cores()
    if t < 1:
        t = 1
    var env = getenv("SHINRIN_TABM_THREADS")
    if env:
        var parsed = Int(String(env))
        if parsed >= 1:
            t = parsed
    return t


@always_inline
def keep_mask_from_bits(v: UInt64, keep: Float32, thresh: UInt64) -> SIMD[DType.float32, SIMDW]:
    """Expand 8 dropout decisions (one byte each) from the top of v.

    A lane is kept iff its byte >= thresh, giving P(drop) = thresh/256
    (~1/2560 granular error vs the float-draw scheme at one 64-bit draw
    per 8 elements instead of eight).
    """
    var out = SIMD[DType.float32, SIMDW](0.0)
    comptime for lane in range(SIMDW):
        if ((v >> UInt64(56 - 8 * lane)) & 0xFF) >= thresh:
            out[lane] = keep
    return out


@always_inline
def vec_zero(dst: Pointer[Float32, MutUntrackedOrigin], n: Int):
    var i = 0
    var z = SIMD[DType.float32, SIMDW](0.0)
    while i + SIMDW <= n:
        dst.unsafe_store[width=SIMDW](i, z)
        i += SIMDW
    while i < n:
        dst.unsafe_store[width=1](i, 0.0)
        i += 1


@always_inline
def dot_f32(a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], n: Int) -> Float64:
    var acc: Float64 = 0.0
    var i = 0
    while i + SIMDW <= n:
        acc += Float64((a.unsafe_load[width=SIMDW](i) * b.unsafe_load[width=SIMDW](i)).reduce_add())
        i += SIMDW
    while i < n:
        acc += Float64(a.unsafe_load[width=1](i) * b.unsafe_load[width=1](i))
        i += 1
    return acc


@always_inline
def dot_f64(a: Pointer[Float64, MutUntrackedOrigin], b: Pointer[Float64, MutUntrackedOrigin], n: Int) -> Float64:
    var acc: Float64 = 0.0
    var i = 0
    while i + 4 <= n:
        acc += (a.unsafe_load[width=4](i) * b.unsafe_load[width=4](i)).reduce_add()
        i += 4
    while i < n:
        acc += a[i] * b[i]
        i += 1
    return acc


@always_inline
def sum_abs_f32(a: Pointer[Float32, MutUntrackedOrigin], n: Int) -> Float64:
    var acc: Float64 = 0.0
    var i = 0
    while i + SIMDW <= n:
        acc += Float64(abs(a.unsafe_load[width=SIMDW](i)).reduce_add())
        i += SIMDW
    while i < n:
        var v = a.unsafe_load[width=1](i)
        acc += Float64(v if v > 0.0 else -v)
        i += 1
    return acc


@always_inline
def max_abs_f32(a: Pointer[Float32, MutUntrackedOrigin], n: Int) -> Float64:
    var best: Float64 = 0.0
    var i = 0
    while i < n:
        var v = a.unsafe_load[width=1](i)
        var av = Float64(v if v > 0.0 else -v)
        if av > best:
            best = av
        i += 1
    return best


@always_inline
def sum_sq_f32(a: Pointer[Float32, MutUntrackedOrigin], n: Int) -> Float64:
    var acc: Float64 = 0.0
    var i = 0
    while i + SIMDW <= n:
        acc += Float64((a.unsafe_load[width=SIMDW](i) * a.unsafe_load[width=SIMDW](i)).reduce_add())
        i += SIMDW
    while i < n:
        acc += Float64(a.unsafe_load[width=1](i) * a.unsafe_load[width=1](i))
        i += 1
    return acc


@always_inline
def accumulate_into(dst: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin], n: Int):
    """dst += src elementwise (gradient reduction across workers)."""
    var i = 0
    while i + SIMDW <= n:
        dst.unsafe_store[width=SIMDW](
            i, dst.unsafe_load[width=SIMDW](i) + src.unsafe_load[width=SIMDW](i)
        )
        i += SIMDW
    while i < n:
        dst[i] = dst[i] + src[i]
        i += 1


@always_inline
def saxpy(dst: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin], scale: Float64, n: Int):
    # dst[i] += scale * src[i]
    var sc = SIMD[DType.float32, SIMDW](Float32(scale))
    var i = 0
    while i + SIMDW <= n:
        var d = dst.unsafe_load[width=SIMDW](i)
        d += sc * src.unsafe_load[width=SIMDW](i)
        dst.unsafe_store[width=SIMDW](i, d)
        i += SIMDW
    while i < n:
        dst.unsafe_store[width=1](i, dst.unsafe_load[width=1](i) + Float32(scale) * src.unsafe_load[width=1](i))
        i += 1


# =============================================================================
# GEMM kernels (row-major, SIMD over columns)
# =============================================================================


@always_inline
def gemm_nt(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) = A (m,kk) @ B (n,kk)^T   -- overwrites C
    # Tiled over 4 output rows sharing each B row segment; four independent
    # accumulators break the FMA dependency chain. Lanes stay within rows.
    var it = 0
    while it < m:
        var j = 0
        while j < n:
            var bp = b + j * kk
            if it + 4 <= m:
                var ap0 = a + it * kk
                var ap1 = a + (it + 1) * kk
                var ap2 = a + (it + 2) * kk
                var ap3 = a + (it + 3) * kk
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
            else:
                var r = it
                while r < m:
                    var ap = a + r * kk
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
                    r += 1
            j += 1
        it += 4


@always_inline
def gemm_nn(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) = A (m,kk) @ B (kk,n)   -- overwrites C
    # Register-tiled as 4i x 8j blocks: each B row segment is loaded once
    # per 4 output rows and the four accumulators break the FMA chain.
    var it = 0
    while it < m:
        var j = 0
        while j + SIMDW <= n:
            if it + 4 <= m:
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
            else:
                var r = it
                while r < m:
                    var accr = SIMD[DType.float32, SIMDW](0.0)
                    var t = 0
                    while t < kk:
                        accr += SIMD[DType.float32, SIMDW](a[r * kk + t]) * b.unsafe_load[width=SIMDW](t * n + j)
                        t += 1
                    c.unsafe_store[width=SIMDW](r * n + j, accr)
                    r += 1
            j += SIMDW
        while j < n:
            var r = 0
            while r < 4 and it + r < m:
                var acc: Float32 = 0.0
                var t = 0
                while t < kk:
                    acc += a[(it + r) * kk + t] * b[t * n + j]
                    t += 1
                c[(it + r) * n + j] = acc
                r += 1
            j += 1
        it += 4


@always_inline
def gemm_tn_acc(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) += A (kk,m)^T @ B (kk,n)  -- accumulates INTO C
    # Register-tiled as 4i x 8j blocks: each B row segment is loaded once
    # per 4 output rows and the four accumulators break the FMA chain.
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
# RNG (xorshift64*)
# =============================================================================


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed ^ 0x9E3779B97F4A7C15
        if self.s == 0:
            self.s = 88172645463325252

    def next_u64(mut self) -> UInt64:
        self.s ^= self.s >> 12
        self.s ^= self.s << 25
        self.s ^= self.s >> 27
        return self.s * 2685821657736338717

    def uniform(mut self) -> Float32:
        return Float32((self.next_u64() >> 40) & 0xFFFFFF) * (1.0 / 16777216.0)


# =============================================================================
# Scratch workspace for one row-chunk
# =============================================================================


struct Workspace(Movable):
    var h: Pointer[Float32, MutUntrackedOrigin]
    var pl: Pointer[Float32, MutUntrackedOrigin]
    var tmp: Pointer[Float32, MutUntrackedOrigin]
    var tmp2: Pointer[Float32, MutUntrackedOrigin]
    var xnum: Pointer[Float32, MutUntrackedOrigin]
    var enc_snap: Pointer[Float32, MutUntrackedOrigin]
    var preds: Pointer[Float32, MutUntrackedOrigin]
    var dpreds: Pointer[Float32, MutUntrackedOrigin]
    var da: Pointer[Float32, MutUntrackedOrigin]
    var dq: Pointer[Float32, MutUntrackedOrigin]
    var dv: Pointer[Float32, MutUntrackedOrigin]
    var dh: Pointer[Float32, MutUntrackedOrigin]
    # Per-block caches live in one contiguous slab (fewer allocations, better
    # locality, and plain-pointer fields keep the struct coroutine-friendly).
    var slab: Pointer[Float32, MutUntrackedOrigin]
    var vs_off: Pointer[Int, MutUntrackedOrigin]
    var qs_off: Pointer[Int, MutUntrackedOrigin]
    var rm_off: Pointer[Int, MutUntrackedOrigin]
    var dm_off: Pointer[Int, MutUntrackedOrigin]
    var ac_off: Pointer[Int, MutUntrackedOrigin]
    var nb: Int

    def __init__(
        out self,
        chunk: Int,
        k: Int,
        nb: Int,
        d_in: Int,
        db: Int,
        dout: Int,
        F: Int,
        demb: Int,
        denc: Int,
    ):
        var bk = chunk * k
        var bmax = max(d_in, db)
        self.h = alloc[Float32](chunk * d_in + 16)
        self.pl = alloc[Float32](chunk * F * demb + 16)
        # tmp doubles as the (chunk, cnt) per-feature encoding snapshot in the
        # piecewise-linear forward/backward, so it must fit the widest bin
        # count (<= denc), not just the embedding/backbone widths.
        self.tmp = alloc[Float32](chunk * max(max(F * demb, d_in), denc) + 16)
        self.tmp2 = alloc[Float32](chunk * demb + 16)
        self.xnum = alloc[Float32](chunk * F + 16)
        self.enc_snap = alloc[Float32](chunk * denc + 16)
        self.preds = alloc[Float32](bk * dout + 16)
        self.dpreds = alloc[Float32](bk * dout + 16)
        self.da = alloc[Float32](bk * bmax + 16)
        self.dq = alloc[Float32](bk * db + 16)
        self.dv = alloc[Float32](bk * bmax + 16)
        self.dh = alloc[Float32](chunk * d_in + 16)
        self.nb = nb
        self.vs_off = alloc[Int](nb + 16)
        self.qs_off = alloc[Int](nb + 16)
        self.rm_off = alloc[Int](nb + 16)
        self.dm_off = alloc[Int](nb + 16)
        self.ac_off = alloc[Int](nb + 16)
        comptime PAD = 16
        var total = 0
        for i in range(nb):
            var b_in = d_in if i == 0 else db
            self.vs_off[i] = total
            total += bk * b_in + PAD
            self.qs_off[i] = total
            total += bk * db + PAD
            self.rm_off[i] = total
            total += bk * db + PAD
            self.dm_off[i] = total
            total += bk * db + PAD
            self.ac_off[i] = total
            total += bk * db + PAD
        self.slab = alloc[Float32](total)

    @always_inline
    def vs_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.vs_off[i]

    @always_inline
    def qs_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.qs_off[i]

    @always_inline
    def rmask_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.rm_off[i]

    @always_inline
    def dmask_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.dm_off[i]

    @always_inline
    def act_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.ac_off[i]

    def unsafe_free(self):
        self.h.unsafe_free()
        self.pl.unsafe_free()
        self.tmp.unsafe_free()
        self.tmp2.unsafe_free()
        self.xnum.unsafe_free()
        self.enc_snap.unsafe_free()
        self.preds.unsafe_free()
        self.dpreds.unsafe_free()
        self.da.unsafe_free()
        self.dq.unsafe_free()
        self.dv.unsafe_free()
        self.dh.unsafe_free()
        self.slab.unsafe_free()
        self.vs_off.unsafe_free()
        self.qs_off.unsafe_free()
        self.rm_off.unsafe_free()
        self.dm_off.unsafe_free()
        self.ac_off.unsafe_free()


# =============================================================================
# Trainer
# =============================================================================


struct TabMTrainer(ImplicitlyCopyable, Movable, Writable):
    def write_to(mut self, mut writer: Some[Writer]):
        writer.write("TabMTrainer(P=", self.P, ")")

    var k: Int
    var nb: Int
    var db: Int
    var dout: Int
    var F: Int
    var denc: Int
    var ccat: Int
    var use_emb: Bool
    var demb: Int
    var d_in: Int
    var bin_counts: Pointer[Int, MutUntrackedOrigin]
    var bin_offsets: Pointer[Int, MutUntrackedOrigin]
    var offs: Pointer[Int, MutUntrackedOrigin]
    var n_offs: Int
    var P: Int

    def __init__(out self, dims: PythonObject, bins: PythonObject) raises:
        var dp = ptr_i64(dims)
        self.k = Int(dp[0])
        self.nb = Int(dp[1])
        self.db = Int(dp[2])
        self.dout = Int(dp[3])
        self.F = Int(dp[4])
        self.denc = Int(dp[5])
        self.ccat = Int(dp[6])
        self.use_emb = Int(dp[7]) == 1
        self.demb = Int(dp[8])

        self.bin_counts = alloc[Int](self.F + 16)
        self.bin_offsets = alloc[Int](self.F + 17)
        var total_bins = 0
        self.bin_offsets[0] = 0
        var bp = ptr_i64(bins)
        var nbins = len(bins)
        for f in range(self.F):
            # Absent bins (e.g. embeddings disabled) arrive as a zero-length
            # array; treat every count as 0 rather than reading out of bounds.
            var cnt = Int(bp[f]) if f < nbins else 0
            self.bin_counts[f] = cnt
            total_bins += cnt
            self.bin_offsets[f + 1] = total_bins
        if total_bins != self.denc:
            raise Error("bin counts do not match d_enc")

        if self.use_emb and self.F > 0:
            self.d_in = self.F * self.demb + self.ccat
        else:
            self.d_in = self.F + self.ccat

        # Parameter offsets mirroring TabMParams.arrays insertion order.
        var ec = 0
        if self.use_emb and self.F > 0:
            ec = 2 + self.F
        self.n_offs = ec + 4 * self.nb + 2
        self.offs = alloc[Int](self.n_offs + 16)
        var cur = 0
        var pos = 0
        if self.use_emb and self.F > 0:
            self.offs[pos] = cur
            pos += 1
            cur += self.F * self.demb  # emb_w0
            self.offs[pos] = cur
            pos += 1
            cur += self.F * self.demb  # emb_b0
            for f in range(self.F):
                self.offs[pos] = cur
                pos += 1
                cur += self.bin_counts[f] * self.demb  # emb_wp_{f}
        for i in range(self.nb):
            var b_in = self.d_in if i == 0 else self.db
            self.offs[pos] = cur
            pos += 1
            cur += self.db * b_in  # w
            self.offs[pos] = cur
            pos += 1
            cur += self.k * b_in  # r
            self.offs[pos] = cur
            pos += 1
            cur += self.k * self.db  # s
            self.offs[pos] = cur
            pos += 1
            cur += self.k * self.db  # b
        self.offs[pos] = cur
        pos += 1
        cur += self.k * self.db * self.dout  # head_w
        self.offs[pos] = cur
        pos += 1
        cur += self.k * self.dout  # head_b
        self.P = cur

    @staticmethod
    def py_init(out self: TabMTrainer, args: PythonObject, kwargs: PythonObject) raises:
        _ = kwargs
        if len(args) != 2:
            raise Error("TabMTrainer(dims, bins) expects 2 arguments")
        self = Self(args[0], args[1])

    # -- offset accessors ------------------------------------------------------

    @always_inline
    def emb_count(self) -> Int:
        if self.use_emb and self.F > 0:
            return 2 + self.F
        return 0

    @always_inline
    def off_emb_w0(self) -> Int:
        return self.offs[0]

    @always_inline
    def off_emb_b0(self) -> Int:
        return self.offs[1]

    @always_inline
    def off_emb_wp(self, f: Int) -> Int:
        return self.offs[2 + f]

    @always_inline
    def off_w(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 4 * i]

    @always_inline
    def off_r(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 4 * i + 1]

    @always_inline
    def off_s(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 4 * i + 2]

    @always_inline
    def off_b(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 4 * i + 3]

    @always_inline
    def off_head_w(self) -> Int:
        return self.offs[self.n_offs - 2]

    @always_inline
    def off_head_b(self) -> Int:
        return self.offs[self.n_offs - 1]

    @always_inline
    def block_in(self, i: Int) -> Int:
        return self.d_in if i == 0 else self.db

    # -- forward -----------------------------------------------------------------

    def embed_forward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        x_cat: Pointer[Float32, MutUntrackedOrigin],
        rows: Pointer[Int, MutUntrackedOrigin],
        b: Int,
    ):
        """Build ws.h (b, d_in) for the given row indices; snapshots inputs."""
        var demb = self.demb
        if self.use_emb and self.F > 0:
            var r = 0
            while r < b:
                var src = rows[r]
                var t = 0
                while t < self.F:
                    ws.xnum[r * self.F + t] = x_num[src * self.F + t]
                    t += 1
                t = 0
                while t < self.denc:
                    ws.enc_snap[r * self.denc + t] = x_enc[src * self.denc + t]
                    t += 1
                r += 1
            var w0 = theta + self.off_emb_w0()
            var b0 = theta + self.off_emb_b0()
            var demb_vec = demb % SIMDW == 0
            r = 0
            while r < b:
                var src = rows[r]
                var f = 0
                while f < self.F:
                    var xv = x_num[src * self.F + f]
                    var hp = ws.h + r * self.d_in + f * demb
                    var wp0 = w0 + f * demb
                    var bp0 = b0 + f * demb
                    if demb_vec:
                        var xv_simd = SIMD[DType.float32, SIMDW](xv)
                        var o2 = 0
                        while o2 < demb:
                            hp.unsafe_store[width=SIMDW](
                                o2, xv_simd * wp0.unsafe_load[width=SIMDW](o2) + bp0.unsafe_load[width=SIMDW](o2)
                            )
                            o2 += SIMDW
                    else:
                        var o3 = 0
                        while o3 < demb:
                            hp[o3] = xv * wp0[o3] + bp0[o3]
                            o3 += 1
                    f += 1
                r += 1
            # piecewise component per feature into ws.pl (pre-relu values)
            for f in range(self.F):
                var cnt = self.bin_counts[f]
                var wp = theta + self.off_emb_wp(f)
                var enc_off = self.bin_offsets[f]
                r = 0
                while r < b:
                    var src = rows[r]
                    var t = 0
                    while t < cnt:
                        ws.tmp[r * cnt + t] = x_enc[src * self.denc + enc_off + t]
                        t += 1
                    r += 1
                gemm_nn(b, demb, cnt, ws.tmp, wp, ws.tmp2)
                r = 0
                while r < b:
                    var o = 0
                    while o < demb:
                        ws.pl[r * self.F * demb + f * demb + o] = ws.tmp2[r * demb + o]
                        o += 1
                    r += 1
            r = 0
            while r < b:
                var o = 0
                while o < self.F * demb:
                    var val = ws.pl[r * self.F * demb + o]
                    if val > 0:
                        ws.h[r * self.d_in + o] += val
                    o += 1
                r += 1
            if self.ccat > 0:
                r = 0
                while r < b:
                    var src = rows[r]
                    var t = 0
                    while t < self.ccat:
                        ws.h[r * self.d_in + self.F * demb + t] = x_cat[src * self.ccat + t]
                        t += 1
                    r += 1
        else:
            var r = 0
            while r < b:
                var src = rows[r]
                var t = 0
                while t < self.F:
                    ws.h[r * self.d_in + t] = x_num[src * self.F + t]
                    t += 1
                t = 0
                while t < self.ccat:
                    ws.h[r * self.d_in + self.F + t] = x_cat[src * self.ccat + t]
                    t += 1
                r += 1

    def backbone_forward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        b: Int,
        mut rng: Rng,
        dropout: Float32,
        train: Bool,
    ):
        """ws.h -> per-block caches; final activation in act_ptr(nb-1)."""
        var keep = 1.0 / (1.0 - dropout)
        var use_dropout = train and dropout > 0.0
        # Byte-wise dropout thresholds: byte >= thresh keeps the element.
        var thresh = UInt64(dropout * 256.0)
        if thresh < 1:
            thresh = 1
        var zero = SIMD[DType.float32, SIMDW](0.0)
        var ones = SIMD[DType.float32, SIMDW](1.0)
        # Bit-buffer for dropout draws: one 64-bit draw covers 64 keep/drop
        # decisions instead of 64 uniform() calls (~8x fewer RNG ops).
        var bits: UInt64 = 0
        var nbits = 0
        var i = 0
        while i < self.nb:
            var b_in = self.block_in(i)
            var w = theta + self.off_w(i)
            var rr = theta + self.off_r(i)
            var ss = theta + self.off_s(i)
            var bb = theta + self.off_b(i)
            var v = ws.vs_ptr(i)
            var q = ws.qs_ptr(i)
            var rmask = ws.rmask_ptr(i)
            var dmask = ws.dmask_ptr(i)
            var act = ws.act_ptr(i)

            # v[row, :] = input[row or broadcast] * r[j, :]
            var row = 0
            while row < b * self.k:
                var j = row % self.k
                var src: Pointer[Float32, MutUntrackedOrigin]
                if i == 0:
                    src = ws.h + (row // self.k) * self.d_in
                else:
                    src = ws.act_ptr(i - 1) + row * self.db
                var dst = v + row * b_in
                var rrow = rr + j * b_in
                var t = 0
                while t + SIMDW <= b_in:
                    dst.unsafe_store[width=SIMDW](
                        t, src.unsafe_load[width=SIMDW](t) * rrow.unsafe_load[width=SIMDW](t)
                    )
                    t += SIMDW
                while t < b_in:
                    dst[t] = src[t] * rrow[t]
                    t += 1
                row += 1

            # q = v @ W^T ; W is (db, b_in)
            gemm_nt(b * self.k, self.db, b_in, v, w, q)

            # u = q*s + b ; act = relu(u) ; masks (vectorized over columns;
            # dropout draws stay in row-major lane order for reproducibility)
            row = 0
            while row < b * self.k:
                var j = row % self.k
                var qp = q + row * self.db
                var sp = ss + j * self.db
                var bp = bb + j * self.db
                var rp = rmask + row * self.db
                var dp = dmask + row * self.db
                var ap = act + row * self.db
                var o = 0
                while o + SIMDW <= self.db:
                    var uv = (
                        qp.unsafe_load[width=SIMDW](o) * sp.unsafe_load[width=SIMDW](o)
                        + bp.unsafe_load[width=SIMDW](o)
                    )
                    var mv = SIMD[DType.float32, SIMDW](0.0)
                    var l = 0
                    while l < SIMDW:
                        mv[l] = 1.0 if uv[l] > 0.0 else 0.0
                        l += 1
                    var dv = ones
                    if use_dropout:
                        if nbits < SIMDW * 8:
                            bits = rng.next_u64()
                            nbits = 64
                        dv = keep_mask_from_bits(bits, keep, thresh)
                        bits <<= SIMDW * 8
                        nbits -= SIMDW * 8
                    rp.unsafe_store[width=SIMDW](o, mv)
                    dp.unsafe_store[width=SIMDW](o, dv)
                    ap.unsafe_store[width=SIMDW](o, max(uv, zero) * dv)
                    o += SIMDW
                while o < self.db:
                    var u = qp[o] * sp[o] + bp[o]
                    var pos = u > 0.0
                    var a = u if pos else 0.0
                    rp[o] = 1.0 if pos else 0.0
                    var dm: Float32 = 1.0
                    if use_dropout:
                        if nbits < 8:
                            bits = rng.next_u64()
                            nbits = 64
                        dm = keep if ((bits >> 56) & 0xFF) >= thresh else 0.0
                        bits <<= 8
                        nbits -= 8
                    dp[o] = dm
                    ap[o] = a * dm
                    o += 1
                row += 1
            i += 1

    def head_forward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        b: Int,
    ):
        # preds[row, o] = sum_t act_last[row,t]*head_w[j,t,o] + head_b[j,o]
        var act_last = ws.act_ptr(self.nb - 1)
        var hw = theta + self.off_head_w()
        var hb = theta + self.off_head_b()
        if self.dout == 1:
            # Fast path: fully contiguous over t -> SIMD dot per member row.
            var row = 0
            while row < b * self.k:
                var j = row % self.k
                var ap = act_last + row * self.db
                var hwp = hw + j * self.db
                var acc = hb[j]
                var t = 0
                while t + SIMDW <= self.db:
                    acc += (
                        ap.unsafe_load[width=SIMDW](t) * hwp.unsafe_load[width=SIMDW](t)
                    ).reduce_add()
                    t += SIMDW
                while t < self.db:
                    acc += ap[t] * hwp[t]
                    t += 1
                ws.preds[row] = acc
                row += 1
            return
        var row = 0
        while row < b * self.k:
            var j = row % self.k
            var o = 0
            while o < self.dout:
                var acc = hb[j * self.dout + o]
                var t = 0
                while t < self.db:
                    acc += act_last[row * self.db + t] * hw[j * self.db * self.dout + t * self.dout + o]
                    t += 1
                ws.preds[row * self.dout + o] = acc
                o += 1
            row += 1

    def forward_chunk(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        x_cat: Pointer[Float32, MutUntrackedOrigin],
        rows: Pointer[Int, MutUntrackedOrigin],
        b: Int,
        mut rng: Rng,
        dropout: Float32,
        train: Bool,
    ):
        self.embed_forward(theta, ws, x_num, x_enc, x_cat, rows, b)
        self.backbone_forward(theta, ws, b, rng, dropout, train)
        self.head_forward(theta, ws, b)

    # -- loss --------------------------------------------------------------------

    def loss_and_dpreds(
        self,
        ws: Workspace,
        y: Pointer[Float32, MutUntrackedOrigin],
        rows: Pointer[Int, MutUntrackedOrigin],
        b: Int,
        task: Int,
        denom_b: Int,
    ) -> Float64:
        """Fill ws.dpreds; returns chunk loss summed over members.

        task: 0 = regression, 1 = binary, 2 = multiclass. denom_b is the row
        count of the full minibatch the chunk belongs to (gradient/loss
        normalization must use the minibatch size even when a chunk is a
        thread-local slice of it).
        """
        var members = b * self.k
        var gmembers = denom_b * self.k
        var total = gmembers * self.dout
        var local_total = members * self.dout
        var loss: Float64 = 0.0
        var row = 0
        while row < members:
            var src = rows[row // self.k]
            if task == 0:
                var o = 0
                while o < self.dout:
                    var diff = ws.preds[row * self.dout + o] - y[src * self.dout + o]
                    loss += Float64(diff * diff)
                    ws.dpreds[row * self.dout + o] = (2.0 / Float32(total)) * diff
                    o += 1
            elif task == 1:
                # BCE with logits, numerically stable:
                # loss = softplus(z) - y*z ; dpred = (sigmoid(z) - y)/members
                var z = ws.preds[row * self.dout]
                var target = y[src * self.dout]
                loss += Float64(softplus(z) - target * z)
                ws.dpreds[row * self.dout] = (sigmoid(z) - target) / Float32(gmembers)
            else:
                # softmax cross-entropy over dout logits
                var base = row * self.dout
                var mx = ws.preds[base]
                var o = 1
                while o < self.dout:
                    if ws.preds[base + o] > mx:
                        mx = ws.preds[base + o]
                    o += 1
                var lse: Float32 = 0.0
                o = 0
                while o < self.dout:
                    var e = fast_exp(ws.preds[base + o] - mx)
                    ws.dpreds[base + o] = e
                    lse += e
                    o += 1
                var inv = 1.0 / lse
                # y holds integer class labels (B, 1); one label per row.
                var cls = Int(y[src])
                loss -= Float64(ws.preds[base + cls] - mx - fast_log(lse))
                o = 0
                while o < self.dout:
                    var sm = ws.dpreds[base + o] * inv
                    var onehot: Float32 = 0.0
                    if o == cls:
                        onehot = 1.0
                    ws.dpreds[base + o] = (sm - onehot) / Float32(gmembers)
                    o += 1
            row += 1
        if task == 0:
            # Returned loss stays a per-chunk mean; only dpreds scaling uses
            # the full-minibatch denominator.
            return loss / Float64(local_total)
        return loss / Float64(members)

    # -- backward ------------------------------------------------------------------

    def backward_chunk(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        g: Pointer[Float32, MutUntrackedOrigin],
        b: Int,
    ):
        """Accumulate parameter gradients from ws.dpreds into g."""
        var k = self.k
        var bk = b * k

        # ---- head ----
        var act_last = ws.act_ptr(self.nb - 1)
        var hw = theta + self.off_head_w()
        var dhw = g + self.off_head_w()
        var dhb = g + self.off_head_b()
        if self.dout == 1:
            # Fast path: da[row,t] = dpred[row]*hw[j,t]; dhw[j,t] += act*dpred.
            var row0 = 0
            while row0 < bk:
                var j0 = row0 % k
                var dp0 = ws.dpreds[row0]
                var hwp0 = hw + j0 * self.db
                var dap0 = ws.da + row0 * self.db
                var t0v = 0
                while t0v + SIMDW <= self.db:
                    dap0.unsafe_store[width=SIMDW](
                        t0v, SIMD[DType.float32, SIMDW](dp0) * hwp0.unsafe_load[width=SIMDW](t0v)
                    )
                    t0v += SIMDW
                while t0v < self.db:
                    dap0[t0v] = dp0 * hwp0[t0v]
                    t0v += 1
                row0 += 1
            var j1 = 0
            while j1 < k:
                var dhwj = dhw + j1 * self.db
                var bb1 = 0
                var dhb_acc: Float32 = 0.0
                while bb1 < b:
                    var rowj1 = bb1 * k + j1
                    var dp1 = ws.dpreds[rowj1]
                    dhb_acc += dp1
                    var ap1 = act_last + rowj1 * self.db
                    var t1 = 0
                    while t1 + SIMDW <= self.db:
                        var cur = dhwj.unsafe_load[width=SIMDW](t1)
                        cur += SIMD[DType.float32, SIMDW](dp1) * ap1.unsafe_load[width=SIMDW](t1)
                        dhwj.unsafe_store[width=SIMDW](t1, cur)
                        t1 += SIMDW
                    while t1 < self.db:
                        dhwj[t1] += ap1[t1] * dp1
                        t1 += 1
                    bb1 += 1
                dhb[j1] += dhb_acc
                j1 += 1
        else:
            # da[row, t] = sum_o dpred[row,o]*head_w[j,t,o]
            var row = 0
            while row < bk:
                var j = row % k
                var t = 0
                while t < self.db:
                    var acc = SIMD[DType.float32, SIMDW](0.0)
                    var o = 0
                    while o + SIMDW <= self.dout:
                        acc += ws.dpreds.unsafe_load[width=SIMDW](row * self.dout + o) * hw.unsafe_load[width=SIMDW](
                            j * self.db * self.dout + t * self.dout + o
                        )
                        o += SIMDW
                    var da_val = acc.reduce_add()
                    while o < self.dout:
                        da_val += ws.dpreds[row * self.dout + o] * hw[j * self.db * self.dout + t * self.dout + o]
                        o += 1
                    ws.da[row * self.db + t] = da_val
                    t += 1
                row += 1
            # dhw[j,t,o] += sum_bb act_last[bb,j,t]*dpred[bb,j,o]; dhb[j,o] += sum_bb dpred
            var j = 0
            while j < k:
                var bb = 0
                while bb < b:
                    var rowj = bb * k + j
                    var t = 0
                    while t < self.db:
                        var av = act_last[rowj * self.db + t]
                        var o = 0
                        while o + SIMDW <= self.dout:
                            var cur = dhw.unsafe_load[width=SIMDW](j * self.db * self.dout + t * self.dout + o)
                            cur += SIMD[DType.float32, SIMDW](av) * ws.dpreds.unsafe_load[width=SIMDW](rowj * self.dout + o)
                            dhw.unsafe_store[width=SIMDW](j * self.db * self.dout + t * self.dout + o, cur)
                            o += SIMDW
                        while o < self.dout:
                            dhw[j * self.db * self.dout + t * self.dout + o] += av * ws.dpreds[rowj * self.dout + o]
                            o += 1
                        t += 1
                    var o2 = 0
                    while o2 + SIMDW <= self.dout:
                        var cur = dhb.unsafe_load[width=SIMDW](j * self.dout + o2)
                        cur += ws.dpreds.unsafe_load[width=SIMDW](rowj * self.dout + o2)
                        dhb.unsafe_store[width=SIMDW](j * self.dout + o2, cur)
                        o2 += SIMDW
                    while o2 < self.dout:
                        dhb[j * self.dout + o2] += ws.dpreds[rowj * self.dout + o2]
                        o2 += 1
                    bb += 1
                j += 1

        # ---- blocks in reverse ----
        var i = self.nb - 1
        while i >= 0:
            var b_in = self.block_in(i)
            var w = theta + self.off_w(i)
            var rr = theta + self.off_r(i)
            var ss = theta + self.off_s(i)
            var v = ws.vs_ptr(i)
            var q = ws.qs_ptr(i)
            var rmask = ws.rmask_ptr(i)
            var dmask = ws.dmask_ptr(i)
            var da = ws.da
            var dq = ws.dq
            var dv = ws.dv

            # dropout then relu on incoming da
            var idx = 0
            var bkdb = bk * self.db
            while idx + SIMDW <= bkdb:
                dq.unsafe_store[width=SIMDW](
                    idx,
                    da.unsafe_load[width=SIMDW](idx)
                    * dmask.unsafe_load[width=SIMDW](idx)
                    * rmask.unsafe_load[width=SIMDW](idx),
                )
                idx += SIMDW
            while idx < bkdb:
                dq[idx] = da[idx] * dmask[idx] * rmask[idx]
                idx += 1

            # biases: db_grad[j,o] += dq[bb,j,o]
            var db_grad = g + self.off_b(i)
            var ds_grad = g + self.off_s(i)
            row = 0
            while row < bk:
                var jj = row % k
                var o = 0
                while o + SIMDW <= self.db:
                    var cur = db_grad.unsafe_load[width=SIMDW](jj * self.db + o)
                    cur += dq.unsafe_load[width=SIMDW](row * self.db + o)
                    db_grad.unsafe_store[width=SIMDW](jj * self.db + o, cur)
                    o += SIMDW
                while o < self.db:
                    db_grad[jj * self.db + o] += dq[row * self.db + o]
                    o += 1
                row += 1
            # scales: ds_grad[jj,o] += sum_bb dq*q
            row = 0
            while row < bk:
                var jj = row % k
                var o = 0
                while o + SIMDW <= self.db:
                    var cur = ds_grad.unsafe_load[width=SIMDW](jj * self.db + o)
                    cur += dq.unsafe_load[width=SIMDW](row * self.db + o) * q.unsafe_load[width=SIMDW](row * self.db + o)
                    ds_grad.unsafe_store[width=SIMDW](jj * self.db + o, cur)
                    o += SIMDW
                while o < self.db:
                    ds_grad[jj * self.db + o] += dq[row * self.db + o] * q[row * self.db + o]
                    o += 1
                row += 1

            # dq *= s[jj]
            row = 0
            while row < bk:
                var jj = row % k
                var o = 0
                while o + SIMDW <= self.db:
                    var cur = dq.unsafe_load[width=SIMDW](row * self.db + o)
                    cur *= ss.unsafe_load[width=SIMDW](jj * self.db + o)
                    dq.unsafe_store[width=SIMDW](row * self.db + o, cur)
                    o += SIMDW
                while o < self.db:
                    dq[row * self.db + o] *= ss[jj * self.db + o]
                    o += 1
                row += 1

            # dw += dq^T @ v ; dv = dq @ W
            gemm_tn_acc(self.db, b_in, bk, dq, v, g + self.off_w(i))
            gemm_nn(bk, b_in, self.db, dq, w, dv)

            # dr[jj,t] += sum_bb dv[bb,jj,t]*xin[bb,jj,t] ; da_new = dv*r
            var dr = g + self.off_r(i)
            var da_new = ws.da  # reuse; bmax >= b_in so sizes fit
            row = 0
            while row < bk:
                var jj = row % k
                var src: Pointer[Float32, MutUntrackedOrigin]
                if i == 0:
                    src = ws.h + (row // k) * self.d_in
                else:
                    src = ws.act_ptr(i - 1) + row * self.db
                var dvp = dv + row * b_in
                var drp = dr + jj * b_in
                var anp = da_new + row * b_in
                var rp2 = rr + jj * b_in
                var t = 0
                while t + SIMDW <= b_in:
                    var dvv = dvp.unsafe_load[width=SIMDW](t)
                    var cur = drp.unsafe_load[width=SIMDW](t)
                    cur += dvv * src.unsafe_load[width=SIMDW](t)
                    drp.unsafe_store[width=SIMDW](t, cur)
                    anp.unsafe_store[width=SIMDW](t, dvv * rp2.unsafe_load[width=SIMDW](t))
                    t += SIMDW
                while t < b_in:
                    var dvv_s = dvp[t]
                    drp[t] += dvv_s * src[t]
                    anp[t] = dvv_s * rp2[t]
                    t += 1
                row += 1

            if i == 0:
                # accumulate member axis into ws.dh (b, d_in)
                vec_zero(ws.dh, b * self.d_in)
                var bb2 = 0
                while bb2 < b:
                    var jj2 = 0
                    while jj2 < k:
                        var t2 = 0
                        while t2 + SIMDW <= self.d_in:
                            var cur = ws.dh.unsafe_load[width=SIMDW](bb2 * self.d_in + t2)
                            cur += da_new.unsafe_load[width=SIMDW]((bb2 * k + jj2) * self.d_in + t2)
                            ws.dh.unsafe_store[width=SIMDW](bb2 * self.d_in + t2, cur)
                            t2 += SIMDW
                        while t2 < self.d_in:
                            ws.dh[bb2 * self.d_in + t2] += da_new[(bb2 * k + jj2) * self.d_in + t2]
                            t2 += 1
                        jj2 += 1
                    bb2 += 1
            i -= 1

        # ---- embeddings ----
        if self.use_emb and self.F > 0:
            self.embed_backward(theta, ws, g, b)

    def embed_backward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        g: Pointer[Float32, MutUntrackedOrigin],
        b: Int,
    ):
        """Gradients for embeddings from ws.dh (b, d_in); uses snapshots."""
        var demb = self.demb
        var dw0 = g + self.off_emb_w0()
        var db0 = g + self.off_emb_b0()
        var demb_vec = demb % SIMDW == 0
        var r = 0
        while r < b:
            var f = 0
            while f < self.F:
                var xn = ws.xnum[r * self.F + f]
                var dhp = ws.dh + r * self.d_in + f * demb
                if demb_vec:
                    var xn_simd = SIMD[DType.float32, SIMDW](xn)
                    var dwp = dw0 + f * demb
                    var dbp = db0 + f * demb
                    var o2 = 0
                    while o2 < demb:
                        var dg = dhp.unsafe_load[width=SIMDW](o2)
                        dwp.unsafe_store[width=SIMDW](o2, dwp.unsafe_load[width=SIMDW](o2) + xn_simd * dg)
                        dbp.unsafe_store[width=SIMDW](o2, dbp.unsafe_load[width=SIMDW](o2) + dg)
                        o2 += SIMDW
                else:
                    var o3 = 0
                    while o3 < demb:
                        var dg = dhp[o3]
                        dw0[f * demb + o3] += xn * dg
                        db0[f * demb + o3] += dg
                        o3 += 1
                f += 1
            r += 1
        # dpl = dh * (pl >= 0), then dwp_f += enc^T @ dpl_f
        for f in range(self.F):
            var cnt = self.bin_counts[f]
            var enc_off = self.bin_offsets[f]
            var r2 = 0
            while r2 < b:
                var o = 0
                while o < demb:
                    var plv = ws.pl[r2 * self.F * demb + f * demb + o]
                    var dg = ws.dh[r2 * self.d_in + f * demb + o]
                    ws.tmp2[r2 * demb + o] = dg if plv >= 0.0 else 0.0
                    o += 1
                r2 += 1
            # gather enc columns into tmp (b, cnt)
            var r3 = 0
            while r3 < b:
                var t = 0
                while t < cnt:
                    ws.tmp[r3 * cnt + t] = ws.enc_snap[r3 * self.denc + enc_off + t]
                    t += 1
                r3 += 1
            gemm_tn_acc(cnt, demb, b, ws.tmp, ws.tmp2, g + self.off_emb_wp(f))

    # -- full-batch helpers ----------------------------------------------------------

    def full_loss_grad(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        grad: Pointer[Float32, MutUntrackedOrigin],
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        x_cat: Pointer[Float32, MutUntrackedOrigin],
        y: Pointer[Float32, MutUntrackedOrigin],
        N: Int,
        task: Int,
        add_l2: Bool,
        alpha: Float32,
        nthreads: Int,
    ) -> Float64:
        return k_full_loss_grad(
            self, theta, grad, x_num, x_enc, x_cat, y, N, task, add_l2, alpha, nthreads
        )

    # -- bound-type entry points -------------------------------------------------------

    def adam_epoch_impl(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        m: Pointer[Float32, MutUntrackedOrigin],
        v: Pointer[Float32, MutUntrackedOrigin],
        t0: Int,
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        x_cat: Pointer[Float32, MutUntrackedOrigin],
        y: Pointer[Float32, MutUntrackedOrigin],
        N: Int,
        lr: Float32,
        bs: Int,
        dropout: Float32,
        alpha: Float32,
        seed: UInt64,
        task: Int,
        nthreads: Int,
    ) -> Tuple[Float64, Int]:
        return k_adam_epoch(
            self, theta, m, v, t0, x_num, x_enc, x_cat, y, N, lr, bs, dropout, alpha, seed, task, nthreads
        )

    def lbfgs_minimize_impl(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        x_cat: Pointer[Float32, MutUntrackedOrigin],
        y: Pointer[Float32, MutUntrackedOrigin],
        N: Int,
        max_iter: Int,
        tol: Float64,
        maxcor: Int,
        alpha: Float32,
        losses_out: Pointer[Float64, MutUntrackedOrigin],
        task: Int,
        nthreads: Int,
    ) -> Int:
        var P = self.P
        # Model state and evaluations stay float32; all L-BFGS bookkeeping
        # (curvature history, direction, candidate steps) runs in float64.
        # float32 history pairs lose the small s/y differences that carry
        # curvature information, which stalls convergence.
        var g = alloc[Float32](P)
        var gnew = alloc[Float32](P)
        var cand = alloc[Float32](P)
        var th64 = alloc[Float64](P)
        var g64 = alloc[Float64](P)
        var gnew64 = alloc[Float64](P)
        var cand64 = alloc[Float64](P)
        var dir = alloc[Float64](P)
        var S = alloc[Float64](maxcor * P)
        var Y = alloc[Float64](maxcor * P)
        var rho = alloc[Float64](maxcor)
        var alphas = alloc[Float64](maxcor)

        var loss = self.full_loss_grad(theta, g, x_num, x_enc, x_cat, y, N, task, True, alpha, nthreads)
        losses_out[0] = loss
        var n_losses = 1
        var i = 0
        while i < P:
            th64[i] = Float64(theta[i])
            g64[i] = Float64(g[i])
            i += 1
        if max_abs_f32(g, P) <= tol:
            g.free(); gnew.free(); cand.free()
            th64.free(); g64.free(); gnew64.free(); cand64.free(); dir.free()
            S.free(); Y.free(); rho.free(); alphas.free()
            return 0

        var nhist = 0
        var hptr = 0  # next slot to write (circular)
        var last_ys: Float64 = 1.0
        var last_yy: Float64 = 1.0
        var first_iter = True
        var consec_fail = 0
        var it = 0
        while it < max_iter:
            # two-loop recursion -> dir = -H g
            i = 0
            while i < P:
                dir[i] = -g64[i]
                i += 1
            var nj = 0
            while nj < nhist:
                var slot = (hptr - 1 - nj + maxcor) % maxcor
                var sp = S + slot * P
                var yp = Y + slot * P
                var aj = rho[slot] * dot_f64(sp, dir, P)
                alphas[nj] = aj
                i = 0
                while i < P:
                    dir[i] -= aj * yp[i]
                    i += 1
                nj += 1
            if nhist > 0:
                var gamma = last_ys / last_yy
                i = 0
                while i < P:
                    dir[i] *= gamma
                    i += 1
            var pj = nhist - 1
            while pj >= 0:
                # Same slot indexing as the first loop (pj counts back from
                # the most recent entry), visited oldest-first as required
                # by the two-loop recursion.
                var slot = (hptr - 1 - pj + maxcor) % maxcor
                var sp = S + slot * P
                var yp = Y + slot * P
                var bj = rho[slot] * dot_f64(yp, dir, P)
                i = 0
                while i < P:
                    dir[i] += (alphas[pj] - bj) * sp[i]
                    i += 1
                pj -= 1

            var dg = dot_f64(g64, dir, P)
            if dg >= 0.0:
                # direction not descending; reset and fall back to steepest descent
                nhist = 0
                i = 0
                while i < P:
                    dir[i] = -g64[i]
                    i += 1
                dg = -dot_f64(g64, g64, P)
            var step: Float64 = 1.0
            if first_iter:
                var gn: Float64 = 0.0
                i = 0
                while i < P:
                    gn += abs(g64[i])
                    i += 1
                if gn > 0.0:
                    step = min(1.0, 1.0 / gn)
            var accepted = False
            var ls = 0
            while ls < 20:
                i = 0
                while i < P:
                    cand64[i] = th64[i] + step * dir[i]
                    cand[i] = Float32(cand64[i])
                    i += 1
                var loss_new = self.full_loss_grad(cand, gnew, x_num, x_enc, x_cat, y, N, task, True, alpha, nthreads)
                if loss_new <= loss + 1e-4 * step * dg:
                    # success: shift state
                    i = 0
                    while i < P:
                        gnew64[i] = Float64(gnew[i])
                        S[hptr * P + i] = cand64[i] - th64[i]
                        Y[hptr * P + i] = gnew64[i] - g64[i]
                        i += 1
                    var ys = dot_f64(S + hptr * P, Y + hptr * P, P)
                    var yy = dot_f64(Y + hptr * P, Y + hptr * P, P)
                    # Keep any pair with positive curvature; an absolute
                    # floor here rejects the small-but-valid pairs produced
                    # by heavily backtracked steps and reduces the method
                    # to steepest descent.
                    if ys > 0.0:
                        rho[hptr] = 1.0 / ys
                        last_ys = ys
                        last_yy = yy
                        hptr = (hptr + 1) % maxcor
                        if nhist < maxcor:
                            nhist += 1
                    else:
                        nhist = 0
                    i = 0
                    while i < P:
                        th64[i] = cand64[i]
                        g64[i] = gnew64[i]
                        theta[i] = cand[i]
                        g[i] = gnew[i]
                        i += 1
                    loss = loss_new
                    losses_out[n_losses] = loss
                    n_losses += 1
                    accepted = True
                    break
                step *= 0.5
                ls += 1
            if not accepted:
                consec_fail += 1
                if consec_fail >= 3:
                    break
                it += 1
                continue
            consec_fail = 0
            first_iter = False
            if max_abs_f32(g, P) <= tol:
                break
            it += 1

        var result = n_losses - 1
        g.free(); gnew.free(); cand.free()
        th64.free(); g64.free(); gnew64.free(); cand64.free(); dir.free()
        S.free(); Y.free(); rho.free(); alphas.free()
        return result

    # -- Python-visible methods -----------------------------------------------------

    @staticmethod
    def adam_epoch(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var theta = ptr_f32(parts[0])
        var m = ptr_f32(parts[1])
        var v = ptr_f32(parts[2])
        var t0 = Int(py=parts[3])
        var x_num = ptr_f32(parts[4])
        var x_enc = ptr_f32(parts[5])
        var x_cat = ptr_f32(parts[6])
        var y = ptr_f32(parts[7])
        var N = iface_dim(parts[7], 0)
        var lr = Float32(Float64(py=parts[8]))
        var bs = Int(py=parts[9])
        var dropout = Float32(Float64(py=parts[10]))
        var alpha = Float32(Float64(py=parts[11]))
        var seed = UInt64(Int(py=parts[12]))
        var task = Int(py=parts[13])
        var res = self.adam_epoch_impl(theta, m, v, t0, x_num, x_enc, x_cat, y, N, lr, bs, dropout, alpha, seed, task, resolve_threads())
        return Python.tuple(res[0], res[1])

    @staticmethod
    def lbfgs_minimize(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var y = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)
        var max_iter = Int(py=parts[5])
        var tol = Float64(py=parts[6])
        var maxcor = Int(py=parts[7])
        var alpha = Float32(Float64(py=parts[8]))
        var losses_out = ptr_f64(parts[9])
        var task = Int(py=parts[10])
        var nit = self.lbfgs_minimize_impl(theta, x_num, x_enc, x_cat, y, N, max_iter, tol, maxcor, alpha, losses_out, task, resolve_threads())
        return Int(nit)

    @staticmethod
    def loss_grad(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var np = np_module()
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var y = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)
        var task = Int(py=parts[5])
        var alpha = Float32(Float64(py=parts[6]))
        var grad_arr = np_empty_1d(np, self.P, "float32")
        var grad = ptr_f32(grad_arr)
        var loss = self.full_loss_grad(theta, grad, x_num, x_enc, x_cat, y, N, task, True, alpha, resolve_threads())
        return Python.tuple(Float64(loss), grad_arr)

    @staticmethod
    def forward_avg(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var out = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)
        var rng = Rng(1)
        var chunk = max(1, 8192 // self.k)
        var inv_k = 1.0 / Float32(self.k)
        var idx = alloc[Int](N)
        var i = 0
        while i < N:
            idx[i] = i
            i += 1
        var ws = Workspace(chunk, self.k, self.nb, self.d_in, self.db, self.dout, self.F, self.demb, self.denc)
        var start = 0
        while start < N:
            var b = min(chunk, N - start)
            self.forward_chunk(theta, ws, x_num, x_enc, x_cat, idx + start, b, rng, 0.0, False)
            var r = 0
            while r < b:
                var j = 0
                while j < self.k:
                    var o = 0
                    while o < self.dout:
                        out[(start + r) * self.dout + o] += ws.preds[(r * self.k + j) * self.dout + o] * inv_k
                        o += 1
                    j += 1
                r += 1
            start += b
        ws.unsafe_free()
        idx.unsafe_free()
        return Python.none()

    @staticmethod
    def param_count(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var self = self_ptr[]
        return Int(self.P)


# =============================================================================
# Data-parallel workers (one private workspace + gradient buffer per thread)
# =============================================================================


struct ChunkWorker(Movable):
    var tr: TabMTrainer
    var ws: Workspace
    var g: Pointer[Float32, MutUntrackedOrigin]
    var theta: Pointer[Float32, MutUntrackedOrigin]
    var x_num: Pointer[Float32, MutUntrackedOrigin]
    var x_enc: Pointer[Float32, MutUntrackedOrigin]
    var x_cat: Pointer[Float32, MutUntrackedOrigin]
    var y: Pointer[Float32, MutUntrackedOrigin]
    var rows: Pointer[Int, MutUntrackedOrigin]
    var nrows: Int
    var denom_b: Int
    var task: Int
    var dropout: Float32
    var train: Bool
    var seed: UInt64
    var loss_out: Float64

    def __init__(
        out self,
        tr: TabMTrainer,
        max_rows: Int,
        theta: Pointer[Float32, MutUntrackedOrigin],
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        x_cat: Pointer[Float32, MutUntrackedOrigin],
        y: Pointer[Float32, MutUntrackedOrigin],
        task: Int,
        dropout: Float32,
        train: Bool,
        rows: Pointer[Int, MutUntrackedOrigin],
    ):
        self.tr = tr
        self.ws = Workspace(max_rows, tr.k, tr.nb, tr.d_in, tr.db, tr.dout, tr.F, tr.demb, tr.denc)
        self.g = alloc[Float32](tr.P + 16)
        self.theta = theta
        self.x_num = x_num
        self.x_enc = x_enc
        self.x_cat = x_cat
        self.y = y
        self.rows = rows
        self.nrows = 0
        self.denom_b = 1
        self.task = task
        self.dropout = dropout
        self.train = train
        self.seed = 1
        self.loss_out = 0.0

    def release(mut self):
        self.ws.unsafe_free()
        self.g.unsafe_free()

    async def run(mut self):
        """forward -> loss/dpreds -> backward for this worker's row slice."""
        if self.nrows <= 0:
            self.loss_out = 0.0
            return
        vec_zero(self.g, self.tr.P)
        # Per-worker RNG stream: deterministic for a fixed seed + thread count.
        var rng = Rng(self.seed)
        self.tr.forward_chunk(
            self.theta, self.ws, self.x_num, self.x_enc, self.x_cat,
            self.rows, self.nrows, rng, self.dropout, self.train,
        )
        self.loss_out = self.tr.loss_and_dpreds(
            self.ws, self.y, self.rows, self.nrows, self.task, self.denom_b
        )
        self.tr.backward_chunk(self.theta, self.ws, self.g, self.nrows)


@always_inline
def adam_update(
    theta: Pointer[Float32, MutUntrackedOrigin],
    m: Pointer[Float32, MutUntrackedOrigin],
    v: Pointer[Float32, MutUntrackedOrigin],
    g: Pointer[Float32, MutUntrackedOrigin],
    P: Int,
    alpha: Float32,
    lr: Float32,
    bc1: Float64,
    bc2: Float64,
):
    """Vectorized bias-corrected Adam update (mirrors _optim.adam_step)."""
    var af = SIMD[DType.float32, SIMDW](alpha)
    var lrv = SIMD[DType.float32, SIMDW](lr)
    var eps = SIMD[DType.float32, SIMDW](1e-8)
    var b1inv = SIMD[DType.float32, SIMDW](Float32(1.0 / bc1))
    var b2inv = SIMD[DType.float32, SIMDW](Float32(1.0 / bc2))
    var c90 = SIMD[DType.float32, SIMDW](0.9)
    var c10 = SIMD[DType.float32, SIMDW](0.1)
    var c999 = SIMD[DType.float32, SIMDW](0.999)
    var c001 = SIMD[DType.float32, SIMDW](0.001)
    var p = 0
    while p + SIMDW <= P:
        var gv = g.unsafe_load[width=SIMDW](p) + af * theta.unsafe_load[width=SIMDW](p)
        var mv = m.unsafe_load[width=SIMDW](p) * c90 + gv * c10
        var vv = v.unsafe_load[width=SIMDW](p) * c999 + gv * gv * c001
        m.unsafe_store[width=SIMDW](p, mv)
        v.unsafe_store[width=SIMDW](p, vv)
        var upd = lrv * (mv * b1inv) / (sqrt(vv * b2inv) + eps)
        theta.unsafe_store[width=SIMDW](p, theta.unsafe_load[width=SIMDW](p) - upd)
        p += SIMDW
    while p < P:
        var gi = g[p] + alpha * theta[p]
        m[p] = 0.9 * m[p] + 0.1 * gi
        v[p] = 0.999 * v[p] + 0.001 * gi * gi
        var mh = m[p] / Float32(bc1)
        var vh = v[p] / Float32(bc2)
        theta[p] -= lr * mh / (sqrt(vh) + 1e-8)
        p += 1


def run_round(
    mut workers: List[ChunkWorker],
    nthreads: Int,
    tr: TabMTrainer,
    theta: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int, MutUntrackedOrigin],
    offset: Int,
    nrows_total: Int,
    dropout: Float32,
    train: Bool,
    task: Int,
    denom_b: Int,
    seed: UInt64,
    round_id: UInt64,
    g_out: Pointer[Float32, MutUntrackedOrigin],
) -> Float64:
    """Process rows [offset, offset+nrows_total) across worker threads.

    Returns the loss SUM over those rows; accumulates every worker's
    gradients into g_out. Deterministic for fixed nthreads and seeds.
    """
    var T = min(nthreads, nrows_total)
    if T < 1:
        T = 1
    var rows_per = (nrows_total + T - 1) // T
    var assigned = 0
    var tg = TaskGroup()
    for t in range(T):
        var lo = assigned
        var hi = min(lo + rows_per, nrows_total)
        assigned = hi
        workers[t].rows = idx + offset + lo
        workers[t].nrows = hi - lo
        workers[t].denom_b = denom_b
        workers[t].seed = mix_u64(seed ^ mix_u64(round_id * 0x9E3779B97F4A7C15 + UInt64(t)))
        tg.create_task(workers[t].run())
    tg.wait()
    assigned = 0
    var wsum: Float64 = 0.0
    for t2 in range(T):
        var lo = assigned
        var hi = min(lo + rows_per, nrows_total)
        assigned = hi
        wsum += workers[t2].loss_out * Float64(hi - lo)
        accumulate_into(g_out, workers[t2].g, tr.P)
    return wsum


def k_full_loss_grad(
    tr: TabMTrainer,
    theta: Pointer[Float32, MutUntrackedOrigin],
    grad: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    N: Int,
    task: Int,
    add_l2: Bool,
    alpha: Float32,
    nthreads: Int,
) -> Float64:
    """Parallel chunked forward/backward over all N rows; mean loss."""
    vec_zero(grad, tr.P)
    var chunk = max(1, 8192 // tr.k)
    var rows_cap = (chunk + nthreads - 1) // nthreads
    var dummy_rows = alloc[Int](1)
    var workers = List[ChunkWorker]()
    for t in range(nthreads):
        workers.append(ChunkWorker(tr, rows_cap, theta, x_num, x_enc, x_cat, y, task, 0.0, False, dummy_rows))
    var idx = alloc[Int](N)
    var i = 0
    while i < N:
        idx[i] = i
        i += 1
    var weighted: Float64 = 0.0
    var start = 0
    var rnd = 0
    while start < N:
        var b = min(chunk, N - start)
        weighted += run_round(
            workers, nthreads, tr, theta, x_num, x_enc, x_cat, y,
            idx, start, b, 0.0, False, task, b, 1, mix_u64(UInt64(rnd)), grad,
        )
        start += b
        rnd += 1
    for t2 in range(nthreads):
        workers[t2].release()
    dummy_rows.unsafe_free()
    idx.unsafe_free()
    var loss = weighted / Float64(N)
    if add_l2:
        loss += 0.5 * Float64(alpha) * sum_sq_f32(theta, tr.P)
        saxpy(grad, theta, Float64(alpha), tr.P)
    return loss


def k_adam_epoch(
    tr: TabMTrainer,
    theta: Pointer[Float32, MutUntrackedOrigin],
    m: Pointer[Float32, MutUntrackedOrigin],
    v: Pointer[Float32, MutUntrackedOrigin],
    t0: Int,
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    N: Int,
    lr: Float32,
    bs: Int,
    dropout: Float32,
    alpha: Float32,
    seed: UInt64,
    task: Int,
    nthreads: Int,
) -> Tuple[Float64, Int]:
    var rng = Rng(seed)
    var idx = alloc[Int](N)
    var i = 0
    while i < N:
        idx[i] = i
        i += 1
    # Fisher-Yates shuffle (same stream as the serial kernels)
    var ii = N - 1
    while ii > 0:
        var jj = Int(rng.next_u64() % UInt64(ii + 1))
        var tmp = idx[ii]
        idx[ii] = idx[jj]
        idx[jj] = tmp
        ii -= 1

    var chunk = max(1, 8192 // tr.k)
    var rows_cap = (chunk + nthreads - 1) // nthreads
    var workers = List[ChunkWorker]()
    for t in range(nthreads):
        workers.append(ChunkWorker(tr, rows_cap, theta, x_num, x_enc, x_cat, y, task, dropout, True, idx))
    var g = alloc[Float32](tr.P)
    var l2_term = 0.5 * Float64(alpha) * sum_sq_f32(theta, tr.P)
    var weighted: Float64 = 0.0
    var t = t0
    var start = 0
    var mb = 0
    while start < N:
        var be = min(start + bs, N)
        var nb_rows = be - start
        vec_zero(g, tr.P)
        var batch_loss: Float64 = 0.0
        var c0 = start
        while c0 < be:
            var rb = min(chunk, be - c0)
            batch_loss += run_round(
                workers, nthreads, tr, theta, x_num, x_enc, x_cat, y,
                idx, c0, rb, dropout, True, task, rb, seed,
                mix_u64(UInt64(mb) * 2654435761 + UInt64(c0)), g,
            )
            c0 += rb
        mb += 1
        batch_loss = batch_loss / Float64(nb_rows) + l2_term
        # Adam update (bias-corrected, mirrors _optim.adam_step)
        t += 1
        adam_update(theta, m, v, g, tr.P, alpha, lr, 1.0 - pow(0.9, Float64(t)), 1.0 - pow(0.999, Float64(t)))
        weighted += batch_loss * Float64(nb_rows)
        start = be
    for t3 in range(nthreads):
        workers[t3].release()
    idx.unsafe_free()
    g.unsafe_free()
    return (weighted / Float64(N), t)


@export
def PyInit__native_tabm() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native_tabm")
        _ = (
            m.add_type[TabMTrainer]("TabMTrainer")
            .def_py_init[TabMTrainer.py_init]()
            .def_method[TabMTrainer.adam_epoch]("adam_epoch")
            .def_method[TabMTrainer.lbfgs_minimize]("lbfgs_minimize")
            .def_method[TabMTrainer.loss_grad]("loss_grad")
            .def_method[TabMTrainer.forward_avg]("forward_avg")
            .def_method[TabMTrainer.param_count]("param_count")
        )
        var mod = m.finalize()
        return mod
    except e:
        abort(String("failed to create module _native_tabm: ", e))
