"""Mojo training kernels for the plain MLP estimators.

Compiles to the ``shinrin._native_mlp`` Python extension module (build
with ``just build-mlp-mojo``). Exposes an ``MLPTrainer`` bound type whose
methods run complete training steps without returning to Python:

- ``adam_epoch(parts)``: one shuffled minibatch Adam epoch (dropout included)
- ``lbfgs_minimize(parts)``: full-batch L-BFGS with backtracking line search
- ``loss_grad(parts)``: full-batch loss + gradient (parity testing)
- ``forward(parts)``: predictions into a preallocated array

All arrays cross the boundary as NumPy buffers accessed through raw
pointers (float32 data, int64 dims). The parameter layout matches
``shinrin._mlp._layers.MLPParams.flatten`` exactly:

    emb_w0, emb_b0, emb_wp_0..F-1,     (when embeddings enabled)
    l{i}_w, l{i}_b,                    for i in 0..n_layers-1

Loss conventions mirror ``shinrin._mlp._model.MLPCore.loss_and_dpreds``:
regression loss is ``0.5 * sum((pred - y)^2) / B``; binary/multiclass
losses are means over rows. L2 adds ``alpha*theta`` to gradients and
``0.5*alpha*||theta||^2`` to the loss.

Activation codes: 0 identity, 1 logistic, 2 tanh, 3 relu.
Task codes: 0 regression, 1 binary, 2 multiclass.
"""

from std.os import abort
from std.math import abs, ceil, exp, floor, log, pow, sqrt
from std.memory import alloc
from std.io import Writer
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime.asyncrt import TaskGroup
from std.sys.info import num_performance_cores

comptime SIMDW = 8


# =============================================================================
# numpy interop helpers (same pattern as _tabm_kernels.mojo)
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
def act_apply(code: Int, z: Float32) -> Float32:
    """Hidden-layer activation (codes: 0 id, 1 logistic, 2 tanh, 3 relu)."""
    if code == 3:
        return z if z > 0.0 else 0.0
    if code == 1:
        return sigmoid(z)
    if code == 2:
        if z > 20.0:
            return 1.0
        if z < -20.0:
            return -1.0
        var e = fast_exp(2.0 * z)
        return (e - 1.0) / (e + 1.0)
    return z


@always_inline
def act_deriv(code: Int, z: Float32) -> Float32:
    """Derivative of the hidden activation at its pre-activation value."""
    if code == 3:
        return 1.0 if z > 0.0 else 0.0
    if code == 1:
        var s = sigmoid(z)
        return s * (1.0 - s)
    if code == 2:
        if z > 20.0 or z < -20.0:
            return 0.0
        var e = fast_exp(2.0 * z)
        var t = (e - 1.0) / (e + 1.0)
        return 1.0 - t * t
    return 1.0


@always_inline
def mix_u64(x: UInt64) -> UInt64:
    """splitmix64 finalizer: derives independent per-worker RNG streams."""
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def resolve_threads() raises -> Int:
    """Worker count; override with SHINRIN_MLP_THREADS."""
    var t = num_performance_cores()
    if t < 1:
        t = 1
    from std.os import getenv

    var env = getenv("SHINRIN_MLP_THREADS")
    if env:
        var parsed = Int(String(env))
        if parsed >= 1:
            t = parsed
    return t


@always_inline
def keep_mask_from_bits(v: UInt64, keep: Float32, thresh: UInt64) -> SIMD[DType.float32, SIMDW]:
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
def accumulate_into(dst: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin], n: Int):
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
# GEMM kernels (row-major, SIMD over columns) -- same as _tabm_kernels.mojo
# =============================================================================


# -----------------------------------------------------------------------------
# Ternary (BitLinear) quantization helpers
# -----------------------------------------------------------------------------


@always_inline
def round_half_even(x: Float32) -> Float32:
    # Matches np.round's half-to-even rule on the clipped [-1, 1] domain.
    var y = floor(x)
    var frac = x - y
    if frac > 0.5:
        y += 1.0
    elif frac == 0.5:
        # Bump only when the floor is odd; Int % keeps the sign of the
        # dividend, so odd negatives still compare non-zero.
        if Int(y) % 2 != 0:
            y += 1.0
    return y


@always_inline
def gemm_nt(m: Int, n: Int, kk: Int, a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], c: Pointer[Float32, MutUntrackedOrigin]):
    # C (m,n) = A (m,kk) @ B (n,kk)^T   -- overwrites C
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
# Ternary (BitLinear) GEMM kernels with 2-bit packed weights
# =============================================================================

# Ternary encoding: 0b00→0, 0b01→-1, 0b10→+1 (2 bits per value, 4 values/byte)

@always_inline
def _tri_code_f32(code: Int) -> Float32:
    """Branchless decode of one 2-bit ternary code to {-1.0, 0.0, +1.0}.

    Uses sign = (code >> 1) & 1 and neg = code & 1, so
        0b00 → 0 - 0 = 0,  0b01 → 0 - 1 = -1,  0b10 → 1 - 0 = +1.
    """
    return Float32((code >> 1) & 1) - Float32(code & 1)


@always_inline
def _unpack_tri4(b: UInt8) -> SIMD[DType.uint16, 4]:
    """Extract the four 2-bit codes of ``b`` into uint16 lanes (SIMD)."""
    var v = SIMD[DType.uint16, 4](UInt16(b))
    comptime SHIFTS = SIMD[DType.uint16, 4](0, 2, 4, 6)
    return (v >> SHIFTS) & 3


@always_inline
def _unpack_tri8(lo: UInt8, hi: UInt8) -> SIMD[DType.float32, SIMDW]:
    """Expand two packed bytes into SIMDW (=8) ternary level floats.

    Branchless decode: level = sign - neg with sign = (code >> 1) & 1,
    neg = code & 1, so 0b00→0, 0b01→-1, 0b10→+1.
    """
    var clo = _unpack_tri4(lo)
    var chi = _unpack_tri4(hi)
    var vlo = (((clo >> 1) & 1).cast[DType.float32]()) - (
        (clo & 1).cast[DType.float32]()
    )
    var vhi = (((chi >> 1) & 1).cast[DType.float32]()) - (
        (chi & 1).cast[DType.float32]()
    )
    var out = SIMD[DType.float32, SIMDW](0.0)
    comptime for lane in range(4):
        out[lane] = vlo[lane]
        out[lane + 4] = vhi[lane]
    return out


@always_inline
def _tri_decode_row(
    kk: Int,
    prow: Pointer[UInt8, MutUntrackedOrigin],
    wbuf: Pointer[Float32, MutUntrackedOrigin],
):
    """Decode one packed ternary weight row into ``wbuf`` as float levels."""
    var npair = kk // SIMDW
    var p = 0
    while p < npair:
        wbuf.unsafe_store[width=SIMDW](
            p * SIMDW, _unpack_tri8(prow[p * 2], prow[p * 2 + 1])
        )
        p += 1
    var t = npair * SIMDW
    while t < kk:
        var code = Int((prow[t // 4] >> UInt8((t % 4) * 2)) & 0x03)
        wbuf.unsafe_store[width=1](t, _tri_code_f32(code))
        t += 1


@always_inline
def gemm_nt_ternary(
    m: Int, n: Int, kk: Int,
    a: Pointer[Float32, MutUntrackedOrigin],
    w_packed: Pointer[UInt8, MutUntrackedOrigin],
    scales: Pointer[Float32, MutUntrackedOrigin],
    c: Pointer[Float32, MutUntrackedOrigin],
    wbuf: Pointer[Float32, MutUntrackedOrigin],
):
    """C (m,n) = A (m,kk) @ W (n,kk)^T with 2-bit packed ternary weights.

    W rows are packed four ternary levels per byte (encoding 0b00→0,
    0b01→-1, 0b10→+1, rows zero-padded to a whole byte) with one float32
    dequant scale per row:
        C[i,j] = scales[j] * Σ_k A[i,k] * level(W[j,k])

    Each weight row is decoded once per activation panel into ``wbuf``
    (float scratch; the packed source stream is 4x smaller than float32
    weights) and then multiplied against every input row with the same
    SIMD inner loop as the dense kernel; the decode cost amortizes over
    the panel rows and the scale is applied once per column.
    """
    var stride = (kk + 3) // 4
    # Activation panel size: enough rows that a panel of A stays resident in
    # L1 across the full weight-column sweep (panel bytes ≈ 48 KB cap).
    var panel = (49152 // (kk * 4 + 1)) & ~3
    if panel < 16:
        panel = 16
    var p0 = 0
    while p0 < m:
        var pe = p0 + panel
        if pe > m:
            pe = m
        var j = 0
        while j < n:
            var sc = scales[j]
            var prow = w_packed + j * stride
            _tri_decode_row(kk, prow, wbuf)
            var it = p0
            while it < pe:
                if it + 4 <= pe:
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
                        var bv = wbuf.unsafe_load[width=SIMDW](t)
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
                        s0 += ap0[t] * wbuf[t]
                        s1 += ap1[t] * wbuf[t]
                        s2 += ap2[t] * wbuf[t]
                        s3 += ap3[t] * wbuf[t]
                        t += 1
                    c[it * n + j] = s0 * sc
                    c[(it + 1) * n + j] = s1 * sc
                    c[(it + 2) * n + j] = s2 * sc
                    c[(it + 3) * n + j] = s3 * sc
                else:
                    var r = it
                    while r < pe:
                        var ap = a + r * kk
                        var acc = SIMD[DType.float32, SIMDW](0.0)
                        var t = 0
                        while t + SIMDW <= kk:
                            acc += (
                                ap.unsafe_load[width=SIMDW](t)
                                * wbuf.unsafe_load[width=SIMDW](t)
                            )
                            t += SIMDW
                        var s = acc.reduce_add()
                        while t < kk:
                            s += ap[t] * wbuf[t]
                            t += 1
                        c[r * n + j] = s * sc
                        r += 1
                it += 4
            j += 1
        p0 += panel


# ==============================================================================
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
    var dh: Pointer[Float32, MutUntrackedOrigin]
    # Per-layer caches live in one contiguous slab (activation, activation
    # derivative and dropout mask per layer).
    var slab: Pointer[Float32, MutUntrackedOrigin]
    var act_off: Pointer[Int, MutUntrackedOrigin]
    var der_off: Pointer[Int, MutUntrackedOrigin]
    var msk_off: Pointer[Int, MutUntrackedOrigin]
    var nl: Int
    # Scratch for decoding packed ternary weight rows to float levels.
    # Per-workspace (i.e. per thread) so parallel workers never share it.
    var tri_buf: Pointer[Float32, MutUntrackedOrigin]

    def __init__(
        out self,
        chunk: Int,
        nl: Int,
        layersizes: Pointer[Int, MutUntrackedOrigin],
        d_in: Int,
        dout: Int,
        F: Int,
        demb: Int,
        denc: Int,
    ):
        comptime PAD = 16
        var maxw = d_in
        var maxfi = d_in
        for i in range(nl):
            var w = Int(layersizes[i + 1])
            if w > maxw:
                maxw = w
            var fi = Int(layersizes[i])
            if fi > maxfi:
                maxfi = fi
        self.h = alloc[Float32](chunk * d_in + PAD)
        self.pl = alloc[Float32](chunk * F * demb + PAD)
        # tmp doubles as the per-feature encoding snapshot in the PLE
        # forward/backward, so it must fit the widest bin count too.
        self.tmp = alloc[Float32](chunk * max(max(maxw, F * demb), denc) + PAD)
        self.tmp2 = alloc[Float32](chunk * max(demb, maxw) + PAD)
        self.xnum = alloc[Float32](chunk * F + PAD)
        self.enc_snap = alloc[Float32](chunk * denc + PAD)
        self.preds = alloc[Float32](chunk * dout + PAD)
        self.dpreds = alloc[Float32](chunk * dout + PAD)
        self.da = alloc[Float32](chunk * maxw + PAD)
        self.dh = alloc[Float32](chunk * d_in + PAD)
        self.nl = nl
        self.act_off = alloc[Int](nl + PAD)
        self.der_off = alloc[Int](nl + PAD)
        self.msk_off = alloc[Int](nl + PAD)
        var total = 0
        for i in range(nl):
            var w = Int(layersizes[i + 1])
            self.act_off[i] = total
            total += chunk * w + PAD
            self.der_off[i] = total
            total += chunk * w + PAD
            self.msk_off[i] = total
            total += chunk * w + PAD
        self.slab = alloc[Float32](total)
        self.tri_buf = alloc[Float32](maxfi + SIMDW)

    @always_inline
    def act_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.act_off[i]

    @always_inline
    def der_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.der_off[i]

    @always_inline
    def msk_ptr(self, i: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.slab + self.msk_off[i]

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
        self.dh.unsafe_free()
        self.slab.unsafe_free()
        self.act_off.unsafe_free()
        self.der_off.unsafe_free()
        self.msk_off.unsafe_free()
        self.tri_buf.unsafe_free()


# =============================================================================
# Trainer
# =============================================================================


struct MLPTrainer(ImplicitlyCopyable, Movable, Writable):
    def write_to(mut self, mut writer: Some[Writer]):
        writer.write("MLPTrainer(P=", self.P, ")")

    var nl: Int
    var F: Int
    var denc: Int
    var ccat: Int
    var use_emb: Bool
    var demb: Int
    var act_code: Int
    var d_in: Int
    var dout: Int
    var layersizes: Pointer[Int, MutUntrackedOrigin]
    var bin_counts: Pointer[Int, MutUntrackedOrigin]
    var bin_offsets: Pointer[Int, MutUntrackedOrigin]
    var offs: Pointer[Int, MutUntrackedOrigin]
    var n_offs: Int
    var P: Int
    # BitLinear ternary quantization (0=none, 1=per_row, 2=per_tensor).
    # ``qw`` holds dequantized effective weights at the same offsets as
    # theta; refreshed whenever theta changes.
    # ``qw_packed`` holds 2-bit packed ternary weights (4 values per byte);
    # ``qw_scales`` holds per-row dequant scales for the packed format.
    var quant: Int
    var quant_out: Bool
    var qw: Optional[Pointer[Float32, MutUntrackedOrigin]]
    var qw_packed: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var qw_scales: Optional[Pointer[Float32, MutUntrackedOrigin]]
    # Per-layer byte offsets into qw_packed / element offsets into
    # qw_scales (padded strides make P//4 indexing wrong when fin % 4 != 0).
    var pk_off: Pointer[Int, MutUntrackedOrigin]
    var sc_off: Pointer[Int, MutUntrackedOrigin]

    def __init__(out self, dims: PythonObject, layersizes: PythonObject, bins: PythonObject) raises:
        var dp = ptr_i64(dims)
        self.F = Int(dp[0])
        self.denc = Int(dp[1])
        self.ccat = Int(dp[2])
        self.use_emb = Int(dp[3]) == 1
        self.demb = Int(dp[4])
        self.act_code = Int(dp[5])

        var nl = Int(len(layersizes)) - 1
        if nl < 1:
            raise Error("layer sizes need at least input and output entries")
        self.nl = nl
        self.layersizes = alloc[Int](nl + 1 + 16)
        for i in range(nl + 1):
            self.layersizes[i] = Int(py=layersizes[i])
        self.d_in = Int(self.layersizes[0])
        self.dout = Int(self.layersizes[nl])

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

        # Parameter offsets mirroring MLPParams.arrays insertion order.
        var ec = 0
        if self.use_emb and self.F > 0:
            ec = 2 + self.F
        self.n_offs = ec + 2 * self.nl
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
        # Layer i maps layer_sizes[i] -> layer_sizes[i+1]; weights stored
        # row-major (fan_out, fan_in) so forward evaluates x @ W.T.
        for i in range(self.nl):
            var fan_in = Int(self.layersizes[i])
            var fan_out = Int(self.layersizes[i + 1])
            self.offs[pos] = cur
            pos += 1
            cur += fan_out * fan_in  # l{i}_w
            self.offs[pos] = cur
            pos += 1
            cur += fan_out  # l{i}_b
        self.P = cur

        self.quant = 0
        self.quant_out = False
        self.qw = None
        self.qw_packed = None
        self.qw_scales = None
        self.pk_off = alloc[Int](self.nl + 17)
        self.sc_off = alloc[Int](self.nl + 17)
        if len(dims) > 6:
            self.quant = Int(dp[6])
            if self.quant < 0 or self.quant > 2:
                raise Error("dims[6] must be a quantization code in {0, 1, 2}")
            if len(dims) > 7:
                self.quant_out = Int(dp[7]) == 1
            self.qw = alloc[Float32](self.P + 16)
            # Packed ternary storage: 2 bits per weight (4 values/byte,
            # row stride rounds up) plus one float32 scale per output row.
            var tot_pk = 0
            var tot_sc = 0
            for li in range(self.nl):
                self.pk_off[li] = tot_pk
                self.sc_off[li] = tot_sc
                var fi = Int(self.layersizes[li])
                var fo = Int(self.layersizes[li + 1])
                tot_pk += fo * ((fi + 3) // 4)
                tot_sc += fo
            self.pk_off[self.nl] = tot_pk
            self.sc_off[self.nl] = tot_sc
            self.qw_packed = alloc[UInt8](tot_pk + 16)
            self.qw_scales = alloc[Float32](tot_sc + 16)

    @staticmethod
    def py_init(out self: MLPTrainer, args: PythonObject, kwargs: PythonObject) raises:
        _ = kwargs
        if len(args) != 3:
            raise Error("MLPTrainer(dims, layersizes, bins) expects 3 arguments")
        self = Self(args[0], args[1], args[2])

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
        return self.offs[self.emb_count() + 2 * i]

    @always_inline
    def off_b(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 2 * i + 1]

    @always_inline
    def fan_in(self, i: Int) -> Int:
        return Int(self.layersizes[i])

    @always_inline
    def fan_out(self, i: Int) -> Int:
        return Int(self.layersizes[i + 1])

    @always_inline
    def layer_quantized(self, i: Int) -> Bool:
        """Whether layer ``i``'s weight matrix is ternary-quantized.

        Mirrors MLPConfig.layer_is_quantized: hidden layers only, unless
        the output layer was explicitly included.
        """
        if self.quant == 0:
            return False
        if i == self.nl - 1:
            return self.quant_out
        return True

    @always_inline
    def weight_for_fwd(
        self, theta: Pointer[Float32, MutUntrackedOrigin], i: Int
    ) -> Pointer[Float32, MutUntrackedOrigin]:
        """Weight pointer for forward/input-gradient GEMMs of layer ``i``.

        Quantized layers read from the refreshed scratch buffer; gradient
        accumulation always uses the latent theta weights (STE).
        """
        if self.layer_quantized(i):
            return self.qw.value() + self.off_w(i)
        return theta + self.off_w(i)

    @always_inline
    def weight_for_fwd_ternary(
        self, i: Int
    ) -> Tuple[Pointer[UInt8, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Int, Int]:
        """Return (packed_weights_ptr, scales_ptr, fan_in, fan_out) for ternary GEMM.

        Weights are 2-bit packed (4 ternary values per byte); scales are
        per-row dequant multipliers.
        """
        var fin = self.fan_in(i)
        var fout = self.fan_out(i)
        return (self.qw_packed.value() + self.pk_off[i],
                self.qw_scales.value() + self.sc_off[i], fin, fout)

    def refresh_quantized(
        self, theta: Pointer[Float32, MutUntrackedOrigin]
    ):
        """Recompute ternary effective weights into ``qw`` and pack into ``qw_packed``.

        Must be called whenever the latent weights change (after each Adam
        round / L-BFGS candidate step) and before any GEMM consumes them.
        Single-threaded by design: it runs before workers are spawned.
        Scale = mean(|W|) per output row (per_row) or over the whole
        matrix (per_tensor); rounding is half-to-even, matching np.round.
        """
        if self.quant == 0:
            return
        var i = 0
        while i < self.nl:
            if self.layer_quantized(i):
                var fin = self.fan_in(i)
                var fout = self.fan_out(i)
                var src = theta + self.off_w(i)
                var dst = self.qw.value() + self.off_w(i)
                var is_tensor = self.quant == 2
                self._quant_pack(src, dst,
                                 self.qw_packed.value() + self.pk_off[i],
                                 self.qw_scales.value() + self.sc_off[i],
                                 fin, fout, is_tensor)
            i += 1

    @always_inline
    def _quant_pack_row(
        self,
        src: Pointer[Float32, MutUntrackedOrigin],
        qw_dst: Pointer[Float32, MutUntrackedOrigin],
        pk_dst: Pointer[UInt8, MutUntrackedOrigin],
        gamma: Float32,
        fin: Int,
    ):
        """Quantize-dequantize one row into ``qw_dst`` and pack its ternary
        levels into ``pk_dst``.

        For clamped q in [-1, 1], ceil-based level indicators reproduce
        round-half-even exactly (+1 iff q > 0.5, -1 iff q < -0.5, ties go
        to zero). The per-lane IEEE division matches the scalar path
        bit-for-bit.
        """
        var t = 0
        while t + SIMDW <= fin:
            var sv = src.unsafe_load[width=SIMDW](t)
            var q = sv / SIMD[DType.float32, SIMDW](gamma)
            q = min(max(q, -1.0), 1.0)
            # Exact level indicators on clamped q in [-1, 1]:
            #   max(ceil(q-0.5), 0)  is 1 iff q > 0.5,
            #   max(ceil(-q-0.5), 0) is 1 iff q < -0.5;
            # ties at ±0.5 give 0, matching round-half-even to zero.
            var posf = max(ceil(q - 0.5), 0.0)
            var negf = max(ceil(-q - 0.5), 0.0)
            qw_dst.unsafe_store[width=SIMDW](
                t, (posf - negf) * gamma
            )
            var b0: UInt8 = 0
            var b1: UInt8 = 0
            comptime for lane in range(4):
                var c0 = UInt8(posf[lane]) * 2 + UInt8(negf[lane])
                b0 |= c0 << UInt8(lane * 2)
                var c1 = UInt8(posf[lane + 4]) * 2 + UInt8(negf[lane + 4])
                b1 |= c1 << UInt8(lane * 2)
            pk_dst[t // 4] = b0
            pk_dst[t // 4 + 1] = b1
            t += SIMDW
        while t < fin:
            var q = src[t] / gamma
            if q > 1.0:
                q = 1.0
            elif q < -1.0:
                q = -1.0
            var lvl = round_half_even(q)
            qw_dst[t] = lvl * gamma
            var tri: UInt8 = 0
            if lvl > 0.0:
                tri = 2
            elif lvl < 0.0:
                tri = 1
            pk_dst[t // 4] = pk_dst[t // 4] | (tri << UInt8((t % 4) * 2))
            t += 1

    @always_inline
    def _absmean_scale(
        self, src: Pointer[Float32, MutUntrackedOrigin], count: Int
    ) -> Float32:
        """Mean |x| over ``count`` values, accumulated in float64 (values
        are widened exactly, so only f64 addition order differs from the
        scalar loop -- far below any rounding boundary)."""
        var acc: Float64 = 0.0
        var t = 0
        while t + SIMDW <= count:
            acc += abs(
                src.unsafe_load[width=SIMDW](t)
            ).cast[DType.float64]().reduce_add()
            t += SIMDW
        while t < count:
            var v = src[t]
            acc += Float64(v if v >= 0 else -v)
            t += 1
        var gamma = Float32(acc / Float64(count))
        if gamma < 1e-12:
            gamma = 1e-12
        return gamma

    @always_inline
    def _quant_pack(
        self,
        src: Pointer[Float32, MutUntrackedOrigin],
        qw_dst: Pointer[Float32, MutUntrackedOrigin],
        pk_dst: Pointer[UInt8, MutUntrackedOrigin],
        scales_dst: Pointer[Float32, MutUntrackedOrigin],
        fin: Int, fout: Int, is_tensor: Bool,
    ):
        """Ternary-quantize into ``qw_dst`` AND pack levels into ``pk_dst``.

        One shared absmean scale per row (per_row) or matrix (per_tensor)
        feeds both outputs, so the packed-GEMM forward stays bit-identical
        to the dequantized-float32 forward. Levels use round-half-even of
        w/gamma (matching shinrin._quant); encoding 0b00→0, 0b01→-1,
        0b10→+1, four values per byte, rows zero-padded to whole bytes.
        """
        var count = fin * fout
        var stride = (fin + 3) // 4
        if is_tensor:
            var gamma = self._absmean_scale(src, count)
            var r = 0
            while r < fout:
                scales_dst[r] = gamma
                r += 1
            # Clear all packed bytes before ORing bits in (alloc memory is
            # uninitialized and refresh runs repeatedly).
            var b = 0
            while b < fout * stride:
                pk_dst[b] = 0
                b += 1
            r = 0
            while r < fout:
                self._quant_pack_row(
                    src + r * fin, qw_dst + r * fin, pk_dst + r * stride,
                    gamma, fin,
                )
                r += 1
        else:
            var r = 0
            while r < fout:
                var row = src + r * fin
                var gamma = self._absmean_scale(row, fin)
                scales_dst[r] = gamma
                var prow = pk_dst + r * stride
                # Clear the row's bytes before ORing bits in.
                var b = 0
                while b < stride:
                    prow[b] = 0
                    b += 1
                self._quant_pack_row(row, qw_dst + r * fin, prow, gamma, fin)
                r += 1

    # -- forward -----------------------------------------------------------------

    def embed_forward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        x_num: Pointer[Float32, MutUntrackedOrigin],
        x_enc: Pointer[Float32, MutUntrackedOrigin],
        rows: Pointer[Int, MutUntrackedOrigin],
        b: Int,
    ):
        """Build ws.h (b, d_in); snapshots inputs for the backward pass."""
        var demb = self.demb
        if not (self.use_emb and self.F > 0):
            return
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
                var xv = ws.xnum[r * self.F + f]
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
            var r2 = 0
            while r2 < b:
                var t = 0
                while t < cnt:
                    ws.tmp[r2 * cnt + t] = ws.enc_snap[r2 * self.denc + enc_off + t]
                    t += 1
                r2 += 1
            gemm_nn(b, demb, cnt, ws.tmp, wp, ws.tmp2)
            r2 = 0
            while r2 < b:
                var o = 0
                while o < demb:
                    # tmp2 holds the pre-ReLU piecewise projection.
                    var val = ws.tmp2[r2 * demb + o]
                    if val > 0:
                        ws.h[r2 * self.d_in + f * demb + o] += val
                    o += 1
                r2 += 1

    def backbone_forward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        b: Int,
        mut rng: Rng,
        dropout: Float32,
        train: Bool,
    ):
        """ws.h -> per-layer caches; raw output predictions in ws.preds."""
        var keep = 1.0 / (1.0 - dropout)
        var use_dropout = train and dropout > 0.0
        var thresh = UInt64(dropout * 256.0)
        if thresh < 1:
            thresh = 1
        var bits: UInt64 = 0
        var nbits = 0
        var n_hidden = self.nl - 1
        var i = 0
        while i < self.nl:
            var fin = self.fan_in(i)
            var fout = self.fan_out(i)
            var w = self.weight_for_fwd(theta, i)
            var bias = theta + self.off_b(i)
            # input to this layer
            var inp: Pointer[Float32, MutUntrackedOrigin] = ws.h
            if i > 0:
                inp = ws.act_ptr(i - 1)
            # z = inp @ W^T into ws.tmp2 buffer region? use da as z storage
            var z = ws.da
            # Quantized layers run the packed-ternary GEMM; gradient work
            # below still uses latent theta weights (STE).
            if self.layer_quantized(i):
                var (packed_w, scales, _, _) = self.weight_for_fwd_ternary(i)
                gemm_nt_ternary(
                    b, fout, fin, inp, packed_w, scales, z, ws.tri_buf
                )
            else:
                gemm_nt(b, fout, fin, inp, w, z)
            # add bias, apply activation (+ dropout on hidden layers)
            var act = ws.act_ptr(i)
            var der = ws.der_ptr(i)
            var msk = ws.msk_ptr(i)
            var n_hidden_local = self.nl - 1
            var is_hidden = i < n_hidden_local
            var row = 0
            while row < b:
                var base = row * fout
                var j = 0
                # Vectorized fast path: relu/identity without dropout are pure
                # SIMD ops; other activations still benefit from the layout.
                while j + SIMDW <= fout:
                    var idx2 = base + j
                    if not is_hidden:
                        var zv2 = (
                            z.unsafe_load[width=SIMDW](idx2)
                            + bias.unsafe_load[width=SIMDW](j)
                        )
                        msk.unsafe_store[width=SIMDW](idx2, SIMD[DType.float32, SIMDW](1.0))
                        der.unsafe_store[width=SIMDW](idx2, SIMD[DType.float32, SIMDW](1.0))
                        act.unsafe_store[width=SIMDW](idx2, zv2)
                    elif self.act_code == 3 and not use_dropout:
                        var zv3 = (
                            z.unsafe_load[width=SIMDW](idx2)
                            + bias.unsafe_load[width=SIMDW](j)
                        )
                        var ones = SIMD[DType.float32, SIMDW](1.0)
                        var zeros = SIMD[DType.float32, SIMDW](0.0)
                        var dv3 = SIMD[DType.float32, SIMDW](0.0)
                        comptime for lane in range(SIMDW):
                            dv3[lane] = 1.0 if zv3[lane] > 0.0 else 0.0
                        msk.unsafe_store[width=SIMDW](idx2, ones)
                        der.unsafe_store[width=SIMDW](idx2, dv3)
                        act.unsafe_store[width=SIMDW](idx2, max(zv3, zeros))
                    else:
                        var l = 0
                        while l < SIMDW:
                            var idx3 = idx2 + l
                            var zv = z[idx3] + bias[j + l]
                            if not is_hidden:
                                msk[idx3] = 1.0
                                der[idx3] = 1.0
                                act[idx3] = zv
                            else:
                                var av = act_apply(self.act_code, zv)
                                var dv = Float32(1.0)
                                if use_dropout:
                                    if nbits < 8:
                                        bits = rng.next_u64()
                                        nbits = 64
                                    dv = keep if ((bits >> 56) & 0xFF) >= thresh else 0.0
                                    bits <<= 8
                                    nbits -= 8
                                msk[idx3] = dv
                                der[idx3] = act_deriv(self.act_code, zv)
                                act[idx3] = av * dv
                            l += 1
                    j += SIMDW
                while j < fout:
                    var idx4 = base + j
                    var zv = z[idx4] + bias[j]
                    if not is_hidden:
                        msk[idx4] = 1.0
                        der[idx4] = 1.0
                        act[idx4] = zv
                    else:
                        var av = act_apply(self.act_code, zv)
                        var dv = Float32(1.0)
                        if use_dropout:
                            if nbits < 8:
                                bits = rng.next_u64()
                                nbits = 64
                            dv = keep if ((bits >> 56) & 0xFF) >= thresh else 0.0
                            bits <<= 8
                            nbits -= 8
                        msk[idx4] = dv
                        der[idx4] = act_deriv(self.act_code, zv)
                        act[idx4] = av * dv
                    j += 1
                row += 1
            # stash raw predictions of the final layer in ws.preds
            if i == self.nl - 1:
                var n_out = b * fout
                var cpy = 0
                while cpy + SIMDW <= n_out:
                    ws.preds.unsafe_store[width=SIMDW](
                        cpy, act.unsafe_load[width=SIMDW](cpy)
                    )
                    cpy += SIMDW
                while cpy < n_out:
                    ws.preds[cpy] = act[cpy]
                    cpy += 1
            i += 1

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
        """Build inputs (embedding or passthrough), then run the network.

        Without the embedding, ws.h directly receives the numerical block
        followed by the categorical one-hot block.
        """
        if self.use_emb and self.F > 0:
            self.embed_forward(theta, ws, x_num, x_enc, rows, b)
            if self.ccat > 0:
                var r = 0
                while r < b:
                    var src = rows[r]
                    var t = 0
                    while t < self.ccat:
                        ws.h[r * self.d_in + self.F * self.demb + t] = x_cat[src * self.ccat + t]
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
        self.backbone_forward(theta, ws, b, rng, dropout, train)


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
        """Fill ws.dpreds; returns the chunk loss (per-chunk mean)."""
        var gtotal = denom_b
        var loss: Float64 = 0.0
        var row = 0
        while row < b:
            var src = rows[row]
            if task == 0:
                var o = 0
                while o < self.dout:
                    var diff = ws.preds[row * self.dout + o] - y[src * self.dout + o]
                    loss += 0.5 * Float64(diff * diff)
                    ws.dpreds[row * self.dout + o] = diff / Float32(gtotal)
                    o += 1
            elif task == 1:
                # BCE with logits: loss = softplus(z) - y*z ; dpred = sigmoid(z)-y
                var z = ws.preds[row * self.dout]
                var target = y[src * self.dout]
                loss += Float64(softplus(z) - target * z)
                ws.dpreds[row * self.dout] = (sigmoid(z) - target) / Float32(gtotal)
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
                    ws.dpreds[base + o] = (sm - onehot) / Float32(gtotal)
                    o += 1
            row += 1
        # Return the per-chunk mean (run_round re-weights by slice size);
        # only dpreds scaling uses the full-minibatch denominator.
        return loss / Float64(b)


# === PART3B ===

    # -- backward ------------------------------------------------------------------

    def backward_chunk(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        g: Pointer[Float32, MutUntrackedOrigin],
        b: Int,
    ):
        """Accumulate parameter gradients from ws.dpreds into g."""
        var da = ws.da
        # da doubles as the z scratch during forward; seed it with dpreds.
        var n_last = b * self.fan_out(self.nl - 1)
        var idx = 0
        while idx + SIMDW <= n_last:
            da.unsafe_store[width=SIMDW](idx, ws.dpreds.unsafe_load[width=SIMDW](idx))
            idx += SIMDW
        while idx < n_last:
            da[idx] = ws.dpreds[idx]
            idx += 1

        var i = self.nl - 1
        while i >= 0:
            var fin = self.fan_in(i)
            var fout = self.fan_out(i)
            # Input-gradients use the effective (possibly quantized)
            # weights; gradient accumulation below stays on the latent
            # theta weights — the straight-through-estimator contract.
            var w = self.weight_for_fwd(theta, i)
            var inp: Pointer[Float32, MutUntrackedOrigin] = ws.h
            if i > 0:
                inp = ws.act_ptr(i - 1)

            # grads l{i}_w += da^T @ inp ; l{i}_b += sum rows da
            gemm_tn_acc(fout, fin, b, da, inp, g + self.off_w(i))
            var bias_grad = g + self.off_b(i)
            var row = 0
            while row < b:
                var j = 0
                while j < fout:
                    bias_grad[j] += da[row * fout + j]
                    j += 1
                row += 1

            if i == 0:
                if self.use_emb and self.F > 0:
                    # dh = da @ W_0 ; embedding gradients follow from dh
                    gemm_nn(b, fin, fout, da, w, ws.dh)
                    self.embed_backward(theta, ws, g, b)
                break

            # dz_{i-1} = (da @ W_i) * deriv_{i-1} [* mask_{i-1}]
            # Copy da first: the GEMM below overwrites its destination.
            var tmpbuf = ws.tmp2
            var nb = b * fout
            idx = 0
            while idx + SIMDW <= nb:
                tmpbuf.unsafe_store[width=SIMDW](idx, da.unsafe_load[width=SIMDW](idx))
                idx += SIMDW
            while idx < nb:
                tmpbuf[idx] = da[idx]
                idx += 1
            var dz_prev = ws.da
            gemm_nn(b, fin, fout, tmpbuf, w, dz_prev)
            var der_prev = ws.der_ptr(i - 1)
            var msk_prev = ws.msk_ptr(i - 1)
            nb = b * fin
            idx = 0
            while idx + SIMDW <= nb:
                var v = dz_prev.unsafe_load[width=SIMDW](idx)
                v *= der_prev.unsafe_load[width=SIMDW](idx)
                v *= msk_prev.unsafe_load[width=SIMDW](idx)
                dz_prev.unsafe_store[width=SIMDW](idx, v)
                idx += SIMDW
            while idx < nb:
                dz_prev[idx] *= der_prev[idx] * msk_prev[idx]
                idx += 1
            i -= 1

    def embed_backward(
        self,
        theta: Pointer[Float32, MutUntrackedOrigin],
        ws: Workspace,
        g: Pointer[Float32, MutUntrackedOrigin],
        b: Int,
    ):
        """Gradients for embeddings from ws.dh (b, d_in); uses snapshots.

        The pre-ReLU piecewise projections are recomputed from the stored
        ``xnum`` / ``enc_snap`` snapshots so no extra pl buffer is needed.
        """
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
                        dwp.unsafe_store[width=SIMDW](
                            o2, dwp.unsafe_load[width=SIMDW](o2) + xn_simd * dg
                        )
                        dbp.unsafe_store[width=SIMDW](
                            o2, dbp.unsafe_load[width=SIMDW](o2) + dg
                        )
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
        for f in range(self.F):
            var cnt = self.bin_counts[f]
            var wp = theta + self.off_emb_wp(f)
            var enc_off = self.bin_offsets[f]
            var r2 = 0
            while r2 < b:
                var t = 0
                while t < cnt:
                    ws.tmp[r2 * cnt + t] = ws.enc_snap[r2 * self.denc + enc_off + t]
                    t += 1
                r2 += 1
            # tmp holds (b, cnt); gemm_nn reads it and writes tmp2 -> safe.
            gemm_nn(b, demb, cnt, ws.tmp, wp, ws.tmp2)
            r2 = 0
            while r2 < b:
                var o = 0
                while o < demb:
                    var plv = ws.tmp2[r2 * demb + o]
                    var dg = ws.dh[r2 * self.d_in + f * demb + o]
                    ws.tmp2[r2 * demb + o] = dg if plv >= 0.0 else 0.0
                    o += 1
                r2 += 1
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
        var res = k_adam_epoch(
            self, theta, m, v, t0, x_num, x_enc, x_cat, y, N,
            lr, bs, dropout, alpha, seed, task, resolve_threads(),
        )
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
        var nit = k_lbfgs(
            self, theta, x_num, x_enc, x_cat, y, N,
            max_iter, tol, maxcor, alpha, losses_out, task, resolve_threads(),
        )
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
        var loss = k_full_loss_grad(
            self, theta, grad, x_num, x_enc, x_cat, y, N, task, True, alpha,
            resolve_threads(),
        )
        return Python.tuple(Float64(loss), grad_arr)

    @staticmethod
    def forward(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var theta = ptr_f32(parts[0])
        self.refresh_quantized(theta)
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var out = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)
        var rng = Rng(1)
        var chunk = 8192
        var idx = alloc[Int](N)
        var i = 0
        while i < N:
            idx[i] = i
            i += 1
        var ws = Workspace(
            chunk, self.nl, self.layersizes, self.d_in, self.dout,
            self.F, self.demb, self.denc,
        )
        var start = 0
        while start < N:
            var b = min(chunk, N - start)
            self.forward_chunk(
                theta, ws, x_num, x_enc, x_cat, idx + start, b, rng, 0.0, False
            )
            var n_out = b * self.dout
            var k = 0
            while k + SIMDW <= n_out:
                out.unsafe_store[width=SIMDW](
                    (start * self.dout) + k, ws.preds.unsafe_load[width=SIMDW](k)
                )
                k += SIMDW
            while k < n_out:
                out[start * self.dout + k] = ws.preds[k]
                k += 1
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
    var tr: MLPTrainer
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
        tr: MLPTrainer,
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
        self.ws = Workspace(
            max_rows, tr.nl, tr.layersizes, tr.d_in, tr.dout, tr.F, tr.demb, tr.denc
        )
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
    """Vectorized bias-corrected Adam update (mirrors _AdamOptimizer)."""
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
    tr: MLPTrainer,
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
    var T = nthreads
    # Scale threads with the round size: spawn/sync costs dominate small
    # minibatch GEMMs, so give each worker at least ~64 rows.
    if nrows_total > 0:
        var by_rows = (nrows_total + 63) // 64
        if by_rows < T:
            T = by_rows
    if nrows_total < 128:
        T = 1
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
        var lo2 = assigned
        var hi2 = min(lo2 + rows_per, nrows_total)
        assigned = hi2
        wsum += workers[t2].loss_out * Float64(hi2 - lo2)
        accumulate_into(g_out, workers[t2].g, tr.P)
    return wsum


def k_full_loss_grad(
    tr: MLPTrainer,
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
    # L-BFGS re-evaluates this after every candidate step, so refreshing
    # here keeps qw in sync with the current latent weights.
    tr.refresh_quantized(theta)
    vec_zero(grad, tr.P)
    var chunk = 8192
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

def k_adam_epoch(
    tr: MLPTrainer,
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
    # Quantized effective weights must track theta, which changes after
    # every Adam round below.
    tr.refresh_quantized(theta)
    var idx = alloc[Int](N)
    var i = 0
    while i < N:
        idx[i] = i
        i += 1
    # Fisher-Yates shuffle (deterministic for fixed seed + thread count)
    var ii = N - 1
    while ii > 0:
        var jj = Int(rng.next_u64() % UInt64(ii + 1))
        var tmp = idx[ii]
        idx[ii] = idx[jj]
        idx[jj] = tmp
        ii -= 1

    var chunk = 8192
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
        t += 1
        adam_update(theta, m, v, g, tr.P, alpha, lr, 1.0 - pow(0.9, Float64(t)), 1.0 - pow(0.999, Float64(t)))
        tr.refresh_quantized(theta)
        weighted += batch_loss * Float64(nb_rows)
        start = be
    for t3 in range(nthreads):
        workers[t3].release()
    idx.unsafe_free()
    g.unsafe_free()
    return (weighted / Float64(N), t)


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


def k_lbfgs(
    tr: MLPTrainer,
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
    """Full-batch L-BFGS with backtracking line search (same as TabM's)."""
    var P = tr.P
    # Model state and evaluations stay float32; all L-BFGS bookkeeping runs
    # in float64 so small s/y differences keep their curvature information.
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

    var loss = tr.full_loss_grad(theta, g, x_num, x_enc, x_cat, y, N, task, True, alpha, nthreads)
    losses_out[0] = loss
    var n_losses = 1
    var i = 0
    while i < P:
        th64[i] = Float64(theta[i])
        g64[i] = Float64(g[i])
        i += 1

    var nhist = 0
    var hptr = 0
    var last_ys: Float64 = 1.0
    var last_yy: Float64 = 1.0
    var first_iter = True
    var consec_fail = 0
    var it = 0
    while it < max_iter:
        if max_abs_f32(g, P) <= tol:
            break
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
            var loss_new = tr.full_loss_grad(cand, gnew, x_num, x_enc, x_cat, y, N, task, True, alpha, nthreads)
            if loss_new <= loss + 1e-4 * step * dg:
                i = 0
                while i < P:
                    gnew64[i] = Float64(gnew[i])
                    S[hptr * P + i] = cand64[i] - th64[i]
                    Y[hptr * P + i] = gnew64[i] - g64[i]
                    i += 1
                var ys = dot_f64(S + hptr * P, Y + hptr * P, P)
                var yy = dot_f64(Y + hptr * P, Y + hptr * P, P)
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
        it += 1

    var result = n_losses - 1
    g.free(); gnew.free(); cand.free()
    th64.free(); g64.free(); gnew64.free(); cand64.free(); dir.free()
    S.free(); Y.free(); rho.free(); alphas.free()
    return result


@export
def PyInit__native_mlp() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native_mlp")
        _ = (
            m.add_type[MLPTrainer]("MLPTrainer")
            .def_py_init[MLPTrainer.py_init]()
            .def_method[MLPTrainer.adam_epoch]("adam_epoch")
            .def_method[MLPTrainer.lbfgs_minimize]("lbfgs_minimize")
            .def_method[MLPTrainer.loss_grad]("loss_grad")
            .def_method[MLPTrainer.forward]("forward")
            .def_method[MLPTrainer.param_count]("param_count")
        )
        var mod = m.finalize()
        return mod
    except e:
        abort(String("failed to create module _native_mlp: ", e))
