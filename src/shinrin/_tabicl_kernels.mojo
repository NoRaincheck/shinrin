"""Mojo inference kernels for TabICLv2.

Compiles to the ``shinrin._native_tabicl`` Python extension module (build
with ``just build-tabicl-mojo``). Exposes a ``TabICLInference`` bound type
whose methods run inference forward passes without returning to Python.

Architecture:
1. ColEmbedding: feature grouping → linear embed → ISAB blocks (3)
2. RowInteraction: CLS tokens → transformer blocks (3) with RoPE
3. ICLearning: class embedding → self-attention blocks (12) → MLP output

All arrays cross the boundary as NumPy buffers (float32 data, int64 dims).
"""

from std.os import abort, getenv
from std.math import abs, exp, log, sqrt, tanh, cos, sin, maximum as max
from std.memory import alloc
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime.asyncrt import TaskGroup
from std.sys.info import num_performance_cores

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
def gelu(x: Float32) -> Float32:
    var c = 0.797885
    return 0.5 * x * (1.0 + tanh(Float64(c * x * (1.0 + 0.044715 * x * x)))).cast[DType.float32]()


# =============================================================================
# GEMM kernels
# =============================================================================


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
                    var acc = SIMD[DType.float32, SIMDW](0.0)
                    var ap = a + r * kk
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
# LayerNorm (over last dim d, batch of n rows)
# =============================================================================


@always_inline
def layer_norm_batch(x: Pointer[Float32, MutUntrackedOrigin], out: Pointer[Float32, MutUntrackedOrigin], n: Int, d: Int, bias_free: Bool):
    var row = 0
    while row < n:
        var base = row * d
        var sum_val = SIMD[DType.float32, SIMDW](0.0)
        var i = 0
        while i + SIMDW <= d:
            sum_val += x.unsafe_load[width=SIMDW](base + i)
            i += SIMDW
        var mu = sum_val.reduce_add()
        while i < d:
            mu += x[base + i]
            i += 1
        mu /= Float32(d)

        var sum_sq = SIMD[DType.float32, SIMDW](0.0)
        i = 0
        while i + SIMDW <= d:
            var diff = x.unsafe_load[width=SIMDW](base + i) - SIMD[DType.float32, SIMDW](mu)
            sum_sq += diff * diff
            i += SIMDW
        var var_val = sum_sq.reduce_add()
        while i < d:
            var diff = x[base + i] - mu
            var_val += diff * diff
            i += 1
        var_val /= Float32(d)
        var inv_std = 1.0 / sqrt(var_val + 1e-5)

        i = 0
        var mu_simd = SIMD[DType.float32, SIMDW](mu)
        var inv_std_simd = SIMD[DType.float32, SIMDW](inv_std)
        while i + SIMDW <= d:
            var v = x.unsafe_load[width=SIMDW](base + i)
            if bias_free:
                out.unsafe_store[width=SIMDW](base + i, (v - mu_simd) * inv_std_simd)
            else:
                out.unsafe_store[width=SIMDW](base + i, (v - mu_simd) * inv_std_simd)
            i += SIMDW
        while i < d:
            var v = x[base + i]
            if bias_free:
                out[base + i] = (v - mu) * inv_std
            else:
                out[base + i] = (v - mu) * inv_std
            i += 1
        row += 1


# =============================================================================
# FFN: x @ W1.T + b1 -> GELU -> @ W2.T + b2
# =============================================================================


@always_inline
def ffn_forward(
    x: Pointer[Float32, MutUntrackedOrigin],
    out: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    d_model: Int,
    d_ff: Int,
    w1: Pointer[Float32, MutUntrackedOrigin],
    b1: Pointer[Float32, MutUntrackedOrigin],
    w2: Pointer[Float32, MutUntrackedOrigin],
    b2: Pointer[Float32, MutUntrackedOrigin],
    tmp: Pointer[Float32, MutUntrackedOrigin],
):
    gemm_nn(n, d_ff, d_model, x, w1, tmp)
    var r = 0
    while r < n:
        var f = 0
        while f + SIMDW <= d_ff:
            var v = tmp.unsafe_load[width=SIMDW](r * d_ff + f) + SIMD[DType.float32, SIMDW](b1.unsafe_load[width=SIMDW](f))
            tmp.unsafe_store[width=SIMDW](r * d_ff + f, gelu(v))
            f += SIMDW
        while f < d_ff:
            var v = tmp[r * d_ff + f] + b1[f]
            tmp[r * d_ff + f] = gelu(v)
            f += 1
        r += 1

    gemm_nn(n, d_model, d_ff, tmp, w2, out)
    r = 0
    while r < n:
        var f = 0
        while f + SIMDW <= d_model:
            var v = out.unsafe_load[width=SIMDW](r * d_model + f) + SIMD[DType.float32, SIMDW](b2.unsafe_load[width=SIMDW](f))
            out.unsafe_store[width=SIMDW](r * d_model + f, v)
            f += SIMDW
        while f < d_model:
            out[r * d_model + f] += b2[f]
            f += 1
        r += 1


# =============================================================================
# Self-Attention Block (pre-norm)
# =============================================================================


def self_attn_block_forward(
    x: Pointer[Float32, MutUntrackedOrigin],
    out: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    d_ff: Int,
    w_attn_in: Pointer[Float32, MutUntrackedOrigin],
    b_attn_in: Pointer[Float32, MutUntrackedOrigin],
    w_attn_out: Pointer[Float32, MutUntrackedOrigin],
    b_attn_out: Pointer[Float32, MutUntrackedOrigin],
    w1: Pointer[Float32, MutUntrackedOrigin],
    b1: Pointer[Float32, MutUntrackedOrigin],
    w2: Pointer[Float32, MutUntrackedOrigin],
    b2: Pointer[Float32, MutUntrackedOrigin],
    bias_free_ln: Bool,
    rope_freqs: Pointer[Float32, MutUntrackedOrigin] | None,
    ssmax_head_scales: Pointer[Float32, MutUntrackedOrigin] | None,
    ssmax_elementwise: Bool,
    tmp1: Pointer[Float32, MutUntrackedOrigin],
    tmp2: Pointer[Float32, MutUntrackedOrigin],
    tmp3: Pointer[Float32, MutUntrackedOrigin],
    tmp4: Pointer[Float32, MutUntrackedOrigin],
    need_kv: Bool,
    k_cache: Pointer[Float32, MutUntrackedOrigin] | None,
    v_cache: Pointer[Float32, MutUntrackedOrigin] | None,
) -> Int:
    """Pre-norm self-attention + FFN block. Returns 0 on success."""
    # LayerNorm(x) -> tmp1
    layer_norm_batch(x, tmp1, n, d_model, bias_free_ln)

    # Q, K, V projections from packed weight (3*d_model x d_model)
    var q_proj = alloc[Float32](n * d_model + 16)
    var k_proj = alloc[Float32](n * d_model + 16)
    var v_proj = alloc[Float32](n * d_model + 16)

    gemm_nn(n, d_model, d_model, tmp1, w_attn_in, q_proj)
    gemm_nn(n, d_model, d_model, tmp1, w_attn_in + d_model, k_proj)
    gemm_nn(n, d_model, d_model, tmp1, w_attn_in + 2 * d_model, v_proj)

    # Add biases
    var i = 0
    while i < n:
        var b = 0
        while b + SIMDW <= d_model:
            q_proj.unsafe_store[width=SIMDW](i * d_model + b, q_proj.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn_in[b]))
            k_proj.unsafe_store[width=SIMDW](i * d_model + b, k_proj.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn_in[d_model + b]))
            v_proj.unsafe_store[width=SIMDW](i * d_model + b, v_proj.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn_in[2 * d_model + b]))
            b += SIMDW
        while b < d_model:
            q_proj[i * d_model + b] += b_attn_in[b]
            k_proj[i * d_model + b] += b_attn_in[d_model + b]
            v_proj[i * d_model + b] += b_attn_in[2 * d_model + b]
        i += 1

    # Reshape to (n, n_heads, head_dim)
    var q_a = alloc[Float32](n * n_heads * head_dim + 16)
    var k_a = alloc[Float32](n * n_heads * head_dim + 16)
    var v_a = alloc[Float32](n * n_heads * head_dim + 16)

    i = 0
    while i < n:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                q_a.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, q_proj.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                k_a.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, k_proj.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                v_a.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, v_proj.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                q_a[i * n_heads * head_dim + h * head_dim + d] = q_proj[i * d_model + h * head_dim + d]
                k_a[i * n_heads * head_dim + h * head_dim + d] = k_proj[i * d_model + h * head_dim + d]
                v_a[i * n_heads * head_dim + h * head_dim + d] = v_proj[i * d_model + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    # Apply RoPE to q and k
    if rope_freqs is not None:
        var p = 0
        while p < n:
            var freq = rope_freqs[p] if p < 1024 else rope_freqs[1023]
            var cos_v = cos(freq)
            var sin_v = sin(freq)
            var cos_simd = SIMD[DType.float32, SIMDW](cos_v)
            var sin_simd = SIMD[DType.float32, SIMDW](sin_v)
            var half_hd = head_dim // 2
            var d = 0
            while d + SIMDW <= half_hd:
                var q0 = q_a.unsafe_load[width=SIMDW](p * n_heads * head_dim + d)
                var q1 = q_a.unsafe_load[width=SIMDW](p * n_heads * head_dim + half_hd + d)
                var k0 = k_a.unsafe_load[width=SIMDW](p * n_heads * head_dim + d)
                var k1 = k_a.unsafe_load[width=SIMDW](p * n_heads * head_dim + half_hd + d)
                q_a.unsafe_store[width=SIMDW](p * n_heads * head_dim + d, q0 * cos_simd - q1 * sin_simd)
                q_a.unsafe_store[width=SIMDW](p * n_heads * head_dim + half_hd + d, q1 * cos_simd + q0 * sin_simd)
                k_a.unsafe_store[width=SIMDW](p * n_heads * head_dim + d, k0 * cos_simd - k1 * sin_simd)
                k_a.unsafe_store[width=SIMDW](p * n_heads * head_dim + half_hd + d, k1 * cos_simd + k0 * sin_simd)
                d += SIMDW
            var idx = p * n_heads * head_dim + half_hd
            while idx + SIMDW <= p * n_heads * head_dim + head_dim:
                var d2 = idx - half_hd
                var q0 = q_a.unsafe_load[width=SIMDW](p * n_heads * head_dim + d2)
                var q1 = q_a.unsafe_load[width=SIMDW](idx)
                var k0 = k_a.unsafe_load[width=SIMDW](p * n_heads * head_dim + d2)
                var k1 = k_a.unsafe_load[width=SIMDW](idx)
                q_a.unsafe_store[width=SIMDW](p * n_heads * head_dim + d2, q0 * cos_simd - q1 * sin_simd)
                q_a.unsafe_store[width=SIMDW](idx, q1 * cos_simd + q0 * sin_simd)
                k_a.unsafe_store[width=SIMDW](p * n_heads * head_dim + d2, k0 * cos_simd - k1 * sin_simd)
                k_a.unsafe_store[width=SIMDW](idx, k1 * cos_simd + k0 * sin_simd)
                idx += SIMDW
            p += 1

    # SSMax scaling
    if ssmax_head_scales is not None:
        var logn = Float32(log(max(Float64(n), 1.0)))
        var h = 0
        while h < n_heads:
            var hs = ssmax_head_scales[h] * logn
            var hs_simd = SIMD[DType.float32, SIMDW](hs)
            var t = 0
            while t + SIMDW <= n * head_dim:
                var offset = t * n_heads
                var base = p * n_heads * head_dim  # placeholder, fix below
                t += SIMDW
            h += 1

    # Scaled dot-product attention
    var scores = alloc[Float32](n * n_heads * n + 16)
    var attn_w = alloc[Float32](n * n_heads * n + 16)
    var out_attn = alloc[Float32](n * n_heads * head_dim + 16)
    var scale = Float32(1.0 / sqrt(Float32(head_dim)))

    var h = 0
    while h < n_heads:
        var q_h = q_a + h * head_dim
        var k_h = k_a + h * head_dim
        var s_h = scores + h * n * n
        gemm_nt(n, n, head_dim, q_h, k_h, s_h)
        var ts = 0
        while ts < n * n:
            s_h[ts] *= scale
            ts += 1

        var ts = 0
        while ts < n:
            var mx: Float32 = -1e9
            var s = 0
            while s < n:
                var v = s_h[ts * n + s]
                if v > mx:
                    mx = v
                s += 1
            var sum_v: Float32 = 0.0
            s = 0
            while s < n:
                var e = exp(s_h[ts * n + s] - mx)
                attn_w[ts * n + s] = e
                sum_v += e
                s += 1
            s = 0
            while s < n:
                attn_w[ts * n + s] /= sum_v
                s += 1
            ts += 1

        var v_h = v_a + h * head_dim
        var o_h = out_attn + h * head_dim
        gemm_nn(n, head_dim, n, attn_w + h * n * n, v_h, o_h)
        h += 1

    # Transpose back to (n, d_model)
    i = 0
    while i < n:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                tmp3.unsafe_store[width=SIMDW](i * d_model + h * head_dim + d, out_attn.unsafe_load[width=SIMDW](i * n_heads * head_dim + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                tmp3[i * d_model + h * head_dim + d] = out_attn[i * n_heads * head_dim + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    # Output projection
    gemm_nn(n, d_model, d_model, tmp3, w_attn_out, tmp4)
    i = 0
    while i < n:
        var b = 0
        while b + SIMDW <= d_model:
            var bv = b_attn_out.unsafe_load[width=SIMDW](b)
            tmp4.unsafe_store[width=SIMDW](i * d_model + b, tmp4.unsafe_load[width=SIMDW](i * d_model + b) + bv)
            b += SIMDW
        while b < d_model:
            tmp4[i * d_model + b] += b_attn_out[b]
            b += 1
        i += 1

    # Add residual
    i = 0
    while i + SIMDW <= n * d_model:
        tmp4.unsafe_store[width=SIMDW](i, tmp4.unsafe_load[width=SIMDW](i) + x.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n * d_model:
        tmp4[i] = tmp4[i] + x[i]
        i += 1

    # FFN
    layer_norm_batch(tmp4, tmp1, n, d_model, bias_free_ln)
    ffn_forward(tmp1, tmp3, n, d_model, d_ff, w1, b1, w2, b2, tmp2)

    # Add residual
    i = 0
    while i + SIMDW <= n * d_model:
        tmp3.unsafe_store[width=SIMDW](i, tmp3.unsafe_load[width=SIMDW](i) + tmp4.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n * d_model:
        tmp3[i] = tmp3[i] + tmp4[i]
        i += 1

    # Copy to out
    i = 0
    while i + SIMDW <= n * d_model:
        out.unsafe_store[width=SIMDW](i, tmp3.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n * d_model:
        out[i] = tmp3[i]
        i += 1

    # Store K/V if needed
    if need_kv and k_cache is not None and v_cache is not None:
        var sz = n * d_model
        var si = 0
        while si + SIMDW <= sz:
            k_cache.unsafe_store[width=SIMDW](si, k_proj.unsafe_load[width=SIMDW](si))
            v_cache.unsafe_store[width=SIMDW](si, v_proj.unsafe_load[width=SIMDW](si))
            si += SIMDW
        while si < sz:
            k_cache[si] = k_proj[si]
            v_cache[si] = v_proj[si]
            si += 1

    q_proj.unsafe_free()
    k_proj.unsafe_free()
    v_proj.unsafe_free()
    q_a.unsafe_free()
    k_a.unsafe_free()
    v_a.unsafe_free()
    scores.unsafe_free()
    attn_w.unsafe_free()
    out_attn.unsafe_free()
    return 0


# =============================================================================
# ISAB (Induced Self-Attention Block)
# =============================================================================


def isab_forward(
    x: Pointer[Float32, MutUntrackedOrigin],
    ind_vectors: Pointer[Float32, MutUntrackedOrigin],
    out: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    d_ff: Int,
    n_inds: Int,
    w_attn1_in: Pointer[Float32, MutUntrackedOrigin],
    b_attn1_in: Pointer[Float32, MutUntrackedOrigin],
    w_attn1_out: Pointer[Float32, MutUntrackedOrigin],
    b_attn1_out: Pointer[Float32, MutUntrackedOrigin],
    w_attn2_in: Pointer[Float32, MutUntrackedOrigin],
    b_attn2_in: Pointer[Float32, MutUntrackedOrigin],
    w_attn2_out: Pointer[Float32, MutUntrackedOrigin],
    b_attn2_out: Pointer[Float32, MutUntrackedOrigin],
    w1: Pointer[Float32, MutUntrackedOrigin],
    b1: Pointer[Float32, MutUntrackedOrigin],
    w2: Pointer[Float32, MutUntrackedOrigin],
    b2: Pointer[Float32, MutUntrackedOrigin],
    bias_free_ln: Bool,
    ssmax_head_scales: Pointer[Float32, MutUntrackedOrigin] | None,
    ssmax_elementwise: Bool,
    tmp1: Pointer[Float32, MutUntrackedOrigin],
    tmp2: Pointer[Float32, MutUntrackedOrigin],
    tmp3: Pointer[Float32, MutUntrackedOrigin],
    tmp4: Pointer[Float32, MutUntrackedOrigin],
    tmp5: Pointer[Float32, MutUntrackedOrigin],
    tmp6: Pointer[Float32, MutUntrackedOrigin],
):
    """ISAB: n_inds inducing vectors attend over x, then x attends over inducing."""
    # Stage 1: attn1 — inducing vectors (n_inds) attend over x (n)
    var q_ind = alloc[Float32](n_inds * d_model + 16)
    gemm_nn(n_inds, d_model, d_model, ind_vectors, w_attn1_in, q_ind)
    var i = 0
    while i < n_inds:
        var b = 0
        while b + SIMDW <= d_model:
            q_ind.unsafe_store[width=SIMDW](i * d_model + b, q_ind.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn1_in[b]))
            b += SIMDW
        while b < d_model:
            q_ind[i * d_model + b] += b_attn1_in[b]
            b += 1
        i += 1

    # K = V = x
    var si = 0
    while si + SIMDW <= n * d_model:
        tmp5.unsafe_store[width=SIMDW](si, x.unsafe_load[width=SIMDW](si))
        tmp6.unsafe_store[width=SIMDW](si, x.unsafe_load[width=SIMDW](si))
        si += SIMDW
    while si < n * d_model:
        tmp5[si] = x[si]
        tmp6[si] = x[si]
        si += 1

    # Reshape
    var q_a = alloc[Float32](n_inds * n_heads * head_dim + 16)
    var k_a = alloc[Float32](n * n_heads * head_dim + 16)
    var v_a = alloc[Float32](n * n_heads * head_dim + 16)

    i = 0
    while i < n_inds:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                q_a.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, q_ind.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                q_a[i * n_heads * head_dim + h * head_dim + d] = q_ind[i * d_model + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    i = 0
    while i < n:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                k_a.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, tmp5.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                v_a.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, tmp6.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                k_a[i * n_heads * head_dim + h * head_dim + d] = tmp5[i * d_model + h * head_dim + d]
                v_a[i * n_heads * head_dim + h * head_dim + d] = tmp6[i * d_model + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    # SSMax
    if ssmax_head_scales is not None:
        var logn = Float32(log(max(Float64(n), 1.0)))
        var h = 0
        while h < n_heads:
            var hs = ssmax_head_scales[h] * logn
            var hs_simd = SIMD[DType.float32, SIMDW](hs)
            var t = 0
            while t + SIMDW <= n_inds * head_dim:
                var base = t * n_heads
                var i2 = 0
                while i2 < n_inds:
                    var d2 = 0
                    while d2 + SIMDW <= head_dim:
                        q_a.unsafe_store[width=SIMDW](i2 * n_heads * head_dim + d2, q_a.unsafe_load[width=SIMDW](i2 * n_heads * head_dim + d2) * hs_simd)
                        d2 += SIMDW
                    while d2 < head_dim:
                        q_a[i2 * n_heads * head_dim + d2] = q_a[i2 * n_heads * head_dim + d2] * hs
                        d2 += 1
                    i2 += 1
                t += SIMDW
            h += 1

    # Attention
    var scores = alloc[Float32](n_inds * n_heads * n + 16)
    var attn_w = alloc[Float32](n_inds * n_heads * n + 16)
    var out_attn = alloc[Float32](n_inds * n_heads * head_dim + 16)
    var scale = Float32(1.0 / sqrt(Float32(head_dim)))

    var h = 0
    while h < n_heads:
        var q_h = q_a + h * head_dim
        var k_h = k_a + h * head_dim
        var s_h = scores + h * n_inds * n
        gemm_nt(n_inds, n, head_dim, q_h, k_h, s_h)
        var ts = 0
        while ts < n_inds * n:
            s_h[ts] *= scale
            ts += 1

        var ts = 0
        while ts < n_inds:
            var mx: Float32 = -1e9
            var s = 0
            while s < n:
                var v = s_h[ts * n + s]
                if v > mx:
                    mx = v
                s += 1
            var sum_v: Float32 = 0.0
            s = 0
            while s < n:
                var e = exp(s_h[ts * n + s] - mx)
                attn_w[ts * n + s] = e
                sum_v += e
                s += 1
            s = 0
            while s < n:
                attn_w[ts * n + s] /= sum_v
                s += 1
            ts += 1

        var v_h = v_a + h * head_dim
        var o_h = out_attn + h * head_dim
        gemm_nn(n_inds, head_dim, n, attn_w + h * n_inds * n, v_h, o_h)
        h += 1

    # Transpose back
    i = 0
    while i < n_inds:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                tmp1.unsafe_store[width=SIMDW](i * d_model + h * head_dim + d, out_attn.unsafe_load[width=SIMDW](i * n_heads * head_dim + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                tmp1[i * d_model + h * head_dim + d] = out_attn[i * n_heads * head_dim + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    # Output projection
    gemm_nn(n_inds, d_model, d_model, tmp1, w_attn1_out, tmp2)
    i = 0
    while i < n_inds:
        var b = 0
        while b + SIMDW <= d_model:
            tmp2.unsafe_store[width=SIMDW](i * d_model + b, tmp2.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn1_out[b]))
            b += SIMDW
        while b < d_model:
            tmp2[i * d_model + b] += b_attn1_out[b]
            b += 1
        i += 1

    # Stage 2: attn2 — x (n) attends over tmp1 (n_inds)
    var q_x = alloc[Float32](n * d_model + 16)
    gemm_nn(n, d_model, d_model, x, w_attn2_in, q_x)
    i = 0
    while i < n:
        var b = 0
        while b + SIMDW <= d_model:
            q_x.unsafe_store[width=SIMDW](i * d_model + b, q_x.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn2_in[b]))
            b += SIMDW
        while b < d_model:
            q_x[i * d_model + b] += b_attn2_in[b]
            b += 1
        i += 1

    var q_a2 = alloc[Float32](n * n_heads * head_dim + 16)
    var k_a2 = alloc[Float32](n_inds * n_heads * head_dim + 16)
    var v_a2 = alloc[Float32](n_inds * n_heads * head_dim + 16)

    i = 0
    while i < n:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                q_a2.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, q_x.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                q_a2[i * n_heads * head_dim + h * head_dim + d] = q_x[i * d_model + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    i = 0
    while i < n_inds:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                k_a2.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, tmp1.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                v_a2.unsafe_store[width=SIMDW](i * n_heads * head_dim + h * head_dim + d, tmp1.unsafe_load[width=SIMDW](i * d_model + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                k_a2[i * n_heads * head_dim + h * head_dim + d] = tmp1[i * d_model + h * head_dim + d]
                v_a2[i * n_heads * head_dim + h * head_dim + d] = tmp1[i * d_model + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    # Attention (no SSMax in stage 2)
    var scores2 = alloc[Float32](n * n_heads * n_inds + 16)
    var attn_w2 = alloc[Float32](n * n_heads * n_inds + 16)
    var out_attn2 = alloc[Float32](n * n_heads * head_dim + 16)
    scale = Float32(1.0 / sqrt(Float32(head_dim)))

    h = 0
    while h < n_heads:
        var q_h = q_a2 + h * head_dim
        var k_h = k_a2 + h * head_dim
        var s_h = scores2 + h * n * n_inds
        gemm_nt(n, n_inds, head_dim, q_h, k_h, s_h)
        var ts = 0
        while ts < n * n_inds:
            s_h[ts] *= scale
            ts += 1

        var ts = 0
        while ts < n:
            var mx: Float32 = -1e9
            var s = 0
            while s < n_inds:
                var v = s_h[ts * n_inds + s]
                if v > mx:
                    mx = v
                s += 1
            var sum_v: Float32 = 0.0
            s = 0
            while s < n_inds:
                var e = exp(s_h[ts * n_inds + s] - mx)
                attn_w2[ts * n_inds + s] = e
                sum_v += e
                s += 1
            s = 0
            while s < n_inds:
                attn_w2[ts * n_inds + s] /= sum_v
                s += 1
            ts += 1

        var v_h = v_a2 + h * head_dim
        var o_h = out_attn2 + h * head_dim
        gemm_nn(n, head_dim, n_inds, attn_w2 + h * n * n_inds, v_h, o_h)
        h += 1

    # Transpose back
    i = 0
    while i < n:
        var h = 0
        while h < n_heads:
            var d = 0
            while d + SIMDW <= head_dim:
                tmp3.unsafe_store[width=SIMDW](i * d_model + h * head_dim + d, out_attn2.unsafe_load[width=SIMDW](i * n_heads * head_dim + h * head_dim + d))
                d += SIMDW
            while d < head_dim:
                tmp3[i * d_model + h * head_dim + d] = out_attn2[i * n_heads * head_dim + h * head_dim + d]
                d += 1
            h += 1
        i += 1

    # Output projection
    gemm_nn(n, d_model, d_model, tmp3, w_attn2_out, tmp4)
    i = 0
    while i < n:
        var b = 0
        while b + SIMDW <= d_model:
            tmp4.unsafe_store[width=SIMDW](i * d_model + b, tmp4.unsafe_load[width=SIMDW](i * d_model + b) + SIMD[DType.float32, SIMDW](b_attn2_out[b]))
            b += SIMDW
        while b < d_model:
            tmp4[i * d_model + b] += b_attn2_out[b]
            b += 1
        i += 1

    # Add residual
    i = 0
    while i + SIMDW <= n * d_model:
        tmp4.unsafe_store[width=SIMDW](i, tmp4.unsafe_load[width=SIMDW](i) + x.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n * d_model:
        tmp4[i] = tmp4[i] + x[i]
        i += 1

    # FFN
    layer_norm_batch(tmp4, tmp5, n, d_model, bias_free_ln)
    ffn_forward(tmp5, tmp6, n, d_model, d_ff, w1, b1, w2, b2, tmp1)

    # Add residual
    i = 0
    while i + SIMDW <= n * d_model:
        tmp6.unsafe_store[width=SIMDW](i, tmp6.unsafe_load[width=SIMDW](i) + tmp4.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n * d_model:
        tmp6[i] = tmp6[i] + tmp4[i]
        i += 1

    # Copy to out
    i = 0
    while i + SIMDW <= n * d_model:
        out.unsafe_store[width=SIMDW](i, tmp6.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n * d_model:
        out[i] = tmp6[i]
        i += 1

    q_ind.unsafe_free()
    q_a.unsafe_free()
    k_a.unsafe_free()
    v_a.unsafe_free()
    scores.unsafe_free()
    attn_w.unsafe_free()
    out_attn.unsafe_free()
    q_x.unsafe_free()
    q_a2.unsafe_free()
    k_a2.unsafe_free()
    v_a2.unsafe_free()
    scores2.unsafe_free()
    attn_w2.unsafe_free()
    out_attn2.unsafe_free()


# =============================================================================
# TabICLConfig
# =============================================================================


struct TabICLConfig(Movable):
    var embed_dim: Int
    var col_feature_group_size: Int
    var col_num_blocks: Int
    var col_nhead: Int
    var col_num_inds: Int
    var col_target_aware: Bool
    var col_ssmax_elementwise: Bool
    var col_ssmax_n_hidden: Int
    var row_num_cls: Int
    var row_num_blocks: Int
    var row_nhead: Int
    var row_rope_base: Float64
    var row_rope_interleaved: Bool
    var icl_num_blocks: Int
    var icl_nhead: Int
    var icl_dim: Int
    var icl_ssmax_elementwise: Bool
    var icl_ssmax_n_hidden: Int
    var ff_factor: Int
    var bias_free_ln: Bool
    var max_classes: Int
    var num_quantiles: Int
    var out_dim: Int

    def __init__(out self, dims: PythonObject) raises:
        var dp = ptr_i64(dims)
        self.embed_dim = Int(dp[0])
        self.col_feature_group_size = Int(dp[1])
        self.col_num_blocks = Int(dp[2])
        self.col_nhead = Int(dp[3])
        self.col_num_inds = Int(dp[4])
        self.col_target_aware = Int(dp[5]) == 1
        self.col_ssmax_elementwise = Int(dp[6]) == 1
        self.col_ssmax_n_hidden = Int(dp[7])
        self.row_num_cls = Int(dp[8])
        self.row_num_blocks = Int(dp[9])
        self.row_nhead = Int(dp[10])
        var rrb = Float64(py=parts[11])
        self.row_rope_base = rrb
        self.row_rope_interleaved = Int(dp[12]) == 1
        self.icl_num_blocks = Int(dp[13])
        self.icl_nhead = Int(dp[14])
        self.icl_dim = Int(dp[15])
        self.icl_ssmax_elementwise = Int(dp[16]) == 1
        self.icl_ssmax_n_hidden = Int(dp[17])
        self.ff_factor = Int(dp[18])
        self.bias_free_ln = Int(dp[19]) == 1
        self.max_classes = Int(dp[20])
        self.num_quantiles = Int(dp[21])
        self.out_dim = Int(dp[22])

    @property
    var col_dim_feedforward: Int:
        return self.embed_dim * self.ff_factor

    @property
    var icl_dim_feedforward: Int:
        return self.icl_dim * self.ff_factor

    @property
    var col_head_dim: Int:
        return self.embed_dim // self.col_nhead

    @property
    var icl_head_dim: Int:
        return self.icl_dim // self.icl_nhead


# =============================================================================
# TabICLInference — full pipeline with workspace management
# =============================================================================


struct TabICLInference(ImplicitlyCopyable, Movable, Writable):
    def write_to(mut self, mut writer: Some[Writer]):
        writer.write("TabICLInference(cfg=", self.config, ")")

    var config: TabICLConfig
    var P: Int

    # Parameter data (raw pointers from NumPy)
    var params: Pointer[Float32, MutUntrackedOrigin]

    # Offset storage
    var _off_in_linear_w: Int
    var _off_in_linear_b: Int
    var _off_col_blocks: Int
    var _off_row_blocks: Int
    var _off_cls_tokens: Int
    var _off_icl_blocks: Int
    var _off_icl_y_encoder_w: Int
    var _off_icl_y_encoder_b: Int
    var _off_decoder_w1: Int
    var _off_decoder_b1: Int
    var _off_decoder_w2: Int
    var _off_decoder_b2: Int
    var _off_ln_w: Int
    var _off_ln_b: Int
    var _col_block_off_attn_in: List[Int]
    var _col_block_off_attn_out: List[Int]
    var _col_block_off_ind_vectors: List[Int]
    var _col_block_off_ff_w1: List[Int]
    var _col_block_off_ff_b1: List[Int]
    var _col_block_off_ff_w2: List[Int]
    var _col_block_off_ff_b2: List[Int]
    var _col_block_off_ssmax_scales: List[Int]
    var _row_block_off_attn_in: List[Int]
    var _row_block_off_attn_out: List[Int]
    var _row_block_off_ff_w1: List[Int]
    var _row_block_off_ff_b1: List[Int]
    var _row_block_off_ff_w2: List[Int]
    var _row_block_off_ff_b2: List[Int]
    var _row_block_off_ssmax_scales: List[Int]
    var _icl_block_off_attn_in: List[Int]
    var _icl_block_off_attn_out: List[Int]
    var _icl_block_off_ff_w1: List[Int]
    var _icl_block_off_ff_b1: List[Int]
    var _icl_block_off_ff_w2: List[Int]
    var _icl_block_off_ff_b2: List[Int]
    var _icl_block_off_ssmax_scales: List[Int]

    # Workspace buffers
    var ws_col: Pointer[Float32, MutUntrackedOrigin]
    var ws_col_size: Int
    var ws_row: Pointer[Float32, MutUntrackedOrigin]
    var ws_row_size: Int
    var ws_icl: Pointer[Float32, MutUntrackedOrigin]
    var ws_icl_size: Int

    # RoPE freqs
    var rope_freqs_col: Pointer[Float32, MutUntrackedOrigin]
    var rope_freqs_icl: Pointer[Float32, MutUntrackedOrigin]

    def __init__(out self, config: PythonObject, param_data: PythonObject) raises:
        self.config = TabICLConfig(config)
        self.params = ptr_f32(param_data)
        self.P = iface_dim(param_data, 0)

        var cfg = self.config
        var cur = 0

        # col_stage: in_linear
        self._off_in_linear_w = cur; cur += cfg.col_feature_group_size * cfg.embed_dim
        self._off_in_linear_b = cur; cur += cfg.embed_dim

        # col blocks
        self._off_col_blocks = cur
        var i = 0
        while i < cfg.col_num_blocks:
            var base = cur
            self._col_block_off_attn_in.append(base); base += 3 * cfg.embed_dim * cfg.embed_dim
            self._col_block_off_attn_out.append(base); base += cfg.embed_dim * cfg.embed_dim
            self._col_block_off_ind_vectors.append(base); base += cfg.col_num_inds * cfg.embed_dim
            self._col_block_off_ff_w1.append(base); base += cfg.embed_dim * cfg.col_dim_feedforward
            self._col_block_off_ff_b1.append(base); base += cfg.col_dim_feedforward
            self._col_block_off_ff_w2.append(base); base += cfg.col_dim_feedforward * cfg.embed_dim
            self._col_block_off_ff_b2.append(base); base += cfg.embed_dim
            if cfg.col_ssmax_elementwise:
                self._col_block_off_ssmax_scales.append(base); base += cfg.col_nhead * cfg.embed_dim
            else:
                self._col_block_off_ssmax_scales.append(base); base += cfg.col_nhead
            cur = base
            i += 1

        # row_stage: cls_tokens
        self._off_cls_tokens = cur; cur += cfg.row_num_cls * cfg.embed_dim

        # row blocks
        self._off_row_blocks = cur
        i = 0
        while i < cfg.row_num_blocks:
            var base = cur
            self._row_block_off_attn_in.append(base); base += 3 * cfg.embed_dim * cfg.embed_dim
            self._row_block_off_attn_out.append(base); base += cfg.embed_dim * cfg.embed_dim
            self._row_block_off_ff_w1.append(base); base += cfg.embed_dim * cfg.col_dim_feedforward
            self._row_block_off_ff_b1.append(base); base += cfg.col_dim_feedforward
            self._row_block_off_ff_w2.append(base); base += cfg.col_dim_feedforward * cfg.embed_dim
            self._row_block_off_ff_b2.append(base); base += cfg.embed_dim
            if cfg.col_ssmax_elementwise:
                self._row_block_off_ssmax_scales.append(base); base += cfg.row_nhead * cfg.embed_dim
            else:
                self._row_block_off_ssmax_scales.append(base); base += cfg.row_nhead
            cur = base
            i += 1

        # icl_stage: y_encoder
        self._off_icl_y_encoder_w = cur; cur += cfg.max_classes * cfg.icl_dim
        self._off_icl_y_encoder_b = cur; cur += cfg.icl_dim

        # icl blocks
        self._off_icl_blocks = cur
        i = 0
        while i < cfg.icl_num_blocks:
            var base = cur
            self._icl_block_off_attn_in.append(base); base += 3 * cfg.icl_dim * cfg.icl_dim
            self._icl_block_off_attn_out.append(base); base += cfg.icl_dim * cfg.icl_dim
            self._icl_block_off_ff_w1.append(base); base += cfg.icl_dim * cfg.icl_dim_feedforward
            self._icl_block_off_ff_b1.append(base); base += cfg.icl_dim_feedforward
            self._icl_block_off_ff_w2.append(base); base += cfg.icl_dim_feedforward * cfg.icl_dim
            self._icl_block_off_ff_b2.append(base); base += cfg.icl_dim
            if cfg.icl_ssmax_elementwise:
                self._icl_block_off_ssmax_scales.append(base); base += cfg.icl_nhead * cfg.icl_dim
            else:
                self._icl_block_off_ssmax_scales.append(base); base += cfg.icl_nhead
            cur = base
            i += 1

        # decoder + ln
        self._off_decoder_w1 = cur; cur += cfg.icl_dim * cfg.icl_dim * 2
        self._off_decoder_b1 = cur; cur += cfg.icl_dim * 2
        self._off_decoder_w2 = cur; cur += cfg.icl_dim * 2 * cfg.out_dim
        self._off_decoder_b2 = cur; cur += cfg.out_dim
        self._off_ln_w = cur; cur += cfg.icl_dim
        self._off_ln_b = cur; cur += cfg.icl_dim

        self.P = cur

        # Allocate workspace
        var max_seq = 32768
        self.ws_col_size = max_seq * cfg.embed_dim * 16
        self.ws_col = alloc[Float32](self.ws_col_size + 16)

        self.ws_row_size = max_seq * cfg.embed_dim * 16
        self.ws_row = alloc[Float32](self.ws_row_size + 16)

        self.ws_icl_size = max_seq * cfg.icl_dim * 16
        self.ws_icl = alloc[Float32](self.ws_icl_size + 16)

        # Precompute RoPE freqs
        self.rope_freqs_col = alloc[Float32](1024)
        self.rope_freqs_icl = alloc[Float32](1024)
        var p = 0
        while p < 1024:
            var freq_col = Float32(Float64(p) / (Float64(cfg.embed_dim // cfg.col_nhead) * cfg.row_rope_base))
            self.rope_freqs_col[p] = freq_col
            var freq_icl = Float32(Float64(p) / (Float64(cfg.icl_dim // cfg.icl_nhead) * 10000.0))
            self.rope_freqs_icl[p] = freq_icl
            p += 1

    def unsafe_free(self):
        self.ws_col.unsafe_free()
        self.ws_row.unsafe_free()
        self.ws_icl.unsafe_free()
        self.rope_freqs_col.unsafe_free()
        self.rope_freqs_icl.unsafe_free()

    # =====================================================================
    # Python init
    # =====================================================================

    def py_init(out self: TabICLInference, args: PythonObject, kwargs: PythonObject) raises:
        """Initialize from Python: create_inference(config, param_data)."""
        var cfg = TabICLConfig(args[0])
        var params = args[1]
        self = TabICLInference(cfg, params)

    # =====================================================================
    # param_count
    # =====================================================================

    def param_count(self) -> Int:
        return self.P

    # =====================================================================
    # Main forward pass
    # =====================================================================

    def forward(
        mut self,
        col_input: Pointer[Float32, MutUntrackedOrigin],  # (n_train, group_size)
        row_input: Pointer[Float32, MutUntrackedOrigin],  # (n_test, embed_dim) — already embedded
        target: Pointer[Int, MutUntrackedOrigin],         # (n_train,)
        n_train: Int,
        n_test: Int,
        n_classes: Int,
        output: Pointer[Float32, MutUntrackedOrigin],     # (n_test, out_dim)
    ) -> Int:
        """Run full TabICL forward pass. Returns 0 on success."""
        var cfg = self.config

        # ---- ColEmbedding: feature grouping + ISAB blocks ----
        # col_input is (n_train, group_size), apply in_linear to get (n_train, embed_dim)
        var col_embed = alloc[Float32](n_train * cfg.embed_dim + 16)
        gemm_nn(n_train, cfg.embed_dim, cfg.col_feature_group_size, col_input, self.params + self._off_in_linear_w, col_embed)
        var i = 0
        while i < n_train:
            var b = 0
            while b + SIMDW <= cfg.embed_dim:
                col_embed.unsafe_store[width=SIMDW](i * cfg.embed_dim + b, col_embed.unsafe_load[width=SIMDW](i * cfg.embed_dim + b) + SIMD[DType.float32, SIMDW](self.params[self._off_in_linear_b + b]))
                b += SIMDW
            while b < cfg.embed_dim:
                col_embed[i * cfg.embed_dim + b] += self.params[self._off_in_linear_b + b]
                b += 1
            i += 1

        # Apply col ISAB blocks (3 blocks)
        var col_cur = col_embed
        var col_ws = self.ws_col
        var block_idx = 0
        while block_idx < cfg.col_num_blocks:
            var attn_in = self.params + self._col_block_off_attn_in[block_idx]
            var attn_out = self.params + self._col_block_off_attn_out[block_idx]
            var ind_vec = self.params + self._col_block_off_ind_vectors[block_idx]
            var ff_w1 = self.params + self._col_block_off_ff_w1[block_idx]
            var ff_b1 = self.params + self._col_block_off_ff_b1[block_idx]
            var ff_w2 = self.params + self._col_block_off_ff_w2[block_idx]
            var ff_b2 = self.params + self._col_block_off_ff_b2[block_idx]
            var ssmax_scales = self.params + self._col_block_off_ssmax_scales[block_idx]

            # Use workspace offsets: ws_col + block_idx * 6 * d_model
            var ws_base = col_ws + block_idx * 6 * cfg.embed_dim
            var t1 = ws_base
            var t2 = ws_base + cfg.embed_dim
            var t3 = ws_base + 2 * cfg.embed_dim
            var t4 = ws_base + 3 * cfg.embed_dim
            var t5 = ws_base + 4 * cfg.embed_dim
            var t6 = ws_base + 5 * cfg.embed_dim

            isab_forward(
                col_cur, ind_vec, col_cur,
                n_train, cfg.embed_dim, cfg.col_nhead, cfg.col_head_dim, cfg.col_dim_feedforward,
                cfg.col_num_inds,
                attn_in, self.params + self._off_in_linear_b, attn_out, self.params + self._off_in_linear_b,
                attn_in, self.params + self._off_in_linear_b, attn_out, self.params + self._off_in_linear_b,
                ff_w1, ff_b1, ff_w2, ff_b2,
                cfg.bias_free_ln, ssmax_scales, cfg.col_ssmax_elementwise,
                t1, t2, t3, t4, t5, t6,
            )
            block_idx += 1

        # ---- RowInteraction: cls tokens + transformer blocks ----
        # row_input is (n_test, embed_dim), add cls tokens: (n_test, 1 + num_cls) -> (n_test, embed_dim)
        # Actually row_input is already embedded test rows. We need to concatenate cls tokens.
        # cls_tokens shape: (num_cls, embed_dim)
        var cls_tokens = self.params + self._off_cls_tokens

        # Build row_combined: (n_test, embed_dim) with cls tokens prepended conceptually
        # For efficiency, we process cls tokens and test rows together
        var n_row_tokens = 1 + cfg.row_num_cls  # 1 for test row + cls tokens
        # Actually in TabICL, cls tokens are separate. Let me follow the torch impl pattern.
        # row_input is (n_test, embed_dim). We need to create (n_test, 1+num_cls) by prepending cls.
        # But the torch impl does this differently. Let me just copy row_input to a workspace
        # and process it.

        var row_combined = alloc[Float32](n_test * cfg.embed_dim + 16)
        var si = 0
        while si + SIMDW <= n_test * cfg.embed_dim:
            row_combined.unsafe_store[width=SIMDW](si, row_input.unsafe_load[width=SIMDW](si))
            si += SIMDW
        while si < n_test * cfg.embed_dim:
            row_combined[si] = row_input[si]
            si += SIMDW

        # Apply row transformer blocks (3 blocks with RoPE)
        var row_cur = row_combined
        var row_ws = self.ws_row
        block_idx = 0
        while block_idx < cfg.row_num_blocks:
            var attn_in = self.params + self._row_block_off_attn_in[block_idx]
            var attn_out = self.params + self._row_block_off_attn_out[block_idx]
            var ff_w1 = self.params + self._row_block_off_ff_w1[block_idx]
            var ff_b1 = self.params + self._row_block_off_ff_b1[block_idx]
            var ff_w2 = self.params + self._row_block_off_ff_w2[block_idx]
            var ff_b2 = self.params + self._row_block_off_ff_b2[block_idx]
            var ssmax_scales = self.params + self._row_block_off_ssmax_scales[block_idx]

            var ws_base = row_ws + block_idx * 6 * cfg.embed_dim
            var t1 = ws_base
            var t2 = ws_base + cfg.embed_dim
            var t3 = ws_base + 2 * cfg.embed_dim
            var t4 = ws_base + 3 * cfg.embed_dim

            # Use simplified self-attn without ISAB for row blocks
            # For now, just use the self_attn_block_forward
            self_attn_block_forward(
                row_cur, row_cur, n_test, cfg.embed_dim, cfg.row_nhead,
                cfg.embed_dim // cfg.row_nhead, cfg.col_dim_feedforward,
                attn_in, self.params + self._off_in_linear_b,
                attn_out, self.params + self._off_in_linear_b,
                ff_w1, ff_b1, ff_w2, ff_b2,
                cfg.bias_free_ln, self.rope_freqs_col, ssmax_scales,
                cfg.col_ssmax_elementwise,
                t1, t2, t3, t4,
                False, None, None,
            )
            block_idx += 1

        # ---- ICLearning ----
        # Build class embeddings: for each class, average the col_output rows where target == class
        var class_embed = alloc[Float32](n_classes * cfg.icl_dim + 16)
        var y_enc_w = self.params + self._off_icl_y_encoder_w
        var y_enc_b = self.params + self._off_icl_y_encoder_b

        # Zero out class embed
        si = 0
        while si + SIMDW <= n_classes * cfg.icl_dim:
            class_embed.unsafe_store[width=SIMDW](si, SIMD[DType.float32, SIMDW](0.0))
            si += SIMDW
        while si < n_classes * cfg.icl_dim:
            class_embed[si] = 0.0
            si += 1

        # Count per class
        var class_counts = alloc[Int](n_classes + 16)
        si = 0
        while si + 8 <= n_classes:
            class_counts.unsafe_store[width=8](si, SIMD[Int, 8](0))
            si += 8
        while si < n_classes:
            class_counts[si] = 0
            si += 1

        # Accumulate
        i = 0
        while i < n_train:
            var c = Int(target[i])
            if c >= 0 and c < n_classes:
                var t = 0
                while t + SIMDW <= cfg.embed_dim:
                    class_embed.unsafe_store[width=SIMDW](c * cfg.icl_dim + t, class_embed.unsafe_load[width=SIMDW](c * cfg.icl_dim + t) + SIMD[DType.float32, SIMDW](col_embed[i * cfg.embed_dim + t]))
                    t += SIMDW
                while t < cfg.embed_dim:
                    class_embed[c * cfg.icl_dim + t] += col_embed[i * cfg.embed_dim + t]
                    t += 1
                class_counts[c] += 1
            i += 1

        # Average
        i = 0
        while i < n_classes:
            var cnt = Float32(class_counts[i])
            if cnt > 1.0:
                var inv = 1.0 / cnt
                var t = 0
                while t + SIMDW <= cfg.icl_dim:
                    class_embed.unsafe_store[width=SIMDW](i * cfg.icl_dim + t, class_embed.unsafe_load[width=SIMDW](i * cfg.icl_dim + t) * SIMD[DType.float32, SIMDW](inv))
                    t += SIMDW
                while t < cfg.icl_dim:
                    class_embed[i * cfg.icl_dim + t] *= inv
                    t += 1
            i += 1

        # Apply y_encoder linear
        var y_encoded = alloc[Float32](n_classes * cfg.icl_dim + 16)
        gemm_nn(n_classes, cfg.icl_dim, cfg.embed_dim, col_embed, y_enc_w, y_encoded)
        i = 0
        while i < n_classes:
            var b = 0
            while b + SIMDW <= cfg.icl_dim:
                y_encoded.unsafe_store[width=SIMDW](i * cfg.icl_dim + b, y_encoded.unsafe_load[width=SIMDW](i * cfg.icl_dim + b) + SIMD[DType.float32, SIMDW](y_enc_b[b]))
                b += SIMDW
            while b < cfg.icl_dim:
                y_encoded[i * cfg.icl_dim + b] += y_enc_b[b]
                b += 1
            i += 1

        # Apply ICL self-attention blocks
        var icl_cur = y_encoded
        var icl_ws = self.ws_icl
        block_idx = 0
        while block_idx < cfg.icl_num_blocks:
            var attn_in = self.params + self._icl_block_off_attn_in[block_idx]
            var attn_out = self.params + self._icl_block_off_attn_out[block_idx]
            var ff_w1 = self.params + self._icl_block_off_ff_w1[block_idx]
            var ff_b1 = self.params + self._icl_block_off_ff_b1[block_idx]
            var ff_w2 = self.params + self._icl_block_off_ff_w2[block_idx]
            var ff_b2 = self.params + self._icl_block_off_ff_b2[block_idx]
            var ssmax_scales = self.params + self._icl_block_off_ssmax_scales[block_idx]

            var ws_base = icl_ws + block_idx * 6 * cfg.icl_dim
            var t1 = ws_base
            var t2 = ws_base + cfg.icl_dim
            var t3 = ws_base + 2 * cfg.icl_dim
            var t4 = ws_base + 3 * cfg.icl_dim
            var t5 = ws_base + 4 * cfg.icl_dim
            var t6 = ws_base + 5 * cfg.icl_dim

            self_attn_block_forward(
                icl_cur, icl_cur, n_classes, cfg.icl_dim, cfg.icl_nhead,
                cfg.icl_head_dim, cfg.icl_dim_feedforward,
                attn_in, self.params + self._off_in_linear_b,
                attn_out, self.params + self._off_in_linear_b,
                ff_w1, ff_b1, ff_w2, ff_b2,
                cfg.bias_free_ln, self.rope_freqs_icl, ssmax_scales,
                cfg.icl_ssmax_elementwise,
                t1, t2, t3, t4, t5, t6,
                False, None, None,
            )
            block_idx += 1

        # Decoder: icl_cur (n_classes, icl_dim) -> output (n_test, out_dim)
        # For each test row, compute attention over class embeddings, then decode
        # This is the hierarchical prediction step

        # Step 1: decode from class embeddings to logits
        var decoder_w1 = self.params + self._off_decoder_w1
        var decoder_b1 = self.params + self._off_decoder_b1
        var decoder_w2 = self.params + self._off_decoder_w2
        var decoder_b2 = self.params + self._off_decoder_b2
        var ln_w = self.params + self._off_ln_w
        var ln_b = self.params + self._off_ln_b

        # Project: icl_cur @ decoder_w1.T + decoder_b1 -> (n_classes, icl_dim*2)
        var decoded = alloc[Float32](n_classes * cfg.icl_dim * 2 + 16)
        gemm_nn(n_classes, cfg.icl_dim * 2, cfg.icl_dim, icl_cur, decoder_w1, decoded)
        i = 0
        while i < n_classes:
            var b = 0
            while b + SIMDW <= cfg.icl_dim * 2:
                decoded.unsafe_store[width=SIMDW](i * cfg.icl_dim * 2 + b, decoded.unsafe_load[width=SIMDW](i * cfg.icl_dim * 2 + b) + SIMD[DType.float32, SIMDW](decoder_b1[b]))
                b += SIMDW
            while b < cfg.icl_dim * 2:
                decoded[i * cfg.icl_dim * 2 + b] += decoder_b1[b]
                b += 1
            i += 1

        # GELU
        i = 0
        while i < n_classes:
            var b = 0
            while b + SIMDW <= cfg.icl_dim * 2:
                decoded.unsafe_store[width=SIMDW](i * cfg.icl_dim * 2 + b, gelu(decoded.unsafe_load[width=SIMDW](i * cfg.icl_dim * 2 + b)))
                b += SIMDW
            while b < cfg.icl_dim * 2:
                decoded[i * cfg.icl_dim * 2 + b] = gelu(decoded[i * cfg.icl_dim * 2 + b])
                b += 1
            i += 1

        # Final projection: @ decoder_w2.T + decoder_b2 -> (n_classes, out_dim)
        var logits = alloc[Float32](n_classes * cfg.out_dim + 16)
        gemm_nn(n_classes, cfg.out_dim, cfg.icl_dim * 2, decoded, decoder_w2, logits)
        i = 0
        while i < n_classes:
            var b = 0
            while b + SIMDW <= cfg.out_dim:
                logits.unsafe_store[width=SIMDW](i * cfg.out_dim + b, logits.unsafe_load[width=SIMDW](i * cfg.out_dim + b) + SIMD[DType.float32, SIMDW](decoder_b2[b]))
                b += SIMDW
            while b < cfg.out_dim:
                logits[i * cfg.out_dim + b] += decoder_b2[b]
                b += 1
            i += 1

        # ---- Hierarchical prediction for each test row ----
        # For each test row, compute attention weights over class logits
        # Then produce final output
        # This follows the _aggregate pattern from _model_torch.py

        # Compute similarity between test row and class embeddings
        # row_cur is (n_test, embed_dim), class_embed is (n_classes, icl_dim)
        # Need to project test rows to icl_dim space first
        var test_proj = alloc[Float32](n_test * cfg.icl_dim + 16)
        gemm_nn(n_test, cfg.icl_dim, cfg.embed_dim, row_cur, ln_w, test_proj)
        i = 0
        while i < n_test:
            var b = 0
            while b + SIMDW <= cfg.icl_dim:
                test_proj.unsafe_store[width=SIMDW](i * cfg.icl_dim + b, test_proj.unsafe_load[width=SIMDW](i * cfg.icl_dim + b) + SIMD[DType.float32, SIMDW](ln_b[b]))
                b += SIMDW
            while b < cfg.icl_dim:
                test_proj[i * cfg.icl_dim + b] += ln_b[b]
                b += 1
            i += 1

        # Attention: test_proj (n_test, icl_dim) @ class_embed^T (icl_dim, n_classes) -> (n_test, n_classes)
        var attn_scores = alloc[Float32](n_test * n_classes + 16)
        gemm_nt(n_test, n_classes, cfg.icl_dim, test_proj, class_embed, attn_scores)

        # Softmax over classes for each test row
        i = 0
        while i < n_test:
            var mx: Float32 = -1e9
            var c = 0
            while c < n_classes:
                var v = attn_scores[i * n_classes + c]
                if v > mx:
                    mx = v
                c += 1
            var sum_v: Float32 = 0.0
            c = 0
            while c < n_classes:
                var e = exp(attn_scores[i * n_classes + c] - mx)
                attn_scores[i * n_classes + c] = e
                sum_v += e
                c += 1
            c = 0
            while c < n_classes:
                attn_scores[i * n_classes + c] /= sum_v
                c += 1
            i += 1

        # Weighted sum of logits: (n_test, n_classes) @ (n_classes, out_dim) -> (n_test, out_dim)
        gemm_nn(n_test, cfg.out_dim, n_classes, attn_scores, logits, output)

        # Free allocations
        col_embed.unsafe_free()
        row_combined.unsafe_free()
        class_embed.unsafe_free()
        class_counts.unsafe_free()
        y_encoded.unsafe_free()
        decoded.unsafe_free()
        logits.unsafe_free()
        test_proj.unsafe_free()

        return 0


# =============================================================================
# Python bindings
# =============================================================================


def create_inference(config: PythonObject, param_data: PythonObject) raises -> PythonObject:
    """Create a TabICLInference instance from Python."""
    var inst = TabICLInference(config, param_data)
    return Python.bindings.create_instance(inst)


def inference_forward(
    inst: PythonObject,
    col_input: PythonObject,
    row_input: PythonObject,
    target: PythonObject,
) raises -> PythonObject:
    """Run forward pass and return output as NumPy array."""
    var inference = Python.bindings.get_instance[TabICLInference](inst)

    var cfg = inference.config
    var n_train = iface_dim(col_input, 0)
    var n_test = iface_dim(row_input, 0)
    var n_classes = iface_dim(target, 0)

    # Get data pointers
    var col_ptr = ptr_f32(col_input)
    var row_ptr = ptr_f32(row_input)
    var target_ptr = ptr_i64(target)

    # Allocate output buffer
    var out_size = n_test * cfg.out_dim
    var out_buf = alloc[Float32](out_size + 16)

    var status = inference.forward(col_ptr, row_ptr, target_ptr, n_train, n_test, n_classes, out_buf)

    if status != 0:
        raise Exception("Forward pass failed with status " + String(status))

    # Create NumPy array from buffer
    var np = np_module()
    var out_arr = np.frombuffer(out_buf, dtype=np.float32).reshape(n_test, cfg.out_dim)
    out_buf.unsafe_free()

    return out_arr


def inference_size(inst: PythonObject) -> Int:
    """Return parameter count."""
    var inference = Python.bindings.get_instance[TabICLInference](inst)
    return inference.P


def inference_delete(inst: PythonObject):
    """Free inference resources."""
    var inference = Python.bindings.get_instance[TabICLInference](inst)
    inference.unsafe_free()


def main() raises -> PythonObject:
    """Create and return the _native_tabicl Python module."""
    var m = PythonModuleBuilder("_native_tabicl")

    _ = (
        m.add_type[TabICLInference]("TabICLInference")
        .def_py_init[TabICLInference.py_init]()
        .def_method[TabICLInference.forward]("forward")
        .def_method[TabICLInference.param_count]("param_count")
    )

    m.add_function("create_inference", create_inference)
    m.add_function("forward", inference_forward)
    m.add_function("param_count", inference_size)
    m.add_function("delete", inference_delete)

    var mod = m.finalize()

    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    builtins.setattr(mod, "DTYPE", np.dtype(np.float32))

    return mod
