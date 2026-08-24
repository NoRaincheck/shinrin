"""Layer primitives for the native TabICL kernels.

Internal module consumed by ``shinrin._tabicl_kernels``. Contains LayerNorm,
the FFN, self-attention / ISAB blocks and the per-block parameter stride
helper. Code moved verbatim from ``_tabicl_kernels.mojo``.
"""

from std.math import cos, exp, max, sin, sqrt, tanh
from std.memory import alloc

from shinrin._tk_core import (
    SIMDW,
    P_U8,
    GemmWorkerFn,
    fast_log,
    gemm_nn,
    gemm_nn_rows,
    gemm_nt,
    gemm_nt_rows,
    gelu8,
    gelu_scalar,
    hardware_threads,
    run_partitioned_range,
)


# =============================================================================
# SSMax variants (mirror of torch SSMax / SSMaxMLP / QASSMaxMLP)
# =============================================================================
# Kinds match shinrin._tabicl._mojo_layout.SSMAX_*:
#   0 none, 1 scales, 2 mlp, 3 mlp-elementwise, 4 qassmax-mlp,
#   5 qassmax-mlp-elementwise.
#
# Packed section layout (see _SpecBuilder.ssmax):
#   kind 1: scales (n_heads,)
#   kinds 2/3: mlp.0.weight (hidden, 1) + mlp.0.bias (hidden,) +
#              mlp.2.weight (base_out, hidden) + mlp.2.bias (base_out,)
#   kinds 4/5: base_mlp as above + query_mlp.0.weight (hidden, head_dim) +
#              query_mlp.0.bias (hidden,) + query_mlp.2.weight
#              (query_out, hidden) + query_mlp.2.bias (query_out,)
# where base_out = n_heads * head_dim if elementwise else n_heads and
# query_out = head_dim if elementwise else 1.


def ssmax_section_size(kind: Int, n_heads: Int, head_dim: Int, hidden: Int) -> Int:
    """Number of packed float32 values the ssmax section needs (0 if absent)."""
    if kind == 0:
        return 0
    if kind == 1:
        return n_heads
    var elementwise = kind == 3 or kind == 5
    var base_out = n_heads * head_dim if elementwise else n_heads
    var size = hidden + hidden + base_out * hidden + base_out
    if kind >= 4:
        var query_out = head_dim if elementwise else 1
        size += head_dim * hidden + hidden + query_out * hidden + query_out
    return size


@always_inline
def _ssmax_base_scales(
    dst: Pointer[Float32, MutUntrackedOrigin],
    w0: Pointer[Float32, MutUntrackedOrigin],
    b0: Pointer[Float32, MutUntrackedOrigin],
    w2: Pointer[Float32, MutUntrackedOrigin],
    b2: Pointer[Float32, MutUntrackedOrigin],
    logn: Float32,
    hidden: Int,
    out_len: Int,
):
    """scales = mlp.2(GELU(mlp.0(logn))); mlp.0 weight has in_features=1."""
    var hbuf = alloc[Float32](hidden + 16)
    var j = 0
    while j < hidden:
        hbuf[j] = gelu_scalar(w0[j] * logn + b0[j])
        j += 1
    var k = 0
    while k < out_len:
        var acc: Float32 = 0.0
        j = 0
        while j < hidden:
            acc += w2[k * hidden + j] * hbuf[j]
            j += 1
        dst[k] = acc + b2[k]
        k += 1
    hbuf.unsafe_free()


def ssmax_apply(
    mut q: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    n_keys: Int,
    n_heads: Int,
    head_dim: Int,
    kind: Int,
    ssmax: Pointer[Float32, MutUntrackedOrigin],
    hidden: Int,
):
    """Apply SSMax scaling to ``q`` ((q_rows, n_heads*head_dim)) in place.

    ``n_keys`` is the attention source length (torch ``src_len``): the
    scale always uses log(n_keys) regardless of how many query rows there
    are, matching ``attn.ssmax_layer(q, src_len)`` in the reference.

    ``q`` uses the head-major blocked layout produced by
    ``attention_block_forward``: element (h, r, d) lives at
    ``h * q_rows * head_dim + r * head_dim + d``, matching torch's
    ``(..., n_heads, seq, head_dim)`` tensor after ``transpose(-3, -2)``.
    """
    if kind == 0:
        return
    var logn = fast_log(Float32(max(n_keys, 1)))

    if kind == 1:
        var h = 0
        while h < n_heads:
            var s = ssmax[h] * logn
            var r = 0
            while r < q_rows:
                _scale_inplace(q + h * q_rows * head_dim + r * head_dim, head_dim, s)
                r += 1
            h += 1
        return

    var elementwise = kind == 3 or kind == 5
    var base_out = n_heads * head_dim if elementwise else n_heads
    var cur = ssmax
    var w0 = cur
    cur += hidden
    var b0 = cur
    cur += hidden
    var w2 = cur
    cur += base_out * hidden
    var b2 = cur
    cur += base_out
    var scales = alloc[Float32](base_out + 16)
    _ssmax_base_scales(scales, w0, b0, w2, b2, logn, hidden, base_out)

    if kind <= 3:
        # SSMaxMLP: static per-head (or per-head-dim) multipliers.
        var h = 0
        while h < n_heads:
            var r = 0
            while r < q_rows:
                var base = h * q_rows * head_dim + r * head_dim
                if elementwise:
                    var d = 0
                    while d + SIMDW <= head_dim:
                        q.unsafe_store[width=SIMDW](
                            base + d,
                            q.unsafe_load[width=SIMDW](base + d)
                            * scales.unsafe_load[width=SIMDW](h * head_dim + d),
                        )
                        d += SIMDW
                    while d < head_dim:
                        q[base + d] *= scales[h * head_dim + d]
                        d += 1
                else:
                    _scale_inplace(q + base, head_dim, scales[h])
                r += 1
            h += 1
        scales.unsafe_free()
        return

    # QASSMaxMLP: modulation = 1 + tanh(query_mlp(q)), applied per token.
    var query_out = head_dim if elementwise else 1
    var qw0 = cur
    cur += head_dim * hidden
    var qb0 = cur
    cur += hidden
    var qw2 = cur
    cur += query_out * hidden
    var qb2 = cur
    var qh = alloc[Float32](hidden + 16)
    var mod_buf = alloc[Float32](head_dim + 16)
    var r = 0
    while r < q_rows:
        var h = 0
        while h < n_heads:
            var qp = q + h * q_rows * head_dim + r * head_dim
            var j = 0
            while j < hidden:
                qh[j] = gelu_scalar(_dot(qw0 + j * head_dim, qp, head_dim) + qb0[j])
                j += 1
            if elementwise:
                var d = 0
                while d < head_dim:
                    mod_buf[d] = (
                        1.0 + tanh(_dot(qw2 + d * hidden, qh, hidden) + qb2[d])
                    )
                    d += 1
            else:
                var acc3 = _dot(qw2, qh, hidden)
                var m = 1.0 + tanh(acc3 + qb2[0])
                var dv = SIMD[DType.float32, SIMDW](m)
                var d = 0
                while d + SIMDW <= head_dim:
                    mod_buf.unsafe_store[width=SIMDW](d, dv)
                    d += SIMDW
                while d < head_dim:
                    mod_buf[d] = m
                    d += 1
            var d2 = 0
            if elementwise:
                while d2 + SIMDW <= head_dim:
                    qp.unsafe_store[width=SIMDW](
                        d2,
                        qp.unsafe_load[width=SIMDW](d2)
                        * scales.unsafe_load[width=SIMDW](h * head_dim + d2)
                        * mod_buf.unsafe_load[width=SIMDW](d2),
                    )
                    d2 += SIMDW
            while d2 < head_dim:
                var s = (
                    scales[h * head_dim + d2] if elementwise else scales[h]
                ) * mod_buf[d2]
                qp[d2] *= s
                d2 += 1
            h += 1
        r += 1
    qh.unsafe_free()
    mod_buf.unsafe_free()
    scales.unsafe_free()


# =============================================================================
# RoPE (positions = token index; freqs buffer holds INVERSE frequencies)
# =============================================================================


@always_inline
def _rope_rotate_rows(
    mut x: Pointer[Float32, MutUntrackedOrigin],
    rows: Int,
    n_heads: Int,
    head_dim: Int,
    freqs: Pointer[Float32, MutUntrackedOrigin],
    interleaved: Bool,
):
    """Rotate one head-major blocked ``(n_heads, rows, head_dim)`` buffer;
    the rotary position of row ``r`` is ``r``.

    Replicates torch ``RotaryEmbedding.rotate`` (float32 trig) exactly,
    INCLUDING its interleaved-mode angle quirk: because cos/sin are built
    from ``cat([freqs, freqs])`` while pairs are interleaved, element ``d``
    uses angle ``r * freqs[d % (head_dim/2)]`` — odd elements do NOT use
    their pair's angle. The non-interleaved path is the classic half-split
    rotation. Element (h, r, d) lives at ``h*rows*head_dim + r*head_dim + d``
    (torch's ``(..., n_heads, seq, head_dim)`` after ``transpose(-3,-2)``).
    """
    var hf = (head_dim + 1) // 2
    var h = 0
    while h < n_heads:
        var r = 0
        while r < rows:
            var base = h * rows * head_dim + r * head_dim
            var j = 0
            while j < hf:
                if interleaved:
                    var d0 = base + 2 * j
                    var a0 = Float32(r) * freqs[(2 * j) % hf]
                    var a1 = Float32(r) * freqs[(2 * j + 1) % hf]
                    var c0 = cos(a0)
                    var s0 = sin(a0)
                    var c1 = cos(a1)
                    var s1 = sin(a1)
                    var x0 = x[d0]
                    var x1 = x[d0 + 1]
                    x[d0] = x0 * c0 - x1 * s0
                    x[d0 + 1] = x1 * c1 + x0 * s1
                else:
                    var d0 = base + j
                    var d1 = base + hf + j
                    var ang = Float32(r) * freqs[j]
                    var c = cos(ang)
                    var s = sin(ang)
                    var x0 = x[d0]
                    var x1 = x[d1]
                    x[d0] = x0 * c - x1 * s
                    x[d1] = x1 * c + x0 * s
                j += 1
            r += 1
        h += 1


def rope_apply(
    mut q: Pointer[Float32, MutUntrackedOrigin],
    mut k: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    k_rows: Int,
    n_heads: Int,
    head_dim: Int,
    freqs: Pointer[Float32, MutUntrackedOrigin],
    interleaved: Bool,
):
    """Apply rotary embeddings to head-split q and k buffers in place."""
    _rope_rotate_rows(q, q_rows, n_heads, head_dim, freqs, interleaved)
    _rope_rotate_rows(k, k_rows, n_heads, head_dim, freqs, interleaved)


# =============================================================================
# Canonical AttentionBlock (affine LayerNorm + SSMax + RoPE)
# =============================================================================
# Mirrors torch AttentionBlock/MultiheadAttention/_attention semantics:
#   qn = norm1(q); kn = qn if k is q else norm1(k); vn = kn
#   x = q + attn(qn, kn, vn)          # RoPE -> ssmax -> scores -> out_proj
#   x = x + linear2(gelu(linear1(norm2(x))))
# All scratch memory is allocated and freed inside the helpers.


def layer_norm_affine(
    x: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    weight: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    has_weight: Bool,
    has_bias: Bool,
):
    """Row-wise LayerNorm with optional affine (dummy pointers when absent).

    Kept scalar verbatim on purpose: the kv-cache parity test pins the
    plain and cached paths to bit-equality and the staged parity tests pin
    mojo to torch/numpy at atol 5e-5; vectorizing the normalize pass
    changed rounding (FMA contraction) enough to break those contracts.
    """
    var i = 0
    while i < n:
        var base = i * d
        var mu: Float32 = 0.0
        var j = 0
        while j < d:
            mu += x[base + j]
            j += 1
        mu /= Float32(d)
        var var_acc: Float32 = 0.0
        j = 0
        while j < d:
            var diff = x[base + j] - mu
            var_acc += diff * diff
            j += 1
        var inv_std = 1.0 / sqrt(var_acc / Float32(d) + 1e-5)
        j = 0
        while j < d:
            var v = (x[base + j] - mu) * inv_std
            if has_weight:
                v *= weight[j]
            if has_bias:
                v += bias[j]
            dst[base + j] = v
            j += 1
        i += 1


struct AttnParams(ImplicitlyCopyable, Movable):
    """Raw pointers into one packed canonical AttentionBlock section.

    Bias-free LayerNorms keep a (never dereferenced) dummy bias pointer;
    absence is encoded by ``ln_has_bias`` instead of Optionals.
    """

    var w_qkv: Pointer[Float32, MutUntrackedOrigin]
    var b_qkv: Pointer[Float32, MutUntrackedOrigin]
    var w_out: Pointer[Float32, MutUntrackedOrigin]
    var b_out: Pointer[Float32, MutUntrackedOrigin]
    var kind: Int
    var ssmax: Pointer[Float32, MutUntrackedOrigin]
    var ssmax_hidden: Int
    var ln1_w: Pointer[Float32, MutUntrackedOrigin]
    var ln1_b: Pointer[Float32, MutUntrackedOrigin]
    var ln2_w: Pointer[Float32, MutUntrackedOrigin]
    var ln2_b: Pointer[Float32, MutUntrackedOrigin]
    var ln_has_bias: Bool
    var w1: Pointer[Float32, MutUntrackedOrigin]
    var b1: Pointer[Float32, MutUntrackedOrigin]
    var w2: Pointer[Float32, MutUntrackedOrigin]
    var b2: Pointer[Float32, MutUntrackedOrigin]

    def __init__(
        out self,
        w_qkv: Pointer[Float32, MutUntrackedOrigin],
        b_qkv: Pointer[Float32, MutUntrackedOrigin],
        w_out: Pointer[Float32, MutUntrackedOrigin],
        b_out: Pointer[Float32, MutUntrackedOrigin],
        kind: Int,
        ssmax: Pointer[Float32, MutUntrackedOrigin],
        ssmax_hidden: Int,
        ln1_w: Pointer[Float32, MutUntrackedOrigin],
        ln1_b: Pointer[Float32, MutUntrackedOrigin],
        ln2_w: Pointer[Float32, MutUntrackedOrigin],
        ln2_b: Pointer[Float32, MutUntrackedOrigin],
        ln_has_bias: Bool,
        w1: Pointer[Float32, MutUntrackedOrigin],
        b1: Pointer[Float32, MutUntrackedOrigin],
        w2: Pointer[Float32, MutUntrackedOrigin],
        b2: Pointer[Float32, MutUntrackedOrigin],
    ):
        self.w_qkv = w_qkv
        self.b_qkv = b_qkv
        self.w_out = w_out
        self.b_out = b_out
        self.kind = kind
        self.ssmax = ssmax
        self.ssmax_hidden = ssmax_hidden
        self.ln1_w = ln1_w
        self.ln1_b = ln1_b
        self.ln2_w = ln2_w
        self.ln2_b = ln2_b
        self.ln_has_bias = ln_has_bias
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2


def attention_block_size(
    d_model: Int,
    d_ff: Int,
    n_heads: Int,
    head_dim: Int,
    kind: Int,
    hidden: Int,
    bias_free_ln: Bool,
) -> Int:
    """Packed float32 count of one AttentionBlock in canonical order."""
    var d = d_model
    var size = 3 * d * d + 3 * d + d * d + d
    size += ssmax_section_size(kind, n_heads, head_dim, hidden)
    # norm1 + norm2: weight always present, bias only when not bias-free
    size += 2 * d if bias_free_ln else 4 * d
    size += d_ff * d + d_ff + d * d_ff + d
    return size


def attn_params_at(
    params: Pointer[Float32, MutUntrackedOrigin],
    off: Int,
    d_model: Int,
    d_ff: Int,
    n_heads: Int,
    head_dim: Int,
    kind: Int,
    hidden: Int,
    bias_free_ln: Bool,
) -> AttnParams:
    """Walk one canonical AttentionBlock section starting at ``params[off]``."""
    var cur = params + off
    var w_qkv = cur
    cur += 3 * d_model * d_model
    var b_qkv = cur
    cur += 3 * d_model
    var w_out = cur
    cur += d_model * d_model
    var b_out = cur
    cur += d_model
    var ssmax = params  # dummy when kind == 0 (never dereferenced)
    if kind != 0:
        ssmax = cur
        cur += ssmax_section_size(kind, n_heads, head_dim, hidden)
    var ln1_w = cur
    cur += d_model
    var ln1_b = params
    if not bias_free_ln:
        ln1_b = cur
        cur += d_model
    var ln2_w = cur
    cur += d_model
    var ln2_b = params
    if not bias_free_ln:
        ln2_b = cur
        cur += d_model
    var w1 = cur
    cur += d_ff * d_model
    var b1 = cur
    cur += d_ff
    var w2 = cur
    cur += d_model * d_ff
    var b2 = cur
    return AttnParams(
        w_qkv=w_qkv,
        b_qkv=b_qkv,
        w_out=w_out,
        b_out=b_out,
        kind=kind,
        ssmax=ssmax,
        ssmax_hidden=hidden,
        ln1_w=ln1_w,
        ln1_b=ln1_b,
        ln2_w=ln2_w,
        ln2_b=ln2_b,
        ln_has_bias=not bias_free_ln,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )


@always_inline
def _add_row_bias(
    mut m: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    rows: Int,
    cols: Int,
):
    var i = 0
    while i < rows:
        var base = i * cols
        var c = 0
        while c + SIMDW <= cols:
            m.unsafe_store[width=SIMDW](
                base + c,
                m.unsafe_load[width=SIMDW](base + c)
                + bias.unsafe_load[width=SIMDW](c),
            )
            c += SIMDW
        while c < cols:
            m[base + c] += bias[c]
            c += 1
        i += 1


@always_inline
def _copy_f32(
    src: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
):
    var i = 0
    while i + SIMDW <= n:
        dst.unsafe_store[width=SIMDW](i, src.unsafe_load[width=SIMDW](i))
        i += SIMDW
    while i < n:
        dst[i] = src[i]
        i += 1


@always_inline
def _add_f32(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
):
    """dst[i] = a[i] + b[i] (dst may alias a or b)."""
    var i = 0
    while i + SIMDW <= n:
        dst.unsafe_store[width=SIMDW](
            i,
            a.unsafe_load[width=SIMDW](i) + b.unsafe_load[width=SIMDW](i),
        )
        i += SIMDW
    while i < n:
        dst[i] = a[i] + b[i]
        i += 1


@always_inline
def _scale_inplace(
    p: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    s: Float32,
):
    var sv = SIMD[DType.float32, SIMDW](s)
    var i = 0
    while i + SIMDW <= n:
        p.unsafe_store[width=SIMDW](
            i, p.unsafe_load[width=SIMDW](i) * sv
        )
        i += SIMDW
    while i < n:
        p[i] *= s
        i += 1


@always_inline
def _dot(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
) -> Float32:
    var acc = SIMD[DType.float32, SIMDW](0.0)
    var i = 0
    while i + SIMDW <= n:
        acc += a.unsafe_load[width=SIMDW](i) * b.unsafe_load[width=SIMDW](i)
        i += SIMDW
    var s = acc.reduce_add()
    while i < n:
        s += a[i] * b[i]
        i += 1
    return s


@always_inline
def _softmax_rows_inplace(
    scores: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    k_rows: Int,
    scale: Float32,
):
    """Scale by ``scale`` then row-wise softmax, in place (SIMD with tails)."""
    var ts = 0
    while ts < q_rows:
        var sp = scores + ts * k_rows
        var nvec = k_rows & ~(SIMDW - 1)
        # pass 1: scale, track row max
        var mxv = SIMD[DType.float32, SIMDW](-3.0e38)
        var s = 0
        while s < nvec:
            var v = sp.unsafe_load[width=SIMDW](s) * scale
            sp.unsafe_store[width=SIMDW](s, v)
            mxv = max(mxv, v)
            s += SIMDW
        var mx = mxv.reduce_max()
        while s < k_rows:
            var v = sp[s] * scale
            sp[s] = v
            if v > mx:
                mx = v
            s += 1
        # pass 2: exp(x - max) + sum
        var sumv = SIMD[DType.float32, SIMDW](0.0)
        s = 0
        while s < nvec:
            var e = exp(sp.unsafe_load[width=SIMDW](s) - mx)
            sp.unsafe_store[width=SIMDW](s, e)
            sumv += e
            s += SIMDW
        var sum_v = sumv.reduce_add()
        while s < k_rows:
            var e = exp(sp[s] - mx)
            sp[s] = e
            sum_v += e
            s += 1
        # pass 3: normalize
        s = 0
        while s < nvec:
            sp.unsafe_store[width=SIMDW](
                s,
                sp.unsafe_load[width=SIMDW](s) / sum_v,
            )
            s += SIMDW
        while s < k_rows:
            sp[s] /= sum_v
            s += 1
        ts += 1


@always_inline
def _heads_attention(
    q_a: Pointer[Float32, MutUntrackedOrigin],
    k_a: Pointer[Float32, MutUntrackedOrigin],
    v_a: Pointer[Float32, MutUntrackedOrigin],
    out_a: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    k_rows: Int,
    n_heads: Int,
    head_dim: Int,
):
    """Per-head scores/softmax/AV over head-major blocked buffers.

    Head ``h`` reads ``q_a/k_a/v_a`` slices ``h*q_rows*head_dim`` /
    ``h*k_rows*head_dim`` and writes ``out_a`` and ``scores`` slices that no
    other head touches, so the loop is data-parallel over heads. Large
    workloads partition heads across pthreads (each partition gets a private
    softmax scratch so concurrent heads never share score rows); small ones
    stay inline with the threaded GEMM dispatchers.
    """
    var scale = 1.0 / sqrt(Float32(head_dim))
    var threads = hardware_threads()
    var parts = 0
    if threads >= 2 and n_heads >= 2:
        var macs = n_heads * 2 * q_rows * k_rows * head_dim
        if macs >= HEAD_ATTN_THREAD_MIN_MACS:
            parts = threads if threads < n_heads else n_heads
            # Per-partition softmax scratch must fit the memory budget;
            # oversized problems keep the serial path (their GEMMs thread).
            if parts * q_rows * k_rows * 4 > HEAD_ATTN_SCRATCH_CAP:
                parts = 0

    if parts >= 2:
        var scratch = alloc[Float32](parts * q_rows * k_rows + 16)
        var jobs = alloc[HeadAttnJob](parts)
        var chunk = (n_heads + parts - 1) // parts
        var p = 0
        var lo = 0
        while p < parts and lo < n_heads:
            var hi = lo + chunk
            if hi > n_heads:
                hi = n_heads
            jobs.unsafe_offset(p)[] = HeadAttnJob(
                q_a,
                k_a,
                v_a,
                out_a,
                scratch + p * q_rows * k_rows,
                q_rows,
                k_rows,
                head_dim,
                lo,
                hi,
            )
            lo = hi
            p += 1
        comptime WorkerFn = GemmWorkerFn
        var fp: WorkerFn = head_attn_worker
        run_partitioned_range(jobs, parts, fp)
        scratch.unsafe_free()
        jobs.unsafe_free()
        return

    var h = 0
    while h < n_heads:
        var q_h = q_a + h * q_rows * head_dim
        var k_h = k_a + h * k_rows * head_dim
        var v_h = v_a + h * k_rows * head_dim
        var o_h = out_a + h * q_rows * head_dim
        gemm_nt(q_rows, k_rows, head_dim, q_h, k_h, scores)
        _softmax_rows_inplace(scores, q_rows, k_rows, scale)
        gemm_nn(q_rows, head_dim, k_rows, scores, v_h, o_h)
        h += 1


comptime HEAD_ATTN_THREAD_MIN_MACS = 1 << 19  # ~0.5M MACs total before head threading pays off
comptime HEAD_ATTN_SCRATCH_CAP = 1 << 29  # 512MB budget for per-partition softmax scratch


struct HeadAttnJob(Movable):
    var q_a: Pointer[Float32, MutUntrackedOrigin]
    var k_a: Pointer[Float32, MutUntrackedOrigin]
    var v_a: Pointer[Float32, MutUntrackedOrigin]
    var out_a: Pointer[Float32, MutUntrackedOrigin]
    var scores: Pointer[Float32, MutUntrackedOrigin]  # this partition's private scratch
    var q_rows: Int
    var k_rows: Int
    var head_dim: Int
    var lo: Int
    var hi: Int

    def __init__(
        out self,
        q_a: Pointer[Float32, MutUntrackedOrigin],
        k_a: Pointer[Float32, MutUntrackedOrigin],
        v_a: Pointer[Float32, MutUntrackedOrigin],
        out_a: Pointer[Float32, MutUntrackedOrigin],
        scores: Pointer[Float32, MutUntrackedOrigin],
        q_rows: Int,
        k_rows: Int,
        head_dim: Int,
        lo: Int,
        hi: Int,
    ):
        self.q_a = q_a
        self.k_a = k_a
        self.v_a = v_a
        self.out_a = out_a
        self.scores = scores
        self.q_rows = q_rows
        self.k_rows = k_rows
        self.head_dim = head_dim
        self.lo = lo
        self.hi = hi


@export("shinrin_head_attn_worker")
def head_attn_worker(raw: P_U8) abi("C") -> None:
    """Run serial row GEMMs + softmax for heads [lo, hi) of one partition.

    Uses the serial row kernels directly: the parallelism here is over heads,
    so nesting the threaded GEMM dispatchers would oversubscribe the cores.
    """
    var jp = raw.unsafe_bitcast[HeadAttnJob]()
    var scale = 1.0 / sqrt(Float32(jp[].head_dim))
    var sc = jp[].scores
    var h = jp[].lo
    while h < jp[].hi:
        var q_h = jp[].q_a + h * jp[].q_rows * jp[].head_dim
        var k_h = jp[].k_a + h * jp[].k_rows * jp[].head_dim
        var v_h = jp[].v_a + h * jp[].k_rows * jp[].head_dim
        var o_h = jp[].out_a + h * jp[].q_rows * jp[].head_dim
        gemm_nt_rows(jp[].q_rows, jp[].k_rows, jp[].head_dim, q_h, k_h, sc, 0, jp[].q_rows)
        _softmax_rows_inplace(sc, jp[].q_rows, jp[].k_rows, scale)
        gemm_nn_rows(jp[].q_rows, jp[].head_dim, jp[].k_rows, sc, v_h, o_h, 0, jp[].q_rows)
        h += 1


@always_inline
def _split_heads(
    src_proj: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    rows: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
):
    """(rows, d_model) row-major -> head-major blocked (n_heads, rows, hd).

    Both the source run (``i*d_model + h*hd``) and destination run
    (``h*rows*hd + i*hd``) are contiguous chunks of ``head_dim`` floats,
    so each is copied with one vectorized memcpy-style pass.
    """
    var i = 0
    while i < rows:
        var src_row = src_proj + i * d_model
        var dst_row_base = i * head_dim
        var h = 0
        while h < n_heads:
            _copy_f32(
                src_row + h * head_dim,
                dst + h * rows * head_dim + dst_row_base,
                head_dim,
            )
            h += 1
        i += 1


@always_inline
def _merge_heads(
    out_a: Pointer[Float32, MutUntrackedOrigin],
    merged: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
):
    """Head-major blocked (n_heads, q_rows, hd) -> (q_rows, d_model)."""
    var mi = 0
    while mi < q_rows:
        var hh = 0
        while hh < n_heads:
            _copy_f32(
                out_a + hh * q_rows * head_dim + mi * head_dim,
                merged + mi * d_model + hh * head_dim,
                head_dim,
            )
            hh += 1
        mi += 1


@always_inline
def _attention_tail(
    q_in: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_a: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    d_ff: Int,
    p: AttnParams,
):
    """Shared attention tail: merge heads -> out proj -> residual -> FFN.

    Used verbatim by both the plain and cached-KV block forwards so the two
    stay bit-exact behind the attention core.
    """
    var d = d_model
    var merged = alloc[Float32](q_rows * d + 16)
    _merge_heads(out_a, merged, q_rows, d, n_heads, head_dim)
    var res = alloc[Float32](q_rows * d + 16)
    gemm_nt(q_rows, d, d, merged, p.w_out, res)
    _add_row_bias(res, p.b_out, q_rows, d)
    _add_f32(q_in, res, dst, q_rows * d)

    # -- sublayer 2: FFN ----------------------------------------------------
    var ffn_in = alloc[Float32](q_rows * d + 16)
    layer_norm_affine(dst, ffn_in, q_rows, d, p.ln2_w, p.ln2_b, True, p.ln_has_bias)
    var ffh = alloc[Float32](q_rows * d_ff + 16)
    gemm_nt(q_rows, d_ff, d, ffn_in, p.w1, ffh)
    _add_row_bias(ffh, p.b1, q_rows, d_ff)
    var ftotal = q_rows * d_ff
    mi = 0
    while mi + SIMDW <= ftotal:
        ffh.unsafe_store[width=SIMDW](
            mi,
            gelu8(ffh.unsafe_load[width=SIMDW](mi)),
        )
        mi += SIMDW
    while mi < ftotal:
        ffh[mi] = gelu_scalar(ffh[mi])
        mi += 1
    gemm_nt(q_rows, d, d_ff, ffh, p.w2, res)
    _add_row_bias(res, p.b2, q_rows, d)
    _add_f32(dst, res, dst, q_rows * d)
    merged.unsafe_free()
    res.unsafe_free()
    ffn_in.unsafe_free()
    ffh.unsafe_free()


def attention_block_forward(
    q_in: Pointer[Float32, MutUntrackedOrigin],
    k_in: Optional[Pointer[Float32, MutUntrackedOrigin]],
    dst: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    k_rows: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    d_ff: Int,
    p: AttnParams,
    rope_freqs: Optional[Pointer[Float32, MutUntrackedOrigin]],
    rope_interleaved: Bool,
    kv_k_out: Optional[Pointer[Float32, MutUntrackedOrigin]] = None,
    kv_v_out: Optional[Pointer[Float32, MutUntrackedOrigin]] = None,
):
    """Run one full pre-norm AttentionBlock (torch-faithful order).

    ``k_in=None`` means self-attention: k/v project from the normed query
    (``kn = qn``, matching torch when ``k is q``). Cross-attention assumes
    ``v is k`` (``vn = kn``), true for every call site here. ``dst`` may
    alias ``q_in`` (in-place residual updates are safe).
    """
    var d = d_model

    # -- sublayer 1: norms ------------------------------------------------
    var qn = alloc[Float32](q_rows * d + 16)
    layer_norm_affine(q_in, qn, q_rows, d, p.ln1_w, p.ln1_b, True, p.ln_has_bias)
    var kn: Pointer[Float32, MutUntrackedOrigin]
    var kn_owned = False
    if k_in is not None:
        kn = alloc[Float32](k_rows * d + 16)
        kn_owned = True
        layer_norm_affine(
            k_in.value(), kn, k_rows, d, p.ln1_w, p.ln1_b, True, p.ln_has_bias
        )
    else:
        kn = qn

    # -- projections + biases ---------------------------------------------
    var q_proj = alloc[Float32](q_rows * d + 16)
    var k_proj = alloc[Float32](k_rows * d + 16)
    var v_proj = alloc[Float32](k_rows * d + 16)
    gemm_nt(q_rows, d, d, qn, p.w_qkv, q_proj)
    _add_row_bias(q_proj, p.b_qkv, q_rows, d)
    gemm_nt(k_rows, d, d, kn, p.w_qkv + d * d, k_proj)
    _add_row_bias(k_proj, p.b_qkv + d, k_rows, d)
    gemm_nt(k_rows, d, d, kn, p.w_qkv + 2 * d * d, v_proj)
    _add_row_bias(v_proj, p.b_qkv + 2 * d, k_rows, d)

    # -- head-major reshape ------------------------------------------------
    var q_a = alloc[Float32](q_rows * n_heads * head_dim + 16)
    var k_a = alloc[Float32](k_rows * n_heads * head_dim + 16)
    var v_a = alloc[Float32](k_rows * n_heads * head_dim + 16)
    _split_heads(q_proj, q_a, q_rows, d, n_heads, head_dim)
    _split_heads(k_proj, k_a, k_rows, d, n_heads, head_dim)
    _split_heads(v_proj, v_a, k_rows, d, n_heads, head_dim)

    # -- RoPE then SSMax (reference applies rope BEFORE ssmax) --------------
    if rope_freqs is not None:
        rope_apply(
            q_a, k_a, q_rows, k_rows, n_heads, head_dim,
            rope_freqs.value(), rope_interleaved,
        )
    # -- optional K/V capture (post-rope; ssmax never touches k/v) ----------
    if kv_k_out is not None:
        _copy_f32(k_a, kv_k_out.value(), k_rows * n_heads * head_dim)
        _copy_f32(v_a, kv_v_out.value(), k_rows * n_heads * head_dim)
    ssmax_apply(q_a, q_rows, k_rows, n_heads, head_dim, p.kind, p.ssmax, p.ssmax_hidden)

    # -- attention per head -------------------------------------------------
    var out_a = alloc[Float32](q_rows * n_heads * head_dim + 16)
    var scores = alloc[Float32](q_rows * k_rows + 16)
    _heads_attention(q_a, k_a, v_a, out_a, scores, q_rows, k_rows, n_heads, head_dim)

    # -- merge heads + out projection + residual + FFN ----------------------
    _attention_tail(q_in, dst, out_a, q_rows, d, n_heads, head_dim, d_ff, p)

    qn.unsafe_free()
    if kn_owned:
        kn.unsafe_free()
    q_proj.unsafe_free()
    k_proj.unsafe_free()
    v_proj.unsafe_free()
    q_a.unsafe_free()
    k_a.unsafe_free()
    v_a.unsafe_free()
    out_a.unsafe_free()
    scores.unsafe_free()


def attention_block_forward_cached(
    q_in: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    q_rows: Int,
    k_rows: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    d_ff: Int,
    p: AttnParams,
    cached_k: Pointer[Float32, MutUntrackedOrigin],
    cached_v: Pointer[Float32, MutUntrackedOrigin],
):
    """Pre-norm AttentionBlock whose K/V come from a prior capture.

    Bit-exact twin of the self-attention path of ``attention_block_forward``
    whenever ``cached_k/v`` were captured (``kv_k_out``/``kv_v_out``) from the
    same key inputs: the query projection reads the same ``w_qkv`` rows, SSMax
    sees the same key count, and every GEMM row's accumulation order is
    unchanged. Skips norming/projecting the keys entirely — that work is what
    the cache exists to avoid. Rope-free by construction: every cacheable
    TabICL stage (col blocks, ICL blocks) has no positional encoding; only
    the uncached row stage uses RoPE. ``dst`` may alias ``q_in``.
    """
    var d = d_model

    # -- sublayer 1: query norm + projection --------------------------------
    var qn = alloc[Float32](q_rows * d + 16)
    layer_norm_affine(q_in, qn, q_rows, d, p.ln1_w, p.ln1_b, True, p.ln_has_bias)
    var q_proj = alloc[Float32](q_rows * d + 16)
    gemm_nt(q_rows, d, d, qn, p.w_qkv, q_proj)
    _add_row_bias(q_proj, p.b_qkv, q_rows, d)

    # -- head-major reshape + ssmax on queries ------------------------------
    var q_a = alloc[Float32](q_rows * n_heads * head_dim + 16)
    _split_heads(q_proj, q_a, q_rows, d, n_heads, head_dim)
    ssmax_apply(q_a, q_rows, k_rows, n_heads, head_dim, p.kind, p.ssmax, p.ssmax_hidden)

    # -- attention against cached K/V ---------------------------------------
    var out_a = alloc[Float32](q_rows * n_heads * head_dim + 16)
    var scores = alloc[Float32](q_rows * k_rows + 16)
    _heads_attention(
        q_a, cached_k, cached_v, out_a, scores, q_rows, k_rows, n_heads, head_dim
    )

    # -- merge heads + out projection + residual + FFN ----------------------
    _attention_tail(q_in, dst, out_a, q_rows, d, n_heads, head_dim, d_ff, p)

    qn.unsafe_free()
    q_proj.unsafe_free()
    q_a.unsafe_free()
    out_a.unsafe_free()
    scores.unsafe_free()


def isab_forward(
    x: Pointer[Float32, MutUntrackedOrigin],
    ind_vectors: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    num_inds: Int,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    d_ff: Int,
    p1: AttnParams,
    p2: AttnParams,
):
    """Induced self-attention: two FULL AttentionBlocks (torch-faithful).

    hidden = block1(ind_vectors, x, x)  (cross-attn onto inducing vectors)
    dst    = block2(x, hidden, hidden)  (each source row attends to hidden)
    """
    var hidden = alloc[Float32](num_inds * d_model + 16)
    attention_block_forward(
        ind_vectors, x, hidden, num_inds, n, d_model, n_heads, head_dim, d_ff,
        p1, None, False,
    )
    attention_block_forward(
        x, hidden, dst, n, num_inds, d_model, n_heads, head_dim, d_ff,
        p2, None, False,
    )
    hidden.unsafe_free()
