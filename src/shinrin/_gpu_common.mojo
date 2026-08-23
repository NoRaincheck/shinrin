"""Shared Metal device kernels and host helpers for the Mojo GPU trainers.

Imported by ``_tabm_gpu_kernels.mojo`` and ``_mlp_gpu_kernels.mojo``
(sibling imports resolve because ``mojo build`` compiles them together).
Everything here must be safe to run inside GPU kernel functions: no
Float64 arithmetic (Apple GPUs have no fp64), no host-only APIs.

Conventions:

- All sizes/flags are ``Int32``; seeds are ``Int64``; data is float32.
- GEMM kernels map one thread to one output element of ``C`` and assume
  contiguous inputs (``A`` row-major with row stride ``kk``, ``B``
  row-major with row stride ``n``); ``ldc`` allows a strided ``C``.
- Reductions write one partial per block; the host sums partials in
  float64 so results stay deterministic for a fixed grid size.
- Dropout decisions hash ``(round_seed, element_id)`` through splitmix64
  instead of consuming a sequential RNG stream, keeping the kernels
  deterministic without cross-thread state. P(drop) = thresh/256, the
  same granularity as the CPU kernels' byte-threshold scheme.
"""

from std.gpu import block_idx, block_dim, grid_dim, thread_idx
from std.math import exp, log, sqrt


comptime GTPB = 256  # threads per block for 1D kernels


# =============================================================================
# device-side helpers
# =============================================================================


@always_inline
def gpu_global_idx() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def mix_u64(x: UInt64) -> UInt64:
    """splitmix64 finalizer (same as the CPU kernels)."""
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def dropout_keep(elem_id: Int, round_seed: Int64, thresh: UInt64) -> Bool:
    """True iff the element survives dropout for this round."""
    var z = mix_u64(UInt64(round_seed) ^ mix_u64(UInt64(elem_id) + 0x9E3779B97F4A7C15))
    return ((z >> 56) & 0xFF) >= thresh


@always_inline
def gpu_sigmoid(x: Float32) -> Float32:
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)


@always_inline
def gpu_softplus(x: Float32) -> Float32:
    if x > 20.0:
        return x
    if x < -20.0:
        return exp(x)
    return log(1.0 + exp(x))


# =============================================================================
# elementwise / reduction kernels
# =============================================================================


def dk_zero(dst: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    var i = gpu_global_idx()
    if i < Int(n):
        dst[i] = 0.0


def dk_copy(dst: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    var i = gpu_global_idx()
    if i < Int(n):
        dst[i] = src[i]


def dk_accumulate(dst: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    # dst += src
    var i = gpu_global_idx()
    if i < Int(n):
        dst[i] += src[i]


def dk_saxpy(dst: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin], scale: Float32, n: Int32):
    # dst += scale * src
    var i = gpu_global_idx()
    if i < Int(n):
        dst[i] += scale * src[i]


def dk_mul3(dst: Pointer[Float32, MutUntrackedOrigin], a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    # dst = dst * a * b (in-place triple multiply)
    var i = gpu_global_idx()
    if i < Int(n):
        dst[i] *= a[i] * b[i]


def dk_sum(src: Pointer[Float32, MutUntrackedOrigin], partials: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    # partials[thread] = that thread's strided partial sum; the host sums
    # grid_dim * block_dim entries in float64. There is one slot per thread,
    # never per block -- without shared-memory reduction a per-block slot
    # would suffer last-writer-wins races.
    var tid = gpu_global_idx()
    var acc: Float32 = 0.0
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var i = tid
    while i < Int(n):
        acc += src[i]
        i += stride
    partials[tid] = acc


def dk_sum_sq(src: Pointer[Float32, MutUntrackedOrigin], partials: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    var tid = gpu_global_idx()
    var acc: Float32 = 0.0
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var i = tid
    while i < Int(n):
        acc += src[i] * src[i]
        i += stride
    partials[tid] = acc


def dk_max_abs(src: Pointer[Float32, MutUntrackedOrigin], partials: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    var tid = gpu_global_idx()
    var acc: Float32 = 0.0
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var i = tid
    while i < Int(n):
        var v = src[i]
        var av = v if v > 0.0 else -v
        if av > acc:
            acc = av
        i += stride
    partials[tid] = acc


def dk_adam_update(
    theta: Pointer[Float32, MutUntrackedOrigin],
    m: Pointer[Float32, MutUntrackedOrigin],
    v: Pointer[Float32, MutUntrackedOrigin],
    g: Pointer[Float32, MutUntrackedOrigin],
    p_total: Int32,
    alpha: Float32,
    lr: Float32,
    bc1: Float32,
    bc2: Float32,
):
    """Bias-corrected Adam step with L2 (mirrors adam_update on CPU)."""
    var i = gpu_global_idx()
    if i < Int(p_total):
        var gi = g[i] + alpha * theta[i]
        var mv = 0.9 * m[i] + 0.1 * gi
        var vv = 0.999 * v[i] + 0.001 * gi * gi
        m[i] = mv
        v[i] = vv
        theta[i] -= lr * (mv / bc1) / (sqrt(vv / bc2) + 1e-8)


# =============================================================================
# input building / embedding kernels (shared by TabM and MLP trainers)
# =============================================================================


def dk_gather_rows(
    dst: Pointer[Float32, MutUntrackedOrigin],
    src: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    b: Int32,
    width: Int32,
):
    # dst[r,t] = src[idx[r],t]
    var e = gpu_global_idx()
    if e < Int(b) * Int(width):
        dst[e] = src[Int(idx[e // Int(width)]) * Int(width) + (e % Int(width))]


def dk_gather_cols_idx(
    dst: Pointer[Float32, MutUntrackedOrigin],
    src: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    b: Int32,
    src_width: Int32,
    off: Int32,
    cnt: Int32,
):
    # dst[r,t] = src[idx[r], off+t]   (cnt columns starting at off)
    var e = gpu_global_idx()
    if e < Int(b) * Int(cnt):
        dst[e] = src[Int(idx[e // Int(cnt)]) * Int(src_width) + Int(off) + (e % Int(cnt))]


def dk_slice_cols(
    dst: Pointer[Float32, MutUntrackedOrigin],
    src: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    src_width: Int32,
    off: Int32,
    cnt: Int32,
):
    # dst[r,t] = src[r, off+t]
    var e = gpu_global_idx()
    if e < Int(b) * Int(cnt):
        dst[e] = src[(e // Int(cnt)) * Int(src_width) + Int(off) + (e % Int(cnt))]


def dk_input_copy(
    h: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    b: Int32,
    F: Int32,
    ccat: Int32,
):
    # h[r,:] = [x_num[idx[r]], x_cat[idx[r]]]  (embeddings disabled)
    var e = gpu_global_idx()
    if e < Int(b) * (Int(F) + Int(ccat)):
        var r = e // (Int(F) + Int(ccat))
        var t = e % (Int(F) + Int(ccat))
        var src = Int(idx[r])
        h[e] = x_num[src * Int(F) + t] if t < Int(F) else x_cat[src * Int(ccat) + (t - Int(F))]


def dk_cat_copy(
    h: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    b: Int32,
    ccat: Int32,
    out_off: Int32,
    d_in: Int32,
):
    # h[r, out_off+t] = x_cat[idx[r],t]
    var e = gpu_global_idx()
    if e < Int(b) * Int(ccat):
        var r = e // Int(ccat)
        var t = e % Int(ccat)
        h[r * Int(d_in) + Int(out_off) + t] = x_cat[Int(idx[r]) * Int(ccat) + t]


def dk_embed_linear(
    h: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    w0: Pointer[Float32, MutUntrackedOrigin],
    b0: Pointer[Float32, MutUntrackedOrigin],
    b_rows: Int32,
    F: Int32,
    demb: Int32,
    d_in: Int32,
):
    # linear embedding component: h[r, f*demb+o] = x*w0+b0
    var e = gpu_global_idx()
    if e < Int(b_rows) * Int(F) * Int(demb):
        var r = e // (Int(F) * Int(demb))
        var fo = e % (Int(F) * Int(demb))
        var xv = x_num[Int(idx[r]) * Int(F) + (fo // Int(demb))]
        h[r * Int(d_in) + fo] = xv * w0[fo] + b0[fo]


def dk_relu_add(
    h: Pointer[Float32, MutUntrackedOrigin],
    pl: Pointer[Float32, MutUntrackedOrigin],
    b_rows: Int32,
    fdemb: Int32,
    d_in: Int32,
):
    # h[r, o] += relu(pl[r, o]) for o < fdemb
    var e = gpu_global_idx()
    if e < Int(b_rows) * Int(fdemb):
        var r = e // Int(fdemb)
        var v = pl[e]
        if v > 0.0:
            h[r * Int(d_in) + (e % Int(fdemb))] += v


def dk_emb_wb_grad(
    dw0: Pointer[Float32, MutUntrackedOrigin],
    db0: Pointer[Float32, MutUntrackedOrigin],
    xnum: Pointer[Float32, MutUntrackedOrigin],
    dh: Pointer[Float32, MutUntrackedOrigin],
    b_rows: Int32,
    F: Int32,
    demb: Int32,
    d_in: Int32,
):
    # dw0[f,o] += sum_r xnum[r,f]*dh[r,f*demb+o] ; db0[f,o] += sum_r dh[r,...]
    var e = gpu_global_idx()
    if e < Int(F) * Int(demb):
        var f = e // Int(demb)
        var acc_w: Float32 = 0.0
        var acc_b: Float32 = 0.0
        for r in range(Int(b_rows)):
            var dg = dh[r * Int(d_in) + e]
            acc_w += xnum[r * Int(F) + f] * dg
            acc_b += dg
        dw0[e] += acc_w
        db0[e] += acc_b


def dk_dpl(
    dpl: Pointer[Float32, MutUntrackedOrigin],
    pl: Pointer[Float32, MutUntrackedOrigin],
    dh: Pointer[Float32, MutUntrackedOrigin],
    b_rows: Int32,
    f: Int32,
    F: Int32,
    demb: Int32,
    d_in: Int32,
):
    # dpl[r,o] = dh[r, f*demb+o] * (pl[r, f*demb+o] >= 0)
    var e = gpu_global_idx()
    if e < Int(b_rows) * Int(demb):
        var r = e // Int(demb)
        var col = Int(f) * Int(demb) + (e % Int(demb))
        var dg = dh[r * Int(d_in) + col]
        dpl[e] = dg if pl[r * Int(F) * Int(demb) + col] >= 0.0 else 0.0



def dk_gemm_nt(
    c: Pointer[Float32, MutUntrackedOrigin],
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    m: Int32,
    n: Int32,
    kk: Int32,
    ldc: Int32,
):
    # C (m,n) = A (m,kk) @ B (n,kk)^T ; C rows have leading dimension ldc
    var idx = gpu_global_idx()
    if idx < Int(m) * Int(n):
        var row = idx // Int(n)
        var col = idx % Int(n)
        var acc: Float32 = 0.0
        var ap = a + row * Int(kk)
        var bp = b + col * Int(kk)
        for t in range(Int(kk)):
            acc += ap[t] * bp[t]
        c[row * Int(ldc) + col] = acc


def dk_gemm_nn(
    c: Pointer[Float32, MutUntrackedOrigin],
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    m: Int32,
    n: Int32,
    kk: Int32,
    ldc: Int32,
):
    # C (m,n) = A (m,kk) @ B (kk,n)
    var idx = gpu_global_idx()
    if idx < Int(m) * Int(n):
        var row = idx // Int(n)
        var col = idx % Int(n)
        var acc: Float32 = 0.0
        var ap = a + row * Int(kk)
        for t in range(Int(kk)):
            acc += ap[t] * b[t * Int(n) + col]
        c[row * Int(ldc) + col] = acc


def dk_gemm_tn_acc(
    c: Pointer[Float32, MutUntrackedOrigin],
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    m: Int32,
    n: Int32,
    kk: Int32,
    ldc: Int32,
):
    # C (m,n) += A (kk,m)^T @ B (kk,n)
    var idx = gpu_global_idx()
    if idx < Int(m) * Int(n):
        var row = idx // Int(n)
        var col = idx % Int(n)
        var acc: Float32 = 0.0
        for t in range(Int(kk)):
            acc += a[t * Int(m) + row] * b[t * Int(n) + col]
        c[row * Int(ldc) + col] += acc
