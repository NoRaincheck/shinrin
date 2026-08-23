"""Metal (Mojo GPU) training kernels for the vendored TabM model.

Compiles to the ``shinrin._native_tabm_gpu`` Python extension module
(build with ``just build-tabm-metal``). Exposes a ``TabMTrainer`` bound
type whose API mirrors the CPU kernels in ``_tabm_kernels.mojo`` exactly,
so ``shinrin._tabm._mojo_trainer`` drives either backend transparently:

- ``adam_epoch(parts)``: one shuffled minibatch Adam epoch (dropout included)
- ``lbfgs_minimize(parts)``: full-batch L-BFGS with backtracking line search
- ``loss_grad(parts)``: full-batch loss + gradient (parity testing)
- ``forward_avg(parts)``: k-member-averaged predictions into a preallocated array

Parameter layout matches ``shinrin._tabm._layers.TabMParams.flatten``
(see ``_tabm_kernels.mojo`` for the table). NumPy arrays cross the
boundary as host pointers and are copied to device memory per call;
``theta``/``m``/``v`` are copied back after mutation.

Differences from the CPU kernels (documented, parity tests tolerate both):

- GEMMs run one thread per output element without shared-memory tiling.
- Reductions accumulate in float32 on device; partial sums are reduced
  in float64 on the host.
- Dropout draws hash ``(round_seed, element_id)`` through splitmix64
  instead of a sequential RNG stream, so results are deterministic for a
  fixed seed but differ from the CPU bit stream. The row shuffle uses
  the identical Fisher-Yates stream as the CPU kernels.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log, pow
from std.memory import alloc
from std.io import Writer
from std.os import abort, getenv
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from _gpu_common import (
    GTPB,
    dk_accumulate,
    dk_adam_update,
    dk_cat_copy,
    dk_copy,
    dk_dpl,
    dk_embed_linear,
    dk_emb_wb_grad,
    dk_gather_cols_idx,
    dk_gather_rows,
    dk_zero,
    dk_gemm_nn,
    dk_gemm_nt,
    dk_gemm_tn_acc,
    dk_input_copy,
    dk_relu_add,
    dk_saxpy,
    dk_slice_cols,
    dk_sum,
    dk_sum_sq,
    dropout_keep,
    gpu_global_idx,
    gpu_sigmoid,
    gpu_softplus,
    mix_u64,
)


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


def iface_dim1(arr: PythonObject) raises -> Int:
    """Second array dimension, or 1 for 1-D arrays."""
    var shape = arr.__array_interface__["shape"]
    if len(shape) > 1:
        return Int(py=shape[1])
    return 1


def np_module() raises -> PythonObject:
    return Python.import_module("numpy")


def np_empty_1d(np: PythonObject, n: Int, dtype: String) raises -> PythonObject:
    return np.empty(Python.tuple(Int(n)), dtype)


def cdiv(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def up_f32(ctx: DeviceContext, src: Pointer[Float32, MutUntrackedOrigin], n: Int) raises -> DeviceBuffer[DType.float32]:
    """Copy a host buffer to a fresh device buffer."""
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(buf, src)
    return buf


def up_i32(ctx: DeviceContext, src: Pointer[Int32, MutUntrackedOrigin], n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(buf, src)
    return buf


@always_inline
def devp(buf: DeviceBuffer[DType.float32]) -> Pointer[Float32, MutUntrackedOrigin]:
    # Device buffer pointers only ever flow into kernel launches (where
    # origins are erased) or pointer arithmetic here, so the stronger
    # device origin adds nothing; rebind to the plain untracked form.
    return rebind[Pointer[Float32, MutUntrackedOrigin]](buf.unsafe_ptr())


@always_inline
def devp_i32(buf: DeviceBuffer[DType.int32]) -> Pointer[Int32, MutUntrackedOrigin]:
    return rebind[Pointer[Int32, MutUntrackedOrigin]](buf.unsafe_ptr())


# =============================================================================
# device kernels specific to TabM's BatchEnsemble blocks
#
# Row index e over a (b*k, width) member-major tensor maps to
# row = e // width, col = e % width, member j = row % k, batch bb = row // k.
# Gradients shared across members (dr, dhw, db_grad, ds_grad) use one thread
# per (j, col) slot looping over batch rows to avoid write races.
# =============================================================================


def dk_bcast_mul(
    dst: Pointer[Float32, MutUntrackedOrigin],
    src: Pointer[Float32, MutUntrackedOrigin],
    rmat: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    b_in: Int32,
    k: Int32,
    first_block: Int32,
    d_in: Int32,
):
    # v[row,t] = src[src_row,t] * r[jj,t]; first_block indexes src by batch
    # row (input h), later blocks by member row (previous activation).
    var e = gpu_global_idx()
    if e < Int(members) * Int(b_in):
        var row = e // Int(b_in)
        var jj = row % Int(k)
        var srow = (row // Int(k)) if first_block == 1 else row
        var soff = (srow * Int(d_in)) if first_block == 1 else (srow * Int(b_in))
        dst[e] = src[soff + (e % Int(b_in))] * rmat[jj * Int(b_in) + (e % Int(b_in))]


def dk_rowwise_mul_s(
    dq: Pointer[Float32, MutUntrackedOrigin],
    s: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    db: Int32,
    k: Int32,
):
    # dq[row,o] *= s[jj,o]
    var e = gpu_global_idx()
    if e < Int(members) * Int(db):
        var j = (e // Int(db)) % Int(k)
        dq[e] *= s[j * Int(db) + (e % Int(db))]


def dk_tabm_act(
    q: Pointer[Float32, MutUntrackedOrigin],
    s: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    rmask: Pointer[Float32, MutUntrackedOrigin],
    dmask: Pointer[Float32, MutUntrackedOrigin],
    act: Pointer[Float32, MutUntrackedOrigin],
    total: Int32,
    width: Int32,
    k: Int32,
    keep: Float32,
    thresh: UInt64,
    use_dropout: Int32,
    round_seed: Int64,
):
    # u = q*s + b ; act = relu(u)*mask ; rmask/dmask cached for backward.
    var e = gpu_global_idx()
    if e < Int(total):
        var o = e % Int(width)
        var j = (e // Int(width)) % Int(k)
        var u = q[e] * s[j * Int(width) + o] + bias[j * Int(width) + o]
        var keepm = True
        if use_dropout == 1:
            keepm = dropout_keep(e, round_seed, thresh)
        var dm: Float32 = keep
        if not keepm:
            dm = 0.0
        var pos = u > 0.0
        var ru: Float32 = u
        if not pos:
            ru = 0.0
        var rm: Float32 = 1.0
        if not pos:
            rm = 0.0
        rmask[e] = rm
        dmask[e] = dm
        act[e] = ru * dm


def dk_head1(
    preds: Pointer[Float32, MutUntrackedOrigin],
    act_last: Pointer[Float32, MutUntrackedOrigin],
    hw: Pointer[Float32, MutUntrackedOrigin],
    hb: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    db: Int32,
    k: Int32,
):
    # dout == 1: preds[row] = dot(act_last[row], hw[j]) + hb[j]
    var row = gpu_global_idx()
    if row < Int(members):
        var j = row % Int(k)
        var acc = hb[j]
        var ap = act_last + row * Int(db)
        var wp = hw + j * Int(db)
        for t in range(Int(db)):
            acc += ap[t] * wp[t]
        preds[row] = acc


def dk_headn(
    preds: Pointer[Float32, MutUntrackedOrigin],
    act_last: Pointer[Float32, MutUntrackedOrigin],
    hw: Pointer[Float32, MutUntrackedOrigin],
    hb: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    dout: Int32,
    db: Int32,
    k: Int32,
):
    # preds[row, o] = sum_t act_last[row,t]*hw[j,t,o] + hb[j,o]
    var idx = gpu_global_idx()
    if idx < Int(members) * Int(dout):
        var row = idx // Int(dout)
        var o = idx % Int(dout)
        var j = row % Int(k)
        var acc = hb[j * Int(dout) + o]
        var ap = act_last + row * Int(db)
        var wp = hw + j * Int(db) * Int(dout) + o
        for t in range(Int(db)):
            acc += ap[t] * wp[t * Int(dout)]
        preds[idx] = acc


def dk_tabm_loss(
    preds: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    loss_buf: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    k: Int32,
    dout: Int32,
    task: Int32,
    gmembers: Int32,
):
    # One thread per member row: fills dpreds + per-row loss.
    # task: 0 regression, 1 binary, 2 multiclass. gmembers = denom_b*k.
    var row = gpu_global_idx()
    if row < Int(members):
        var src = Int(idx[row // Int(k)])
        var lrow: Float32 = 0.0
        if task == 0:
            var total = Int(gmembers) * Int(dout)
            for o in range(Int(dout)):
                var diff = preds[row * Int(dout) + o] - y[src * Int(dout) + o]
                lrow += diff * diff
                dpreds[row * Int(dout) + o] = (2.0 / Float32(total)) * diff
        elif task == 1:
            var z = preds[row * Int(dout)]
            var target = y[src * Int(dout)]
            lrow += gpu_softplus(z) - target * z
            dpreds[row * Int(dout)] = (gpu_sigmoid(z) - target) / Float32(gmembers)
        else:
            # softmax cross-entropy over dout logits
            var base = row * Int(dout)
            var mx = preds[base]
            for o in range(1, Int(dout)):
                if preds[base + o] > mx:
                    mx = preds[base + o]
            var lse: Float32 = 0.0
            for o in range(Int(dout)):
                var ev = exp(preds[base + o] - mx)
                dpreds[base + o] = ev
                lse += ev
            var inv = 1.0 / lse
            var cls = Int(y[src])
            lrow -= preds[base + cls] - mx - log(lse)
            for o in range(Int(dout)):
                var sm = dpreds[base + o] * inv
                var onehot: Float32 = 1.0 if o == cls else 0.0
                dpreds[base + o] = (sm - onehot) / Float32(gmembers)
        loss_buf[row] = lrow


def dk_head_da_dout1(
    da: Pointer[Float32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    hw: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    db: Int32,
    k: Int32,
):
    # da[row,t] = dpred[row]*hw[j,t]
    var e = gpu_global_idx()
    if e < Int(members) * Int(db):
        var row = e // Int(db)
        var j = row % Int(k)
        da[e] = dpreds[row] * hw[j * Int(db) + (e % Int(db))]


def dk_head_wgrad_dout1(
    dhw: Pointer[Float32, MutUntrackedOrigin],
    act_last: Pointer[Float32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    db: Int32,
    k: Int32,
):
    # dhw[j,t] += sum_bb dpred[bb*k+j]*act[(bb*k+j)*db+t]
    var idx = gpu_global_idx()
    if idx < Int(k) * Int(db):
        var bb = Int(b)
        var acc: Float32 = 0.0
        for i in range(bb):
            var row = i * Int(k) + idx // Int(db)
            acc += dpreds[row] * act_last[row * Int(db) + (idx % Int(db))]
        dhw[idx] += acc


def dk_head_bgrad_dout1(
    dhb: Pointer[Float32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    k: Int32,
):
    var j = gpu_global_idx()
    if j < Int(k):
        var acc: Float32 = 0.0
        for i in range(Int(b)):
            acc += dpreds[i * Int(k) + j]
        dhb[j] += acc


def dk_head_da(
    da: Pointer[Float32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    hw: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    dout: Int32,
    db: Int32,
    k: Int32,
):
    # da[row,t] = sum_o dpred[row,o]*hw[j,t,o]
    var idx = gpu_global_idx()
    if idx < Int(members) * Int(db):
        var row = idx // Int(db)
        var t = idx % Int(db)
        var j = row % Int(k)
        var acc: Float32 = 0.0
        var wp = hw + j * Int(db) * Int(dout) + t * Int(dout)
        for o in range(Int(dout)):
            acc += dpreds[row * Int(dout) + o] * wp[o]
        da[idx] = acc


def dk_head_wgrad(
    dhw: Pointer[Float32, MutUntrackedOrigin],
    act_last: Pointer[Float32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    dout: Int32,
    db: Int32,
    k: Int32,
):
    # dhw[j,t,o] += sum_bb act[(bb,j,t)]*dpred[(bb,j,o)]
    var idx = gpu_global_idx()
    if idx < Int(k) * Int(db) * Int(dout):
        var t = (idx // Int(dout)) % Int(db)
        var j = idx // (Int(db) * Int(dout))
        var acc: Float32 = 0.0
        for i in range(Int(b)):
            var row = i * Int(k) + j
            acc += act_last[row * Int(db) + t] * dpreds[row * Int(dout) + (idx % Int(dout))]
        dhw[idx] += acc


def dk_head_bgrad(
    dhb: Pointer[Float32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    dout: Int32,
    k: Int32,
):
    var idx = gpu_global_idx()
    if idx < Int(k) * Int(dout):
        var acc: Float32 = 0.0
        for i in range(Int(b)):
            acc += dpreds[i * Int(k) * Int(dout) + idx]
        dhb[idx] += acc


def dk_bs_grads(
    db_grad: Pointer[Float32, MutUntrackedOrigin],
    ds_grad: Pointer[Float32, MutUntrackedOrigin],
    dq: Pointer[Float32, MutUntrackedOrigin],
    q: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    db: Int32,
    k: Int32,
):
    # db_grad[j,o] += sum_bb dq ; ds_grad[j,o] += sum_bb dq*q
    var idx = gpu_global_idx()
    if idx < Int(k) * Int(db):
        var acc_b: Float32 = 0.0
        var acc_s: Float32 = 0.0
        for i in range(Int(b)):
            var row = i * Int(k) + idx // Int(db)
            var dv = dq[row * Int(db) + (idx % Int(db))]
            acc_b += dv
            acc_s += dv * q[row * Int(db) + (idx % Int(db))]
        db_grad[idx] += acc_b
        ds_grad[idx] += acc_s


def dk_dr_acc(
    dr: Pointer[Float32, MutUntrackedOrigin],
    dv: Pointer[Float32, MutUntrackedOrigin],
    xin: Pointer[Float32, MutUntrackedOrigin],
    h: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    b_in: Int32,
    d_in: Int32,
    db: Int32,
    k: Int32,
    first_block: Int32,
):
    # dr[jj,t] += sum_bb dv[(bb,jj),t]*xin[(bb or bb*jj),t]
    var idx = gpu_global_idx()
    if idx < Int(k) * Int(b_in):
        var jj = idx // Int(b_in)
        var t = idx % Int(b_in)
        var acc: Float32 = 0.0
        for i in range(Int(b)):
            if first_block == 1:
                acc += dv[(i * Int(k) + jj) * Int(b_in) + t] * h[i * Int(d_in) + t]
            else:
                acc += dv[(i * Int(k) + jj) * Int(b_in) + t] * xin[(i * Int(k) + jj) * Int(db) + t]
        dr[idx] += acc


def dk_da_new(
    da_new: Pointer[Float32, MutUntrackedOrigin],
    dv: Pointer[Float32, MutUntrackedOrigin],
    rr: Pointer[Float32, MutUntrackedOrigin],
    members: Int32,
    b_in: Int32,
    k: Int32,
):
    # da_new[row,t] = dv[row,t]*r[jj,t]
    var e = gpu_global_idx()
    if e < Int(members) * Int(b_in):
        var j = (e // Int(b_in)) % Int(k)
        da_new[e] = dv[e] * rr[j * Int(b_in) + (e % Int(b_in))]


def dk_member_reduce(
    dh: Pointer[Float32, MutUntrackedOrigin],
    da_new: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    d_in: Int32,
    k: Int32,
):
    # dh[bb,t] = sum_j da_new[(bb,j),t]
    var idx = gpu_global_idx()
    if idx < Int(b) * Int(d_in):
        var acc: Float32 = 0.0
        for j in range(Int(k)):
            acc += da_new[(idx // Int(d_in) * Int(k) + j) * Int(d_in) + (idx % Int(d_in))]
        dh[idx] = acc


# =============================================================================
# host RNG (xorshift64*) -- drives the Fisher-Yates shuffle exactly like the
# CPU kernels so epoch permutations match across backends.
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


@always_inline
def grid_for(n: Int) -> Int:
    return cdiv(n, GTPB)



@always_inline
def dot_f64(a: Pointer[Float64, MutUntrackedOrigin], b: Pointer[Float64, MutUntrackedOrigin], n: Int) -> Float64:
    var acc: Float64 = 0.0
    for i in range(n):
        acc += a[i] * b[i]
    return acc


@always_inline
def max_abs_host(a: Pointer[Float32, MutUntrackedOrigin], n: Int) -> Float64:
    var best: Float64 = 0.0
    for i in range(n):
        var v = a[i]
        var av = Float64(v if v > 0.0 else -v)
        if av > best:
            best = av
    return best


def free_lbfgs(
    g: Pointer[Float32, MutUntrackedOrigin],
    gnew: Pointer[Float32, MutUntrackedOrigin],
    cand: Pointer[Float32, MutUntrackedOrigin],
    th64: Pointer[Float64, MutUntrackedOrigin],
    g64: Pointer[Float64, MutUntrackedOrigin],
    gnew64: Pointer[Float64, MutUntrackedOrigin],
    cand64: Pointer[Float64, MutUntrackedOrigin],
    dir: Pointer[Float64, MutUntrackedOrigin],
    S: Pointer[Float64, MutUntrackedOrigin],
    Y: Pointer[Float64, MutUntrackedOrigin],
    rho: Pointer[Float64, MutUntrackedOrigin],
    alphas: Pointer[Float64, MutUntrackedOrigin],
):
    g.unsafe_free()
    gnew.unsafe_free()
    cand.unsafe_free()
    th64.unsafe_free()
    g64.unsafe_free()
    gnew64.unsafe_free()
    cand64.unsafe_free()
    dir.unsafe_free()
    S.unsafe_free()
    Y.unsafe_free()
    rho.unsafe_free()
    alphas.unsafe_free()


def dk_dq_prod(
    dq: Pointer[Float32, MutUntrackedOrigin],
    da: Pointer[Float32, MutUntrackedOrigin],
    dmask: Pointer[Float32, MutUntrackedOrigin],
    rmask: Pointer[Float32, MutUntrackedOrigin],
    total: Int32,
):
    # dq = da * dmask * rmask
    var e = gpu_global_idx()
    if e < Int(total):
        dq[e] = da[e] * dmask[e] * rmask[e]


# =============================================================================
# device round orchestration
# =============================================================================


def gpu_forward(
    tr: TabMTrainer,
    mut ctx: DeviceContext,
    theta: Pointer[Float32, MutUntrackedOrigin],
    ws: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    b: Int32,
    dropout: Float32,
    train: Bool,
    round_seed: Int64,
) raises:
    """Forward pass for b consecutive rows of the shuffled index."""
    var bi = Int(b)
    var bk = bi * tr.k
    var fdemb = tr.F * tr.demb
    var use_drop = train and dropout > 0.0
    var thresh = UInt64(dropout * 256.0)
    if thresh < 1:
        thresh = 1
    var keep = 1.0 / (1.0 - dropout)

    # ---- inputs ---------------------------------------------------------
    if tr.use_emb and tr.F > 0:
        ctx.enqueue_function[dk_gather_rows](
            ws + tr.o_xnum, x_num, idx, b, Int32(tr.F),
            grid_dim=grid_for(bi * tr.F), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_gather_rows](
            ws + tr.o_enc, x_enc, idx, b, Int32(tr.denc),
            grid_dim=grid_for(bi * tr.denc), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_embed_linear](
            ws, x_num, idx, theta + tr.off_emb_w0(), theta + tr.off_emb_b0(),
            b, Int32(tr.F), Int32(tr.demb), Int32(tr.d_in),
            grid_dim=grid_for(bi * fdemb), block_dim=GTPB,
        )
        if tr.ccat > 0:
            ctx.enqueue_function[dk_cat_copy](
                ws, x_cat, idx, b, Int32(tr.ccat), Int32(fdemb), Int32(tr.d_in),
                grid_dim=grid_for(bi * tr.ccat), block_dim=GTPB,
            )
        for f in range(tr.F):
            var cnt = tr.bin_counts[f]
            if cnt > 0:
                ctx.enqueue_function[dk_gather_cols_idx](
                    ws + tr.o_tmp, x_enc, idx, b, Int32(tr.denc),
                    Int32(tr.bin_offsets[f]), Int32(cnt),
                    grid_dim=grid_for(bi * cnt), block_dim=GTPB,
                )
                ctx.enqueue_function[dk_gemm_nn](
                    ws + tr.o_pl + f * tr.demb, ws + tr.o_tmp,
                    theta + tr.off_emb_wp(f),
                    Int32(bi), Int32(tr.demb), Int32(cnt), Int32(fdemb),
                    grid_dim=grid_for(bi * tr.demb), block_dim=GTPB,
                )
        ctx.enqueue_function[dk_relu_add](
            ws, ws + tr.o_pl, b, Int32(fdemb), Int32(tr.d_in),
            grid_dim=grid_for(bi * fdemb), block_dim=GTPB,
        )
    else:
        ctx.enqueue_function[dk_input_copy](
            ws, x_num, x_cat, idx, b, Int32(tr.F), Int32(tr.ccat),
            grid_dim=grid_for(bi * tr.d_in), block_dim=GTPB,
        )

    # ---- backbone blocks --------------------------------------------------
    var i = 0
    while i < tr.nb:
        var b_in = tr.block_in(i)
        var src = ws if i == 0 else ws + tr.blk(i - 1, 4)
        ctx.enqueue_function[dk_bcast_mul](
            ws + tr.blk(i, 0), src, theta + tr.off_r(i),
            Int32(bk), Int32(b_in), Int32(tr.k),
            Int32(1 if i == 0 else 0), Int32(tr.d_in),
            grid_dim=grid_for(bk * b_in), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_gemm_nt](
            ws + tr.blk(i, 1), ws + tr.blk(i, 0), theta + tr.off_w(i),
            Int32(bk), Int32(tr.db), Int32(b_in), Int32(tr.db),
            grid_dim=grid_for(bk * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_tabm_act](
            ws + tr.blk(i, 1), theta + tr.off_s(i), theta + tr.off_b(i),
            ws + tr.blk(i, 2), ws + tr.blk(i, 3), ws + tr.blk(i, 4),
            Int32(bk * tr.db), Int32(tr.db), Int32(tr.k),
            keep, thresh, Int32(1 if use_drop else 0), round_seed,
            grid_dim=grid_for(bk * tr.db), block_dim=GTPB,
        )
        ctx.synchronize()  # DEBUG
        i += 1

    # ---- head --------------------------------------------------------------
    var act_last = ws + tr.blk(tr.nb - 1, 4)
    var hw = theta + tr.off_head_w()
    var hb = theta + tr.off_head_b()
    if tr.dout == 1:
        ctx.enqueue_function[dk_head1](
            ws + tr.o_preds, act_last, hw, hb,
            Int32(bk), Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(bk), block_dim=GTPB,
        )
    else:
        ctx.enqueue_function[dk_headn](
            ws + tr.o_preds, act_last, hw, hb,
            Int32(bk), Int32(tr.dout), Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(bk * tr.dout), block_dim=GTPB,
        )


def gpu_round(
    tr: TabMTrainer,
    mut ctx: DeviceContext,
    theta: Pointer[Float32, MutUntrackedOrigin],
    ws: Pointer[Float32, MutUntrackedOrigin],
    g: Pointer[Float32, MutUntrackedOrigin],
    partials_buf: DeviceBuffer[DType.float32],
    partials_host: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    b: Int32,
    task: Int32,
    denom_b: Int32,
    dropout: Float32,
    train: Bool,
    round_seed: Int64,
) raises -> Float64:
    """Forward -> loss/dpreds -> backward; returns the round's loss in the
    same units as the CPU ``run_round`` (sum of per-slice means)."""
    var bi = Int(b)
    var bk = bi * tr.k
    var fdemb = tr.F * tr.demb

    gpu_forward(tr, ctx, theta, ws, x_num, x_enc, x_cat, idx, b, dropout, train, round_seed)

    # ---- loss + dpreds ------------------------------------------------------
    ctx.enqueue_function[dk_tabm_loss](
        ws + tr.o_preds, y, idx, ws + tr.o_dpreds, ws + tr.o_loss,
        Int32(bk), Int32(tr.k), Int32(tr.dout), task, Int32(Int(denom_b) * tr.k),
        grid_dim=grid_for(bk), block_dim=GTPB,
    )
    ctx.enqueue_function[dk_sum](
        ws + tr.o_loss, devp(partials_buf), Int32(bk),
        grid_dim=grid_for(bk), block_dim=GTPB,
    )

    # ---- backward -----------------------------------------------------------
    var act_last = ws + tr.blk(tr.nb - 1, 4)
    var hw = theta + tr.off_head_w()
    var dhw = g + tr.off_head_w()
    var dhb = g + tr.off_head_b()
    var da = ws + tr.o_da
    if tr.dout == 1:
        ctx.enqueue_function[dk_head_da_dout1](
            da, ws + tr.o_dpreds, hw, Int32(bk), Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(bk * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_head_wgrad_dout1](
            dhw, act_last, ws + tr.o_dpreds, b, Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(tr.k * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_head_bgrad_dout1](
            dhb, ws + tr.o_dpreds, b, Int32(tr.k),
            grid_dim=grid_for(tr.k), block_dim=GTPB,
        )
    else:
        ctx.enqueue_function[dk_head_da](
            da, ws + tr.o_dpreds, hw, Int32(bk), Int32(tr.dout), Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(bk * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_head_wgrad](
            dhw, act_last, ws + tr.o_dpreds, b, Int32(tr.dout), Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(tr.k * tr.db * tr.dout), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_head_bgrad](
            dhb, ws + tr.o_dpreds, b, Int32(tr.dout), Int32(tr.k),
            grid_dim=grid_for(tr.k * tr.dout), block_dim=GTPB,
        )

    var i = tr.nb - 1
    while i >= 0:
        var b_in = tr.block_in(i)
        var dq = ws + tr.o_dq
        var dv = ws + tr.o_dv
        ctx.enqueue_function[dk_dq_prod](
            dq, da, ws + tr.blk(i, 3), ws + tr.blk(i, 2), Int32(bk * tr.db),
            grid_dim=grid_for(bk * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_bs_grads](
            g + tr.off_b(i), g + tr.off_s(i), dq, ws + tr.blk(i, 1),
            b, Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(tr.k * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_rowwise_mul_s](
            dq, theta + tr.off_s(i), Int32(bk), Int32(tr.db), Int32(tr.k),
            grid_dim=grid_for(bk * tr.db), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_gemm_tn_acc](
            g + tr.off_w(i), dq, ws + tr.blk(i, 0),
            Int32(tr.db), Int32(b_in), Int32(bk), Int32(b_in),
            grid_dim=grid_for(tr.db * b_in), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_gemm_nn](
            dv, dq, theta + tr.off_w(i),
            Int32(bk), Int32(b_in), Int32(tr.db), Int32(b_in),
            grid_dim=grid_for(bk * b_in), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_dr_acc](
            g + tr.off_r(i), dv, ws + tr.blk(i - 1, 4) if i > 0 else ws,
            ws, b, Int32(b_in), Int32(tr.d_in), Int32(tr.db), Int32(tr.k),
            Int32(1 if i == 0 else 0),
            grid_dim=grid_for(tr.k * b_in), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_da_new](
            da, dv, theta + tr.off_r(i), Int32(bk), Int32(b_in), Int32(tr.k),
            grid_dim=grid_for(bk * b_in), block_dim=GTPB,
        )
        if i == 0:
            ctx.enqueue_function[dk_member_reduce](
                ws + tr.o_dh, da, b, Int32(tr.d_in), Int32(tr.k),
                grid_dim=grid_for(bi * tr.d_in), block_dim=GTPB,
            )
        i -= 1

    if tr.use_emb and tr.F > 0:
        ctx.enqueue_function[dk_emb_wb_grad](
            g + tr.off_emb_w0(), g + tr.off_emb_b0(),
            ws + tr.o_xnum, ws + tr.o_dh,
            b, Int32(tr.F), Int32(tr.demb), Int32(tr.d_in),
            grid_dim=grid_for(fdemb), block_dim=GTPB,
        )
        for f in range(tr.F):
            var cnt = tr.bin_counts[f]
            if cnt > 0:
                ctx.enqueue_function[dk_slice_cols](
                    ws + tr.o_tmp, ws + tr.o_enc, b, Int32(tr.denc),
                    Int32(tr.bin_offsets[f]), Int32(cnt),
                    grid_dim=grid_for(bi * cnt), block_dim=GTPB,
                )
                ctx.enqueue_function[dk_dpl](
                    ws + tr.o_tmp2, ws + tr.o_pl, ws + tr.o_dh,
                    b, Int32(f), Int32(tr.F), Int32(tr.demb), Int32(tr.d_in),
                    grid_dim=grid_for(bi * tr.demb), block_dim=GTPB,
                )
                ctx.enqueue_function[dk_gemm_tn_acc](
                    g + tr.off_emb_wp(f), ws + tr.o_tmp, ws + tr.o_tmp2,
                    Int32(cnt), Int32(tr.demb), Int32(bi), Int32(tr.demb),
                    grid_dim=grid_for(cnt * tr.demb), block_dim=GTPB,
                )

    # ---- loss readback -------------------------------------------------------
    var npart = grid_for(bk) * GTPB
    ctx.synchronize()  # dk_sum must finish before the readback
    ctx.enqueue_copy(partials_host, partials_buf)
    ctx.synchronize()  # partials are ready; host sums in float64
    var total: Float64 = 0.0
    for t in range(npart):
        total += Float64(partials_host[t])
    # Match CPU run_round semantics: task 0 returns sum(diff^2)/(k*dout)
    # (row-count independent); other tasks return the raw row-loss sum so
    # the caller's division by the minibatch row count yields a mean.
    if task == 0:
        return total / (Float64(tr.k) * Float64(tr.dout))
    return total


def gpu_eval(
    tr: TabMTrainer,
    mut ctx: DeviceContext,
    cand_host: Pointer[Float32, MutUntrackedOrigin],
    g_out: Pointer[Float32, MutUntrackedOrigin],
    write_g: Bool,
    add_l2: Bool,
    alpha: Float32,
    ws: Pointer[Float32, MutUntrackedOrigin],
    partials_buf: DeviceBuffer[DType.float32],
    partials_host: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    N: Int,
    task: Int32,
) raises -> Float64:
    """Full-batch loss (+ gradient into g_out); eval mode, no dropout."""
    var theta_dev = up_f32(ctx, cand_host, tr.P)
    var tp = devp(theta_dev)
    var g_dev = ctx.enqueue_create_buffer[DType.float32](tr.P)
    var gp = devp(g_dev)
    ctx.enqueue_function[dk_zero](
        gp, Int32(tr.P), grid_dim=grid_for(tr.P), block_dim=GTPB
    )
    var weighted: Float64 = 0.0
    var start = 0
    var rnd = 0
    while start < N:
        var b = min(tr.chunk, N - start)
        weighted += gpu_round(
            tr, ctx, tp, ws, gp, partials_buf, partials_host,
            x_num, x_enc, x_cat, y, idx + start,
            Int32(b), task, Int32(b), 0.0, False, Int64(mix_u64(UInt64(rnd))),
        )
        start += b
        rnd += 1
    var loss = weighted / Float64(N)
    if add_l2:
        ctx.enqueue_function[dk_sum_sq](
            tp, devp(partials_buf), Int32(tr.P),
            grid_dim=grid_for(tr.P), block_dim=GTPB,
        )
        ctx.synchronize()
        ctx.enqueue_copy(partials_host, partials_buf)
        ctx.synchronize()
        var ss: Float64 = 0.0
        for t in range(grid_for(tr.P) * GTPB):
            ss += Float64(partials_host[t])
        loss += 0.5 * Float64(alpha) * ss
        ctx.enqueue_function[dk_saxpy](
            gp, tp, alpha, Int32(tr.P),
            grid_dim=grid_for(tr.P), block_dim=GTPB,
        )
    if write_g:
        ctx.enqueue_copy(g_out, g_dev)
    return loss


def gpu_adam_epoch(
    tr: TabMTrainer,
    mut ctx: DeviceContext,
    theta_host: Pointer[Float32, MutUntrackedOrigin],
    m_host: Pointer[Float32, MutUntrackedOrigin],
    v_host: Pointer[Float32, MutUntrackedOrigin],
    t0: Int,
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx_dev: Pointer[Int32, MutUntrackedOrigin],
    N: Int,
    lr: Float32,
    bs: Int,
    dropout: Float32,
    alpha: Float32,
    seed: UInt64,
    task: Int32,
    ws: Pointer[Float32, MutUntrackedOrigin],
    partials_buf: DeviceBuffer[DType.float32],
    partials_host: Pointer[Float32, MutUntrackedOrigin],
) raises -> Tuple[Float64, Int]:
    """One shuffled minibatch Adam epoch; theta/m/v updated in place (host)."""
    var theta_dev = up_f32(ctx, theta_host, tr.P)
    var m_dev = up_f32(ctx, m_host, tr.P)
    var v_dev = up_f32(ctx, v_host, tr.P)
    var g_dev = ctx.enqueue_create_buffer[DType.float32](tr.P)
    var tp = devp(theta_dev)
    var mp = devp(m_dev)
    var vp = devp(v_dev)
    var gp = devp(g_dev)

    # L2 term uses the pre-update theta, matching the CPU kernels.
    ctx.enqueue_function[dk_sum_sq](
        tp, devp(partials_buf), Int32(tr.P),
        grid_dim=grid_for(tr.P), block_dim=GTPB,
    )
    ctx.synchronize()
    ctx.enqueue_copy(partials_host, partials_buf)
    ctx.synchronize()
    var ss: Float64 = 0.0
    for t in range(grid_for(tr.P) * GTPB):
        ss += Float64(partials_host[t])
    var l2_term = 0.5 * Float64(alpha) * ss

    var weighted: Float64 = 0.0
    var t = t0
    var start = 0
    var mb = 0
    while start < N:
        var be = min(start + bs, N)
        var nb_rows = be - start
        ctx.enqueue_function[dk_zero](
            g_dev, Int32(tr.P), grid_dim=grid_for(tr.P), block_dim=GTPB
        )
        var batch_loss: Float64 = 0.0
        var c0 = start
        while c0 < be:
            var rb = min(tr.chunk, be - c0)
            batch_loss += gpu_round(
                tr, ctx, tp, ws, gp, partials_buf, partials_host,
                x_num, x_enc, x_cat, y, idx_dev + c0,
                Int32(rb), task, Int32(rb), dropout, True,
                Int64(mix_u64(seed ^ mix_u64(UInt64(mb) * 2654435761 + UInt64(c0)))),
            )
            c0 += rb
        mb += 1
        batch_loss = batch_loss / Float64(nb_rows) + l2_term
        t += 1
        ctx.enqueue_function[dk_adam_update](
            tp, mp, vp, gp, Int32(tr.P), alpha, lr,
            Float32(1.0 - pow(0.9, Float64(t))),
            Float32(1.0 - pow(0.999, Float64(t))),
            grid_dim=grid_for(tr.P), block_dim=GTPB,
        )
        weighted += batch_loss * Float64(nb_rows)
        start = be

    ctx.enqueue_copy(theta_host, theta_dev)
    ctx.enqueue_copy(m_host, m_dev)
    ctx.enqueue_copy(v_host, v_dev)
    return (weighted / Float64(N), t)


def gpu_lbfgs(
    tr: TabMTrainer,
    mut ctx: DeviceContext,
    theta_host: Pointer[Float32, MutUntrackedOrigin],
    x_num: Pointer[Float32, MutUntrackedOrigin],
    x_enc: Pointer[Float32, MutUntrackedOrigin],
    x_cat: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx_dev: Pointer[Int32, MutUntrackedOrigin],
    N: Int,
    max_iter: Int,
    tol: Float64,
    maxcor: Int,
    alpha: Float32,
    losses_out: Pointer[Float64, MutUntrackedOrigin],
    task: Int32,
    ws: Pointer[Float32, MutUntrackedOrigin],
    partials_buf: DeviceBuffer[DType.float32],
    partials_host: Pointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    """Full-batch L-BFGS with backtracking line search.

    Model state and bookkeeping stay on the host in float64 (mirroring the
    CPU kernels); every function evaluation runs on the GPU via gpu_eval.
    """
    var P = tr.P
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

    var loss = gpu_eval(
        tr, ctx, theta_host, g, True, True, alpha,
        ws, partials_buf, partials_host,
        x_num, x_enc, x_cat, y, idx_dev, N, task,
    )
    losses_out[0] = loss
    var n_losses = 1
    var i = 0
    while i < P:
        th64[i] = Float64(theta_host[i])
        g64[i] = Float64(g[i])
        i += 1
    if max_abs_host(g, P) <= tol:
        free_lbfgs(g, gnew, cand, th64, g64, gnew64, cand64, dir, S, Y, rho, alphas)
        return 0

    var nhist = 0
    var hptr = 0
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
            var loss_new = gpu_eval(
                tr, ctx, cand, gnew, True, True, alpha,
                ws, partials_buf, partials_host,
                x_num, x_enc, x_cat, y, idx_dev, N, task,
            )
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
                    theta_host[i] = cand[i]
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
        if max_abs_host(g, P) <= tol:
            break
        it += 1

    var result = n_losses - 1
    free_lbfgs(g, gnew, cand, th64, g64, gnew64, cand64, dir, S, Y, rho, alphas)
    return result

@export
def PyInit__native_tabm_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native_tabm_gpu")
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
        abort(String("failed to create module _native_tabm_gpu: ", e))


# =============================================================================
# Trainer
# =============================================================================


struct TabMTrainer(ImplicitlyCopyable, Movable, Writable):
    def write_to(mut self, mut writer: Some[Writer]):
        writer.write("TabMGPUTrainer(P=", self.P, ")")

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
    var chunk: Int
    var bin_counts: Pointer[Int, MutUntrackedOrigin]
    var bin_offsets: Pointer[Int, MutUntrackedOrigin]
    var offs: Pointer[Int, MutUntrackedOrigin]
    var n_offs: Int
    var P: Int
    # workspace slab offsets (elements) for the fixed chunk size; o_h == 0
    var o_pl: Int
    var o_tmp: Int
    var o_tmp2: Int
    var o_xnum: Int
    var o_enc: Int
    var o_preds: Int
    var o_dpreds: Int
    var o_loss: Int
    var o_da: Int
    var o_dq: Int
    var o_dv: Int
    var o_dh: Int
    var o_end: Int
    var boffs: Pointer[Int, MutUntrackedOrigin]  # nb * 6 per-block offsets
    var warmed: Bool

    @always_inline
    def blk(self, i: Int, which: Int) -> Int:
        # which: 0 vs, 1 qs, 2 rmask, 3 dmask, 4 act
        return self.boffs[6 * i + which]

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

        self.chunk = max(1, 8192 // self.k)
        var lb = self.chunk
        var lbk = lb * self.k
        var lfdemb = self.F * self.demb
        var lbmax = max(self.d_in, self.db)
        cur = 0
        cur += lb * self.d_in + 16  # h at offset 0
        self.o_pl = cur
        cur += lb * lfdemb + 16
        self.o_tmp = cur
        cur += lb * max(max(lfdemb, self.d_in), self.denc) + 16
        self.o_tmp2 = cur
        cur += lb * self.demb + 16
        self.o_xnum = cur
        cur += lb * self.F + 16
        self.o_enc = cur
        cur += lb * self.denc + 16
        self.o_preds = cur
        cur += lbk * self.dout + 16
        self.o_dpreds = cur
        cur += lbk * self.dout + 16
        self.o_loss = cur
        cur += lbk + 16
        self.o_da = cur
        cur += lbk * lbmax + 16
        self.o_dq = cur
        cur += lbk * self.db + 16
        self.o_dv = cur
        cur += lbk * lbmax + 16
        self.o_dh = cur
        cur += lb * self.d_in + 16
        self.boffs = alloc[Int](self.nb * 6 + 16)
        var lw = 0
        for i in range(self.nb):
            var b_in = self.d_in if i == 0 else self.db
            self.boffs[lw] = cur
            cur += lbk * b_in + 16  # vs
            lw += 1
            self.boffs[lw] = cur
            cur += lbk * self.db + 16  # qs
            lw += 1
            self.boffs[lw] = cur
            cur += lbk * self.db + 16  # rmask
            lw += 1
            self.boffs[lw] = cur
            cur += lbk * self.db + 16  # dmask
            lw += 1
            self.boffs[lw] = cur
            cur += lbk * self.db + 16  # act
            lw += 1
            self.boffs[lw] = cur
            lw += 1
        self.o_end = cur
        self.warmed = False


    def _warmup(mut self, mut ctx: DeviceContext) raises:
        """Compile every device pipeline and verify each kernel executes.

        Metal pipelines are created lazily on first launch; that creation is
        asynchronous and a launch issued while its pipeline is still being
        compiled can be silently dropped (observed as missing writes and as
        XPC_ERROR_CONNECTION_INTERRUPTED under shader validation). Every
        kernel therefore runs against a one-filled scratch buffer until its
        output slot differs from the fill value, retrying a few times.
        Later launches reuse cached pipelines on this trainer's long-lived
        context and are reliable.
        """
        var scratch = ctx.enqueue_create_buffer[DType.float32](96)
        var sp = devp(scratch)
        var sread = ctx.enqueue_create_buffer[DType.float32](96)
        var shost = alloc[Float32](96)
        var iscratch = ctx.enqueue_create_buffer[DType.int32](8)

        # deterministic seeds: indices zeroed, slots get DISTINCT fills so a
        # kernel's output can always be told apart from an unwritten slot
        var izero = alloc[Int32](8)
        for i in range(8):
            izero[i] = 0
        ctx.enqueue_copy(iscratch, izero)
        var pat_host = alloc[Float32](96)
        for i in range(96):
            pat_host[i] = Float32(i + 2) * 0.375
        var pat_buf = ctx.enqueue_create_buffer[DType.float32](96)
        ctx.enqueue_copy(pat_buf, pat_host)
        var ip = devp_i32(iscratch)
        ctx.synchronize()

        comptime NK = 30
        var done = alloc[Bool](NK)
        for i in range(NK):
            done[i] = True

        var rounds = 0
        var pending_last = NK
        while pending_last > 0 and rounds < 5:
            rounds += 1
            ctx.enqueue_copy(scratch, pat_buf)
            ctx.synchronize()

            if not done[0]:
                ctx.enqueue_function[dk_copy](sp + 0, sp + 40, Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[1]:
                ctx.enqueue_function[dk_accumulate](sp + 1, sp + 41, Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[2]:
                ctx.enqueue_function[dk_saxpy](sp + 2, sp + 42, Float32(3.0), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[3]:
                ctx.enqueue_function[dk_sum](sp + 43, sp + 3, Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[4]:
                ctx.enqueue_function[dk_sum_sq](sp + 44, sp + 4, Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[5]:
                ctx.enqueue_function[dk_adam_update](
                    sp + 5, sp + 45, sp + 46, sp + 47, Int32(1),
                    Float32(0.0), Float32(0.01), Float32(1.0), Float32(1.0),
                    grid_dim=1, block_dim=GTPB,
                )
            if not done[6]:
                ctx.enqueue_function[dk_gemm_nt](sp + 6, sp + 48, sp + 49, Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[7]:
                ctx.enqueue_function[dk_gemm_nn](sp + 7, sp + 48, sp + 49, Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[8]:
                ctx.enqueue_function[dk_gemm_tn_acc](sp + 8, sp + 48, sp + 49, Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[9]:
                ctx.enqueue_function[dk_gather_rows](sp + 9, sp + 50, ip, Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[10]:
                ctx.enqueue_function[dk_gather_cols_idx](sp + 10, sp + 50, ip, Int32(1), Int32(1), Int32(0), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[11]:
                ctx.enqueue_function[dk_slice_cols](sp + 11, sp + 50, Int32(1), Int32(1), Int32(0), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[12]:
                ctx.enqueue_function[dk_input_copy](sp + 12, sp + 51, sp + 51, ip, Int32(1), Int32(1), Int32(0), grid_dim=1, block_dim=GTPB)
            if not done[13]:
                ctx.enqueue_function[dk_cat_copy](sp + 13, sp + 51, ip, Int32(1), Int32(1), Int32(0), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[14]:
                ctx.enqueue_function[dk_embed_linear](sp + 14, sp + 52, ip, sp + 53, sp + 54, Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[15]:
                ctx.enqueue_function[dk_relu_add](sp + 15, sp + 55, Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[16]:
                ctx.enqueue_function[dk_emb_wb_grad](sp + 56, sp + 57, sp + 58, sp + 16, Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[17]:
                ctx.enqueue_function[dk_dpl](sp + 17, sp + 55, sp + 59, Int32(1), Int32(0), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[18]:
                ctx.enqueue_function[dk_bcast_mul](sp + 18, sp + 60, sp + 61, Int32(1), Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[19]:
                ctx.enqueue_function[dk_tabm_act](
                    sp + 62, sp + 20, sp + 21, sp + 22, sp + 23, sp + 19,
                    Int32(1), Int32(1), Int32(1), Float32(1.0), UInt64(256),
                    Int32(0), Int64(0), grid_dim=1, block_dim=GTPB,
                )
            if not done[20]:
                ctx.enqueue_function[dk_head1](sp + 24, sp + 25, sp + 26, sp + 27, Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[21]:
                ctx.enqueue_function[dk_headn](sp + 28, sp + 25, sp + 26, sp + 27, Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[22]:
                ctx.enqueue_function[dk_tabm_loss](
                    sp + 29, sp + 25, ip, sp + 30, sp + 31,
                    Int32(1), Int32(1), Int32(1), Int32(0), Int32(1),
                    grid_dim=1, block_dim=GTPB,
                )
            if not done[23]:
                ctx.enqueue_function[dk_dq_prod](sp + 32, sp + 33, sp + 34, sp + 35, Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[24]:
                ctx.enqueue_function[dk_bs_grads](sp + 36, sp + 37, sp + 38, sp + 39, Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[25]:
                ctx.enqueue_function[dk_rowwise_mul_s](sp + 40, sp + 41, Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[26]:
                ctx.enqueue_function[dk_dr_acc](sp + 42, sp + 43, sp + 44, sp + 45, Int32(1), Int32(1), Int32(1), Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[27]:
                ctx.enqueue_function[dk_da_new](sp + 46, sp + 47, sp + 48, Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[28]:
                ctx.enqueue_function[dk_member_reduce](sp + 49, sp + 50, Int32(1), Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[29]:
                ctx.enqueue_function[dk_zero](sp + 63, Int32(1), grid_dim=1, block_dim=GTPB)
            ctx.synchronize()

            # readback and mark executed slots (value changed away from 1.0);
            # pure-copy kernels whose source equals 1.0 are marked optimistically
            ctx.enqueue_function[dk_copy](
                devp(sread), sp, Int32(64), grid_dim=1, block_dim=GTPB,
            )
            ctx.enqueue_copy(shost, sread)
            ctx.synchronize()
            var nleft = 0
            for i in range(NK):
                if done[i]:
                    continue
                var want = Float32(i + 2) * 0.375
                if shost[i] != want:
                    done[i] = True
                else:
                    nleft += 1
            pending_last = nleft

        izero.unsafe_free()
        pat_host.unsafe_free()
        shost.unsafe_free()

    @staticmethod
    def py_init(out self: TabMTrainer, args: PythonObject, kwargs: PythonObject) raises:
        _ = kwargs
        if len(args) != 2:
            raise Error("TabMTrainer(dims, bins) expects 2 arguments")
        self = Self(args[0], args[1])

    # -- offset accessors (same as the CPU kernels) -----------------------------

    @always_inline
    def emb_count(self) -> Int:
        if self.use_emb and self.F > 0:
            return 2 + self.F
        return 0

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
    def off_emb_wp(self, f: Int) -> Int:
        return self.offs[2 + f]

    @always_inline
    def off_head_w(self) -> Int:
        return self.offs[self.n_offs - 2]

    @always_inline
    def off_head_b(self) -> Int:
        return self.offs[self.n_offs - 1]

    @always_inline
    def off_emb_w0(self) -> Int:
        return self.offs[0]

    @always_inline
    def off_emb_b0(self) -> Int:
        return self.offs[1]

    @always_inline
    def block_in(self, i: Int) -> Int:
        return self.d_in if i == 0 else self.db

    # -- Python-visible methods -----------------------------------------------------

    @staticmethod
    def _identity_idx(ctx: DeviceContext, N: Int) raises -> DeviceBuffer[DType.int32]:
        var idx_host = alloc[Int32](N)
        for i in range(N):
            idx_host[i] = Int32(i)
        var buf = up_i32(ctx, idx_host, N)
        ctx.synchronize()  # idx_host is about to be freed
        idx_host.unsafe_free()
        return buf

    @staticmethod
    def adam_epoch(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var ctx = DeviceContext()
        self._warmup(ctx)
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
        var task = Int32(Int(py=parts[13]))

        # Fisher-Yates on the same xorshift64* stream as the CPU kernels.
        var idx_host = alloc[Int32](N)
        for i in range(N):
            idx_host[i] = Int32(i)
        var rng = Rng(seed)
        var ii = N - 1
        while ii > 0:
            var jj = Int(rng.next_u64() % UInt64(ii + 1))
            var tmpi = idx_host[ii]
            idx_host[ii] = idx_host[jj]
            idx_host[jj] = tmpi
            ii -= 1
        var idx_dev = up_i32(ctx, idx_host, N)

        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[4]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[5]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[6]))
        var y_b = up_f32(ctx, y, N * iface_dim1(parts[7]))

        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var pmax = max(grid_for(self.P), grid_for(self.chunk * self.k)) * GTPB
        var partials_buf = ctx.enqueue_create_buffer[DType.float32](pmax)
        var partials_host = alloc[Float32](pmax)

        var res = gpu_adam_epoch(
            self, ctx, theta, m, v, t0,
            devp(xn_b), devp(xe_b), devp(xc_b), devp(y_b),
            devp_i32(idx_dev), N, lr, bs, dropout, alpha, seed, task,
            devp(ws_buf), partials_buf, partials_host,
        )
        ctx.synchronize()
        idx_host.unsafe_free()
        partials_host.unsafe_free()
        return Python.tuple(res[0], res[1])

    @staticmethod
    def lbfgs_minimize(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var ctx = DeviceContext()
        self._warmup(ctx)
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
        var task = Int32(Int(py=parts[10]))

        var idx_dev = Self._identity_idx(ctx, N)
        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[1]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[2]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[3]))
        var y_b = up_f32(ctx, y, N * iface_dim1(parts[4]))
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var pmax = max(grid_for(self.P), grid_for(self.chunk * self.k)) * GTPB
        var partials_buf = ctx.enqueue_create_buffer[DType.float32](pmax)
        var partials_host = alloc[Float32](pmax)

        var nit = gpu_lbfgs(
            self, ctx, theta,
            devp(xn_b), devp(xe_b), devp(xc_b), devp(y_b),
            devp_i32(idx_dev), N, max_iter, tol, maxcor, alpha,
            losses_out, task, devp(ws_buf), partials_buf, partials_host,
        )
        ctx.synchronize()
        partials_host.unsafe_free()
        return Int(nit)

    @staticmethod
    def loss_grad(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var ctx = DeviceContext()
        self._warmup(ctx)
        var np = np_module()
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var y = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)
        var task = Int32(Int(py=parts[5]))
        var alpha = Float32(Float64(py=parts[6]))

        var grad_arr = np_empty_1d(np, self.P, "float32")
        var grad = ptr_f32(grad_arr)
        var idx_dev = Self._identity_idx(ctx, N)
        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[1]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[2]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[3]))
        var y_b = up_f32(ctx, y, N * iface_dim1(parts[4]))
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var pmax = max(grid_for(self.P), grid_for(self.chunk * self.k)) * GTPB
        var partials_buf = ctx.enqueue_create_buffer[DType.float32](pmax)
        var partials_host = alloc[Float32](pmax)

        var loss = gpu_eval(
            self, ctx, theta, grad, True, True, alpha,
            devp(ws_buf), partials_buf, partials_host,
            devp(xn_b), devp(xe_b), devp(xc_b), devp(y_b),
            devp_i32(idx_dev), N, task,
        )
        ctx.synchronize()
        partials_host.unsafe_free()
        return Python.tuple(Float64(loss), grad_arr)

    @staticmethod
    def forward_avg(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var ctx = DeviceContext()
        self._warmup(ctx)
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var out = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)

        var idx_dev = Self._identity_idx(ctx, N)
        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[1]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[2]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[3]))
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var wp = devp(ws_buf)
        var theta_b = up_f32(ctx, theta, self.P)
        var tp = devp(theta_b)
        ctx.synchronize()

        var bk_max = self.chunk * self.k
        var chunk_dev = ctx.enqueue_create_buffer[DType.float32](bk_max * self.dout)
        var chunk_host = alloc[Float32](bk_max * self.dout)
        var inv_k = 1.0 / Float32(self.k)

        var start = 0
        while start < N:
            var b = min(self.chunk, N - start)
            var bk = b * self.k
            gpu_forward(
                self, ctx, tp, wp,
                devp(xn_b), devp(xe_b), devp(xc_b),
                devp_i32(idx_dev) + start, Int32(b), 0.0, False, 1,
            )
            ctx.synchronize()  # forward must finish before the device copies
            ctx.enqueue_function[dk_copy](
                devp(chunk_dev), wp + self.o_preds, Int32(bk * self.dout),
                grid_dim=grid_for(bk * self.dout), block_dim=GTPB,
            )
            ctx.enqueue_copy(chunk_host, chunk_dev)
            ctx.synchronize()
            var r = 0
            while r < b:
                var j = 0
                while j < self.k:
                    var o = 0
                    while o < self.dout:
                        out[(start + r) * self.dout + o] += chunk_host[(r * self.k + j) * self.dout + o] * inv_k
                        o += 1
                    j += 1
                r += 1
            start += b
        chunk_host.unsafe_free()
        return Python.none()

    @staticmethod
    def param_count(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var self = self_ptr[]
        return Int(self.P)


