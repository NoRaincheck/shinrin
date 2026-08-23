"""Metal (Mojo GPU) training kernels for the plain MLP estimators.

Compiles to the ``shinrin._native_mlp_gpu`` Python extension module (build
with ``just build-mlp-metal``). Exposes an ``MLPTrainer`` bound type whose
API mirrors the CPU kernels in ``_mlp_kernels.mojo`` exactly, so
``shinrin._mlp._mojo_trainer`` drives either backend transparently:

- ``adam_epoch(parts)``, ``lbfgs_minimize(parts)``, ``loss_grad(parts)``,
  ``forward(parts)``, ``param_count()``

Parameter layout matches ``shinrin._mlp._layers.MLPParams.flatten`` (see
``_mlp_kernels.mojo``). Loss conventions mirror ``MLPCore.loss_and_dpreds``
(regression is ``0.5*sum(diff^2)/B``). Activation codes: 0 identity,
1 logistic, 2 tanh, 3 relu. The same runtime caveats documented for the
TabM GPU backend apply (verified warmup against lazy Metal pipeline
creation; experimental on max 26.5 / macOS 26).
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, exp, log, pow
from std.memory import alloc
from std.io import Writer
from std.os import abort, getenv
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from _gpu_common import (
    GTPB,
    dk_accumulate,
    dk_mul3,
    dk_zero,
    dk_adam_update,
    dk_cat_copy,
    dk_copy,
    dk_dpl,
    dk_embed_linear,
    dk_emb_wb_grad,
    dk_gather_cols_idx,
    dk_gather_rows,
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
    mix_u64,
    gpu_global_idx,
    gpu_sigmoid,
    gpu_softplus,
)


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


def iface_dim1(arr: PythonObject) raises -> Int:
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


# =============================================================================
# MLP-specific device kernels
# =============================================================================


@always_inline
def _mlp_act(code: Int32, z: Float32) -> Float32:
    """Hidden-layer activation (0 id, 1 logistic, 2 tanh, 3 relu)."""
    if code == 3:
        return z if z > 0.0 else 0.0
    if code == 1:
        return gpu_sigmoid(z)
    if code == 2:
        var e = exp(2.0 * z)
        return (e - 1.0) / (e + 1.0)
    return z


@always_inline
def _mlp_act_deriv(code: Int32, z: Float32) -> Float32:
    if code == 3:
        return 1.0 if z > 0.0 else 0.0
    if code == 1:
        var s = gpu_sigmoid(z)
        return s * (1.0 - s)
    if code == 2:
        var e = exp(2.0 * z)
        var t = (e - 1.0) / (e + 1.0)
        return 1.0 - t * t
    return 1.0


def dk_mlp_bias_act(
    z: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    act: Pointer[Float32, MutUntrackedOrigin],
    der: Pointer[Float32, MutUntrackedOrigin],
    msk: Pointer[Float32, MutUntrackedOrigin],
    total: Int32,
    fout: Int32,
    is_hidden: Int32,
    act_code: Int32,
    keep: Float32,
    thresh: UInt64,
    use_dropout: Int32,
    round_seed: Int64,
):
    # u = z + bias ; act/der/msk caches (+ dropout on hidden layers)
    var e = gpu_global_idx()
    if e < Int(total):
        var u = z[e] + bias[e % Int(fout)]
        if is_hidden == 0:
            msk[e] = 1.0
            der[e] = 1.0
            act[e] = u
        else:
            var keepm = True
            if use_dropout == 1:
                keepm = dropout_keep(e, round_seed, thresh)
            var dv: Float32 = keep
            if not keepm:
                dv = 0.0
            msk[e] = dv
            der[e] = _mlp_act_deriv(act_code, u)
            act[e] = _mlp_act(act_code, u) * dv


def dk_mlp_colsum(
    gbias: Pointer[Float32, MutUntrackedOrigin],
    da: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    fout: Int32,
):
    # gbias[o] += sum_rows da[:,o]
    var o = gpu_global_idx()
    if o < Int(fout):
        var acc: Float32 = 0.0
        for r in range(Int(b)):
            acc += da[r * Int(fout) + o]
        gbias[o] += acc


def dk_mlp_loss(
    preds: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    dpreds: Pointer[Float32, MutUntrackedOrigin],
    loss_buf: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
    dout: Int32,
    task: Int32,
    gtotal: Int32,
):
    # One thread per row: fills dpreds + per-row loss.
    # task 0 regression: 0.5*sum(diff^2), dpred = diff/gtotal
    # task 1 binary BCE-with-logits, task 2 softmax CE.
    var row = gpu_global_idx()
    if row < Int(b):
        var src = Int(idx[row])
        var lrow: Float32 = 0.0
        if task == 0:
            for o in range(Int(dout)):
                var diff = preds[row * Int(dout) + o] - y[src * Int(dout) + o]
                lrow += 0.5 * diff * diff
                dpreds[row * Int(dout) + o] = diff / Float32(gtotal)
        elif task == 1:
            var z = preds[row * Int(dout)]
            var target = y[src * Int(dout)]
            lrow += gpu_softplus(z) - target * z
            dpreds[row * Int(dout)] = (gpu_sigmoid(z) - target) / Float32(gtotal)
        else:
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
                dpreds[base + o] = (sm - onehot) / Float32(gtotal)
        loss_buf[row] = lrow


# =============================================================================
# host RNG (xorshift64*) for the Fisher-Yates shuffle (CPU-identical stream)
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
# Trainer
# =============================================================================


struct MLPTrainer(ImplicitlyCopyable, Movable, Writable):
    def write_to(mut self, mut writer: Some[Writer]):
        writer.write("MLPGPUTrainer(P=", self.P, ")")

    var nl: Int
    var F: Int
    var denc: Int
    var ccat: Int
    var use_emb: Bool
    var demb: Int
    var act_code: Int
    var d_in: Int
    var dout: Int
    var chunk: Int
    var maxw: Int
    var warmed: Bool
    var o_end: Int
    var layersizes: Pointer[Int, MutUntrackedOrigin]
    var bin_counts: Pointer[Int, MutUntrackedOrigin]
    var bin_offsets: Pointer[Int, MutUntrackedOrigin]
    var offs: Pointer[Int, MutUntrackedOrigin]
    var n_offs: Int
    var P: Int
    # slab offsets for the fixed chunk size; h lives at offset 0
    var o_pl: Int
    var o_tmp: Int
    var o_tmp2: Int
    var o_xnum: Int
    var o_enc: Int
    var o_preds: Int
    var o_dpreds: Int
    var o_loss: Int
    var o_da: Int
    var o_dh: Int
    var loffs: Pointer[Int, MutUntrackedOrigin]  # nl * 3 (act/der/msk)

    @always_inline
    def lay(self, i: Int, which: Int) -> Int:
        return self.loffs[3 * i + which]

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
        self.maxw = self.d_in
        for i in range(nl):
            var w = Int(self.layersizes[i + 1])
            if w > self.maxw:
                self.maxw = w

        self.bin_counts = alloc[Int](self.F + 16)
        self.bin_offsets = alloc[Int](self.F + 17)
        var total_bins = 0
        self.bin_offsets[0] = 0
        var bp = ptr_i64(bins)
        var nbins = len(bins)
        for f in range(self.F):
            var cnt = Int(bp[f]) if f < nbins else 0
            self.bin_counts[f] = cnt
            total_bins += cnt
            self.bin_offsets[f + 1] = total_bins
        if total_bins != self.denc:
            raise Error("bin counts do not match d_enc")

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
        for i in range(self.nl):
            var fin = Int(self.layersizes[i])
            var fout = Int(self.layersizes[i + 1])
            self.offs[pos] = cur
            pos += 1
            cur += fout * fin  # l{i}_w
            self.offs[pos] = cur
            pos += 1
            cur += fout  # l{i}_b
        self.P = cur

        self.chunk = 8192
        self.warmed = False
        var lb = self.chunk
        var lfdemb = self.F * self.demb
        cur = 0
        cur += lb * self.d_in + 16  # h at offset 0
        self.o_pl = cur
        cur += lb * lfdemb + 16
        self.o_tmp = cur
        cur += lb * max(max(lfdemb, self.maxw), self.denc) + 16
        self.o_tmp2 = cur
        cur += lb * max(self.demb, self.maxw) + 16
        self.o_xnum = cur
        cur += lb * self.F + 16
        self.o_enc = cur
        cur += lb * self.denc + 16
        self.o_preds = cur
        cur += lb * self.dout + 16
        self.o_dpreds = cur
        cur += lb * self.dout + 16
        self.o_loss = cur
        cur += lb + 16
        self.o_da = cur
        cur += lb * self.maxw + 16
        self.o_dh = cur
        cur += lb * self.d_in + 16
        self.loffs = alloc[Int](self.nl * 3 + 16)
        var lw = 0
        for i in range(self.nl):
            var w = Int(self.layersizes[i + 1])
            self.loffs[lw] = cur
            cur += lb * w + 16  # act
            lw += 1
            self.loffs[lw] = cur
            cur += lb * w + 16  # der
            lw += 1
            self.loffs[lw] = cur
            cur += lb * w + 16  # msk
            lw += 1
        self.o_end = cur

    @staticmethod
    def py_init(out self: MLPTrainer, args: PythonObject, kwargs: PythonObject) raises:
        _ = kwargs
        if len(args) != 3:
            raise Error("MLPTrainer(dims, layersizes, bins) expects 3 arguments")
        self = Self(args[0], args[1], args[2])

    def _warmup(mut self, mut ctx: DeviceContext) raises:
        """Compile every device pipeline and verify each kernel executes.

        Same rationale as the TabM GPU trainer: Metal pipeline creation is
        lazy and a launch racing it can be silently dropped. Every kernel
        runs against a pattern-filled scratch buffer until its output slot
        differs from the fill value.
        """
        var scratch = ctx.enqueue_create_buffer[DType.float32](64)
        var sp = devp(scratch)
        var sread = ctx.enqueue_create_buffer[DType.float32](64)
        var shost = alloc[Float32](64)
        var iscratch = ctx.enqueue_create_buffer[DType.int32](8)

        var izero = alloc[Int32](8)
        for i in range(8):
            izero[i] = 0
        ctx.enqueue_copy(iscratch, izero)
        var pat_host = alloc[Float32](64)
        for i in range(64):
            pat_host[i] = Float32(i + 2) * 0.375
        var pat_buf = ctx.enqueue_create_buffer[DType.float32](64)
        ctx.enqueue_copy(pat_buf, pat_host)
        var ip = devp_i32(iscratch)
        ctx.synchronize()

        comptime NK = 22
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
                ctx.enqueue_function[dk_mlp_bias_act](
                    sp + 18, sp + 59, sp + 20, sp + 21, sp + 22,
                    Int32(1), Int32(1), Int32(1), Int32(3), Float32(1.0),
                    UInt64(256), Int32(0), Int64(0), grid_dim=1, block_dim=GTPB,
                )
            if not done[19]:
                ctx.enqueue_function[dk_mlp_colsum](sp + 23, sp + 24, Int32(1), Int32(1), grid_dim=1, block_dim=GTPB)
            if not done[20]:
                ctx.enqueue_function[dk_mlp_loss](
                    sp + 25, sp + 26, ip, sp + 27, sp + 28,
                    Int32(1), Int32(1), Int32(0), Int32(1),
                    grid_dim=1, block_dim=GTPB,
                )
            if not done[21]:
                ctx.enqueue_function[dk_zero](sp + 29, Int32(1), grid_dim=1, block_dim=GTPB)
            ctx.synchronize()

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

    # -- offset accessors (same as the CPU kernels) -----------------------------

    @always_inline
    def emb_count(self) -> Int:
        if self.use_emb and self.F > 0:
            return 2 + self.F
        return 0

    @always_inline
    def off_w(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 2 * i]

    @always_inline
    def off_b(self, i: Int) -> Int:
        return self.offs[self.emb_count() + 2 * i + 1]

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
    def fan_in(self, i: Int) -> Int:
        return Int(self.layersizes[i])

    @always_inline
    def fan_out(self, i: Int) -> Int:
        return Int(self.layersizes[i + 1])

    # -- Python-visible methods ---------------------------------------------------

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
        var task = Int32(Int(py=parts[13]))

        var ctx = DeviceContext()
        self._warmup(ctx)
        self.warmed = True
        # Fisher-Yates on the same xorshift64* stream as the CPU kernels
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
        var pmax = max(grid_for(self.P), grid_for(self.chunk)) * GTPB
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

        var ctx = DeviceContext()
        self._warmup(ctx)
        self.warmed = True
        var idx_dev = _identity_idx(ctx, N)
        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[1]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[2]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[3]))
        var y_b = up_f32(ctx, y, N * iface_dim1(parts[4]))
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var pmax = max(grid_for(self.P), grid_for(self.chunk)) * GTPB
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
        var np = np_module()
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var y = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)
        var task = Int32(Int(py=parts[5]))
        var alpha = Float32(Float64(py=parts[6]))

        var ctx = DeviceContext()
        self._warmup(ctx)
        self.warmed = True
        var grad_arr = np_empty_1d(np, self.P, "float32")
        var grad = ptr_f32(grad_arr)
        var idx_dev = _identity_idx(ctx, N)
        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[1]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[2]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[3]))
        var y_b = up_f32(ctx, y, N * iface_dim1(parts[4]))
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var pmax = max(grid_for(self.P), grid_for(self.chunk)) * GTPB
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
    def forward(self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject) raises -> PythonObject:
        var self = self_ptr[]
        var theta = ptr_f32(parts[0])
        var x_num = ptr_f32(parts[1])
        var x_enc = ptr_f32(parts[2])
        var x_cat = ptr_f32(parts[3])
        var out = ptr_f32(parts[4])
        var N = iface_dim(parts[4], 0)

        var ctx = DeviceContext()
        self._warmup(ctx)
        self.warmed = True
        var idx_dev = _identity_idx(ctx, N)
        var xn_b = up_f32(ctx, x_num, N * iface_dim1(parts[1]))
        var xe_b = up_f32(ctx, x_enc, N * iface_dim1(parts[2]))
        var xc_b = up_f32(ctx, x_cat, N * iface_dim1(parts[3]))
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](self.o_end)
        var wp = devp(ws_buf)
        var tp = devp(up_f32(ctx, theta, self.P))

        var chunk_dev = ctx.enqueue_create_buffer[DType.float32](self.chunk * self.dout)
        var chunk_host = alloc[Float32](self.chunk * self.dout)

        var start = 0
        while start < N:
            var b = min(self.chunk, N - start)
            gpu_forward(
                self, ctx, tp, wp,
                devp(xn_b), devp(xe_b), devp(xc_b),
                devp_i32(idx_dev) + start, Int32(b), 0.0, False, 1,
            )
            ctx.enqueue_function[dk_copy](
                devp(chunk_dev), wp + self.o_preds, Int32(b * self.dout),
                grid_dim=grid_for(b * self.dout), block_dim=GTPB,
            )
            ctx.enqueue_copy(chunk_host, chunk_dev)
            ctx.synchronize()
            var n_out = b * self.dout
            var k2 = 0
            while k2 < n_out:
                out[start * self.dout + k2] = chunk_host[k2]
                k2 += 1
            start += b
        chunk_host.unsafe_free()
        return Python.none()

    @staticmethod
    def param_count(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var self = self_ptr[]
        return Int(self.P)


@export
def PyInit__native_mlp_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native_mlp_gpu")
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
        abort(String("failed to create module _native_mlp_gpu: ", e))


# =============================================================================
# device orchestration
# =============================================================================


def gpu_forward(
    tr: MLPTrainer,
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

    # ---- layers -----------------------------------------------------------
    var i = 0
    while i < tr.nl:
        var fin = tr.fan_in(i)
        var fout = tr.fan_out(i)
        var inp = ws if i == 0 else ws + tr.lay(i - 1, 0)
        # z lands in the da scratch (maxw >= fout); bias/act next kernel reads it
        ctx.enqueue_function[dk_gemm_nt](
            ws + tr.o_da, inp, theta + tr.off_w(i),
            Int32(bi), Int32(fout), Int32(fin), Int32(fout),
            grid_dim=grid_for(bi * fout), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_mlp_bias_act](
            ws + tr.o_da, theta + tr.off_b(i),
            ws + tr.lay(i, 0), ws + tr.lay(i, 1), ws + tr.lay(i, 2),
            Int32(bi * fout), Int32(fout),
            Int32(1 if i < tr.nl - 1 else 0), Int32(tr.act_code),
            keep, thresh, Int32(1 if use_drop else 0), round_seed,
            grid_dim=grid_for(bi * fout), block_dim=GTPB,
        )
        i += 1

    # raw predictions of the final layer
    var act_last = ws + tr.lay(tr.nl - 1, 0)
    ctx.enqueue_function[dk_copy](
        ws + tr.o_preds, act_last, Int32(bi * tr.dout),
        grid_dim=grid_for(bi * tr.dout), block_dim=GTPB,
    )


def gpu_round(
    tr: MLPTrainer,
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
    """Forward -> loss/dpreds -> backward; returns the round's row-loss sum."""
    var bi = Int(b)
    var fdemb = tr.F * tr.demb

    gpu_forward(tr, ctx, theta, ws, x_num, x_enc, x_cat, idx, b, dropout, train, round_seed)

    # ---- loss + dpreds ------------------------------------------------------
    ctx.enqueue_function[dk_mlp_loss](
        ws + tr.o_preds, y, idx, ws + tr.o_dpreds, ws + tr.o_loss,
        b, Int32(tr.dout), task, Int32(denom_b),
        grid_dim=grid_for(bi), block_dim=GTPB,
    )
    ctx.enqueue_function[dk_sum](
        ws + tr.o_loss, devp(partials_buf), Int32(b),
        grid_dim=grid_for(bi), block_dim=GTPB,
    )

    # ---- backward -----------------------------------------------------------
    var da = ws + tr.o_da
    ctx.enqueue_function[dk_copy](
        da, ws + tr.o_dpreds, Int32(bi * tr.dout),
        grid_dim=grid_for(bi * tr.dout), block_dim=GTPB,
    )

    var i = tr.nl - 1
    while i >= 0:
        var fin = tr.fan_in(i)
        var fout = tr.fan_out(i)
        var inp = ws if i == 0 else ws + tr.lay(i - 1, 0)
        ctx.enqueue_function[dk_gemm_tn_acc](
            g + tr.off_w(i), da, inp,
            Int32(fout), Int32(fin), Int32(bi), Int32(fin),
            grid_dim=grid_for(fout * fin), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_mlp_colsum](
            g + tr.off_b(i), da, b, Int32(fout),
            grid_dim=grid_for(fout), block_dim=GTPB,
        )
        if i == 0:
            if tr.use_emb and tr.F > 0:
                ctx.enqueue_function[dk_gemm_nn](
                    ws + tr.o_dh, da, theta + tr.off_w(i),
                    Int32(bi), Int32(fin), Int32(fout), Int32(tr.d_in),
                    grid_dim=grid_for(bi * fin), block_dim=GTPB,
                )
                _embed_backward(tr, ctx, theta, ws, g, b)
            break

        # dz_prev = (da @ W_i) * der_{i-1} * msk_{i-1}
        ctx.enqueue_function[dk_copy](
            ws + tr.o_tmp2, da, Int32(bi * fout),
            grid_dim=grid_for(bi * fout), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_gemm_nn](
            da, ws + tr.o_tmp2, theta + tr.off_w(i),
            Int32(bi), Int32(fin), Int32(fout), Int32(fin),
            grid_dim=grid_for(bi * fin), block_dim=GTPB,
        )
        ctx.enqueue_function[dk_mul3](
            da, ws + tr.lay(i - 1, 1), ws + tr.lay(i - 1, 2), Int32(bi * fin),
            grid_dim=grid_for(bi * fin), block_dim=GTPB,
        )
        i -= 1

    # ---- loss readback -------------------------------------------------------
    var npart = grid_for(bi) * GTPB
    ctx.synchronize()  # dk_sum must finish before the readback
    ctx.enqueue_copy(partials_host, partials_buf)
    ctx.synchronize()
    var total: Float64 = 0.0
    for t in range(npart):
        total += Float64(partials_host[t])
    return total


def _embed_backward(
    tr: MLPTrainer,
    mut ctx: DeviceContext,
    theta: Pointer[Float32, MutUntrackedOrigin],
    ws: Pointer[Float32, MutUntrackedOrigin],
    g: Pointer[Float32, MutUntrackedOrigin],
    b: Int32,
) raises:
    """Embedding gradients from ws.dh (uses xnum/enc_snap/pl snapshots)."""
    var bi = Int(b)
    var fdemb = tr.F * tr.demb
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


def gpu_eval(
    tr: MLPTrainer,
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
    tr: MLPTrainer,
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
            gp, Int32(tr.P), grid_dim=grid_for(tr.P), block_dim=GTPB
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
    tr: MLPTrainer,
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
    """Full-batch L-BFGS with backtracking line search (GPU evaluations)."""
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
        i = 0
        while i < P:
            dir[i] = -g64[i]
            i += 1
        var nj = 0
        while nj < nhist:
            var slot = (hptr - 1 - nj + maxcor) % maxcor
            var aj = rho[slot] * dot_f64(S + slot * P, dir, P)
            alphas[nj] = aj
            i = 0
            while i < P:
                dir[i] -= aj * Y[slot * P + i]
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
            var bj = rho[slot] * dot_f64(Y + slot * P, dir, P)
            i = 0
            while i < P:
                dir[i] += (alphas[pj] - bj) * S[slot * P + i]
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


# =============================================================================
# shared index helper
# =============================================================================


def _identity_idx(ctx: DeviceContext, N: Int) raises -> DeviceBuffer[DType.int32]:
    var idx_host = alloc[Int32](N)
    for i in range(N):
        idx_host[i] = Int32(i)
    var buf = up_i32(ctx, idx_host, N)
    ctx.synchronize()  # idx_host is about to be freed
    idx_host.unsafe_free()
    return buf

