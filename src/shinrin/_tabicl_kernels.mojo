"""Mojo inference kernels for TabICLv2 (**experimental scaffold**).

Compiles to the ``shinrin._native_tabicl`` Python extension module (build
with ``just build-tabicl-mojo``). Exposes a ``TabICLInference`` bound type
following the same binding pattern as ``_tabm_kernels.mojo``:

- ``TabICLInference(dims, params)``: construct from an int64 dims vector
  (see ``shinrin._tabicl._config.TabICLConfig.dims_array``) and a flat
  float32 parameter buffer (state dict flattened in sorted-name order).
- ``forward(col_input, row_input, target)``: run the forward pass and
  return the output as a NumPy array.
- ``param_count()``: number of expected float32 parameters.

.. note::
   This kernel is a performance scaffold: it builds and runs but does not
   yet reproduce the reference pipeline bit-for-bit (SSMax MLPs, target
   masking and KV caching are simplified). Numeric parity is tracked by
   ``tests/test_tabicl_parity.py``, which only exercises this backend when
   explicitly enabled.

All arrays cross the boundary as NumPy buffers accessed through raw
pointers (float32 data, int64 dims).
"""

from std.os import abort
from std.math import exp, max, sqrt
from std.memory import alloc
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from shinrin._tk_core import (
    SIMDW,
    ThreadPool,
    gemm_nn,
    gemm_nt,
    gelu8,
    gelu_scalar,
    iface_dim,
    np_module,
    pool_init,
    pool_run,
    pool_shutdown,
    ptr_f32,
    ptr_f64,
    ptr_i64,
)
from shinrin._tk_layers import (
    AttnParams,
    _add_row_bias,
    attention_block_forward,
    attention_block_forward_cached,
    attention_block_size,
    attn_params_at,
    isab_forward,
    layer_norm_affine,
    rope_apply,
)

alias SKIP_VALUE_F32: Float32 = -100.0


# =============================================================================
# TabICLConfig
# =============================================================================


struct TabICLConfig(ImplicitlyCopyable, Movable, Writable):
    """Architecture hyper-parameters unpacked from the int64 dims vector.

    The field order of the vector is fixed by
    ``shinrin._tabicl._config.TabICLConfig.dims_array``; keep both sides in
    sync.
    """

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
    # Appended fields (dims indices 23-25)
    var col_ssmax_kind: Int
    var icl_ssmax_kind: Int
    var col_affine: Bool

    def __init__(out self, dims: PythonObject) raises:
        if iface_dim(dims, 0) < 26:
            raise Error(
                "TabICL dims vector must have at least 26 entries "
                "(rebuild the package so _config.dims_array matches the kernel)"
            )
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
        # rope base is stored rounded to int64 (exact for 100000.0)
        self.row_rope_base = Float64(py=dims[11])
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
        self.col_ssmax_kind = Int(dp[23])
        self.icl_ssmax_kind = Int(dp[24])
        self.col_affine = Int(dp[25]) == 1
        if self.col_affine:
            raise Error(
                "col_affine=True is not supported by the native TabICL kernel"
            )

    def y_encoder_in_features(self) -> Int:
        """Input width of both y-encoders: classes for classification, 1 target."""
        return max(self.max_classes, 1)

    def write_to(mut self, mut writer: Some[Writer]):
        writer.write(
            "TabICLConfig(embed_dim=", self.embed_dim,
            ", icl_dim=", self.icl_dim, ")"
        )

    def col_dim_feedforward(self) -> Int:
        return self.embed_dim * self.ff_factor

    def icl_dim_feedforward(self) -> Int:
        return self.icl_dim * self.ff_factor

    def col_head_dim(self) -> Int:
        return self.embed_dim // self.col_nhead

    def icl_head_dim(self) -> Int:
        return self.icl_dim // self.icl_nhead


# =============================================================================
# TabICLInference — full pipeline with workspace management
# =============================================================================


struct TabICLInference(ImplicitlyCopyable, Movable, Writable):
    def write_to(mut self, mut writer: Some[Writer]):
        writer.write("TabICLInference(P=", self.P, ")")

    var config: TabICLConfig
    var P: Int

    # Parameter data (raw pointer from NumPy)
    var params: Pointer[Float32, MutUntrackedOrigin]

    # Canonical-layout offsets (single source of truth:
    # shinrin._tabicl._mojo_layout.canonical_tensor_specs)
    var _off_in_linear_w: Int
    var _off_in_linear_b: Int
    var _off_col_y_enc_w: Int
    var _off_col_y_enc_b: Int
    var _off_col_blocks: Int
    var _col_blk_stride: Int
    var _col_attn_size: Int
    var _off_cls_tokens: Int
    var _off_row_out_ln_w: Int
    var _off_row_out_ln_b: Int
    var _off_rope_freqs: Int
    var _off_row_blocks: Int
    var _row_attn_size: Int
    var _off_icl_ln_w: Int
    var _off_icl_ln_b: Int
    var _off_icl_y_encoder_w: Int
    var _off_icl_y_encoder_b: Int
    var _off_decoder_w1: Int
    var _off_decoder_b1: Int
    var _off_decoder_w2: Int
    var _off_decoder_b2: Int
    var _off_icl_blocks: Int
    var _icl_attn_size: Int

    def __init__(out self, dims: PythonObject, param_data: PythonObject) raises:
        self.config = TabICLConfig(dims)
        self.params = ptr_f32(param_data)
        self.P = iface_dim(param_data, 0)

        var cfg = self.config
        var e = cfg.embed_dim
        var cff = cfg.col_dim_feedforward()
        var chd = cfg.col_head_dim()
        var icl_d = cfg.icl_dim
        var rhd = e // cfg.row_nhead

        # Walk the canonical tensor order exactly as
        # `_mojo_layout.canonical_tensor_specs` emits it.
        var cur = 0

        # -- Stage 1: ColEmbedding ---------------------------------------- #
        self._off_in_linear_w = cur
        cur += cfg.col_feature_group_size * e
        self._off_in_linear_b = cur
        cur += e
        if cfg.col_target_aware:
            var y_in = cfg.y_encoder_in_features()
            self._off_col_y_enc_w = cur
            cur += e * y_in
            self._off_col_y_enc_b = cur
            cur += e
        else:
            self._off_col_y_enc_w = 0
            self._off_col_y_enc_b = 0
        self._col_attn_size = attention_block_size(
            e, cff, cfg.col_nhead, chd, cfg.col_ssmax_kind,
            cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
        )
        # Per block: ind_vectors + attn1 (with ssmax) + attn2 (ssmax none).
        self._col_blk_stride = (
            cfg.col_num_inds * e
            + self._col_attn_size
            + attention_block_size(
                e, cff, cfg.col_nhead, chd, 0,
                cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
            )
        )
        self._off_col_blocks = cur
        cur += cfg.col_num_blocks * self._col_blk_stride

        # -- Stage 2: RowInteraction -------------------------------------- #
        self._off_cls_tokens = cur
        cur += cfg.row_num_cls * e
        self._off_row_out_ln_w = cur
        cur += e
        if cfg.bias_free_ln:
            self._off_row_out_ln_b = 0
        else:
            self._off_row_out_ln_b = cur
            cur += e
        self._off_rope_freqs = cur
        cur += (rhd + 1) // 2
        self._row_attn_size = attention_block_size(
            e, cff, cfg.row_nhead, rhd, 0, cfg.col_ssmax_n_hidden,
            cfg.bias_free_ln,
        )
        self._off_row_blocks = cur
        cur += cfg.row_num_blocks * self._row_attn_size

        # -- Stage 3: ICLearning ------------------------------------------- #
        self._off_icl_ln_w = cur
        cur += icl_d
        if cfg.bias_free_ln:
            self._off_icl_ln_b = 0
        else:
            self._off_icl_ln_b = cur
            cur += icl_d
        var icl_y_in = cfg.y_encoder_in_features()
        self._off_icl_y_encoder_w = cur
        cur += icl_d * icl_y_in
        self._off_icl_y_encoder_b = cur
        cur += icl_d
        self._off_decoder_w1 = cur
        cur += icl_d * icl_d * 2
        self._off_decoder_b1 = cur
        cur += icl_d * 2
        self._off_decoder_w2 = cur
        cur += cfg.out_dim * icl_d * 2
        self._off_decoder_b2 = cur
        cur += cfg.out_dim
        self._icl_attn_size = attention_block_size(
            icl_d, cfg.icl_dim_feedforward(), cfg.icl_nhead,
            cfg.icl_head_dim(), cfg.icl_ssmax_kind,
            cfg.icl_ssmax_n_hidden, cfg.bias_free_ln,
        )
        self._off_icl_blocks = cur
        cur += cfg.icl_num_blocks * self._icl_attn_size

        # Layout drift becomes a loud construction error instead of silent
        # misaligned reads / heap corruption downstream.
        if cur != self.P:
            raise Error(
                "parameter buffer size mismatch: the canonical TabICL layout "
                "walk disagrees with the provided buffer size; repack the "
                "state dict with "
                "shinrin._tabicl._mojo_layout.pack_params"
            )

    # =====================================================================
    # Python init
    # =====================================================================

    @staticmethod
    def py_init(out self: TabICLInference, args: PythonObject, kwargs: PythonObject) raises:
        """Initialize from Python: TabICLInference(dims, param_data)."""
        _ = kwargs
        if len(args) != 2:
            raise Error("TabICLInference(dims, params) expects 2 arguments")
        self = Self(args[0], args[1])

    # =====================================================================
    # param_count
    # =====================================================================

    @staticmethod
    def param_count(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var self = self_ptr[]
        return Python.int(self.P)

    # =====================================================================
    # layout_offsets
    # =====================================================================

    @staticmethod
    def layout_offsets(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        """Return the kernel's walked layout offsets as a list of ints.

        The Python side rebuilds the same numbers from
        ``_mojo_layout.canonical_tensor_specs`` and compares them exactly;
        any ordering or sizing drift between the two sides becomes a loud
        construction error instead of silent weight misalignment. Order:
        ``[P, in_linear.w/b, col_y_enc.w/b, col_blocks, col_blk_stride,
        col_attn_size, cls_tokens, row_out_ln.w/b, rope_freqs, row_blocks,
        row_attn_size, icl_ln.w/b, icl_y_enc.w/b, decoder.w1/b1/w2/b2,
        icl_blocks, icl_attn_size]``.
        """
        var self = self_ptr[]
        return Python.list(
            [
                Python.int(self.P),
                Python.int(self._off_in_linear_w),
                Python.int(self._off_in_linear_b),
                Python.int(self._off_col_y_enc_w),
                Python.int(self._off_col_y_enc_b),
                Python.int(self._off_col_blocks),
                Python.int(self._col_blk_stride),
                Python.int(self._col_attn_size),
                Python.int(self._off_cls_tokens),
                Python.int(self._off_row_out_ln_w),
                Python.int(self._off_row_out_ln_b),
                Python.int(self._off_rope_freqs),
                Python.int(self._off_row_blocks),
                Python.int(self._row_attn_size),
                Python.int(self._off_icl_ln_w),
                Python.int(self._off_icl_ln_b),
                Python.int(self._off_icl_y_encoder_w),
                Python.int(self._off_icl_y_encoder_b),
                Python.int(self._off_decoder_w1),
                Python.int(self._off_decoder_b1),
                Python.int(self._off_decoder_w2),
                Python.int(self._off_decoder_b2),
                Python.int(self._off_icl_blocks),
                Python.int(self._icl_attn_size),
            ]
        )

    # =====================================================================
    # Main forward pass
    # =====================================================================

    @staticmethod
    def forward(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """Run the forward pass.

        ``parts = (col_input, row_input, target)`` where ``col_input`` is
        ``(n_train, group_size)`` float32, ``row_input`` is ``(max(n_test,
        1), embed_dim)`` float32 and ``target`` is ``(n_train,)`` int64.
        Returns a flat float32 NumPy array of ``max(n_test, 1) * out_dim``
        elements.
        """
        var self = self_ptr[]
        var col_input = ptr_f32(parts[0])
        var row_input = ptr_f32(parts[1])
        var target = ptr_i64(parts[2])
        var n_train = iface_dim(parts[0], 0)
        var n_test = iface_dim(parts[1], 0)
        var n_classes = iface_dim(parts[2], 0)
        var np = np_module()
        var total = max(n_test, 1) * self.config.out_dim
        var out_arr = np.empty(Python.tuple(Int(total)), "float32")
        var output = ptr_f32(out_arr)
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        try:
            _ = self.forward_impl(
                pool_ptr, col_input, row_input, target, n_train, n_test,
                n_classes, output
            )
        except e:
            pool_shutdown(pool_ptr)
            pool_ptr.free()
            raise e
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return out_arr

    def forward_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        col_input: Pointer[Float32, MutUntrackedOrigin],  # (n_train, group_size)
        row_input: Pointer[Float32, MutUntrackedOrigin],  # (n_test, embed_dim)
        target: Pointer[Int, MutUntrackedOrigin],         # (n_train,)
        n_train: Int,
        n_test: Int,
        n_classes: Int,
        output: Pointer[Float32, MutUntrackedOrigin],     # (n_test, out_dim)
    ) -> Int:
        """Run full TabICL forward pass. Returns 0 on success."""
        var cfg = self.config
        var e = cfg.embed_dim
        var cff = cfg.col_dim_feedforward()
        var chd = cfg.col_head_dim()

        # ---- ColEmbedding: in_linear, optional y_encoder, then ISABs ------
        var col_embed = alloc[Float32](n_train * e + 16)
        gemm_nt(pool, 
            n_train, e, cfg.col_feature_group_size,
            col_input, self.params + self._off_in_linear_w, col_embed,
        )
        var i = 0
        while i < n_train:
            var b = 0
            while b + SIMDW <= e:
                col_embed.unsafe_store[width=SIMDW](
                    i * e + b,
                    col_embed.unsafe_load[width=SIMDW](i * e + b)
                    + self.params.unsafe_load[width=SIMDW](self._off_in_linear_b + b),
                )
                b += SIMDW
            while b < e:
                col_embed[i * e + b] += self.params[self._off_in_linear_b + b]
                b += 1
            i += 1

        # Target-aware term: OneHotAndLinear adds y_enc_w[class] + bias to
        # TRAIN-row embeddings only (torch: src[..., :train_size, :] += y_emb).
        if cfg.col_target_aware:
            var yw = self.params + self._off_col_y_enc_w
            var yb = self.params + self._off_col_y_enc_b
            i = 0
            while i < n_train:
                var c = Int(target[i])
                if c >= 0 and c < cfg.max_classes:
                    var b = 0
                    while b < e:
                        col_embed[i * e + b] += yw[c * e + b] + yb[b]
                        b += 1
                i += 1

        var block_idx = 0
        while block_idx < cfg.col_num_blocks:
            var blk = self._off_col_blocks + block_idx * self._col_blk_stride
            var p1 = attn_params_at(
                self.params, blk + cfg.col_num_inds * e, e, cff,
                cfg.col_nhead, chd, cfg.col_ssmax_kind,
                cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
            )
            var p2 = attn_params_at(
                self.params, blk + cfg.col_num_inds * e + self._col_attn_size,
                e, cff, cfg.col_nhead, chd, 0,
                cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
            )
            isab_forward(pool, 
                col_embed, self.params + blk, col_embed,
                n_train, cfg.col_num_inds, e, cfg.col_nhead, chd, cff, p1, p2,
            )
            block_idx += 1

        # ---- RowInteraction: transformer blocks with RoPE -----------------
        # NOTE (documented reduced-graph gap): CLS-token concatenation and
        # the final cross-attention aggregation are not modelled yet; test
        # rows are processed standalone and out_ln / cls_tokens stay unused.
        var row_combined = alloc[Float32](n_test * e + 16)
        var si = 0
        while si + SIMDW <= n_test * e:
            row_combined.unsafe_store[width=SIMDW](
                si, row_input.unsafe_load[width=SIMDW](si)
            )
            si += SIMDW
        while si < n_test * e:
            row_combined[si] = row_input[si]
            si += 1

        var rhd = e // cfg.row_nhead
        block_idx = 0
        while block_idx < cfg.row_num_blocks:
            var off = self._off_row_blocks + block_idx * self._row_attn_size
            var p = attn_params_at(
                self.params, off, e, cff, cfg.row_nhead, rhd, 0,
                cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
            )
            attention_block_forward(pool, 
                row_combined, None, row_combined,
                n_test, n_test, e, cfg.row_nhead, rhd, cff,
                p, self.params + self._off_rope_freqs, cfg.row_rope_interleaved,
            )
            block_idx += 1

        # ---- ICLearning: class prototypes ---------------------------------
        var mc = cfg.max_classes
        var d_icl = cfg.icl_dim
        var tmax = e if e < d_icl else d_icl

        var class_embed = alloc[Float32](mc * d_icl + 16)
        si = 0
        while si < mc * d_icl:
            class_embed[si] = 0.0
            si += 1

        # Per-class sums; zero ALL max_classes slots so tail classes are
        # always defined regardless of how many labels appear in the batch.
        var class_counts = alloc[Int](mc + 16)
        si = 0
        while si < mc:
            class_counts[si] = 0
            si += 1

        i = 0
        while i < n_train:
            var c = Int(target[i])
            if c >= 0 and c < mc:
                var t = 0
                while t < tmax:
                    class_embed[c * d_icl + t] += col_embed[i * e + t]
                    t += 1
                class_counts[c] += 1
            i += 1

        # Average every class that received at least one sample.
        i = 0
        while i < mc:
            var cnt = class_counts[i]
            if cnt >= 1:
                var inv = 1.0 / Float32(cnt)
                var t = 0
                while t < tmax:
                    class_embed[i * d_icl + t] *= inv
                    t += 1
            i += 1
        class_counts.unsafe_free()

        # y_encoder: one-hot picks one weight row; every class row gets the
        # bias (OneHotAndLinear always adds it).
        var y_encoded = alloc[Float32](mc * d_icl + 16)
        var y_enc_w = self.params + self._off_icl_y_encoder_w
        var y_enc_b = self.params + self._off_icl_y_encoder_b
        si = 0
        while si + SIMDW <= mc * d_icl:
            y_encoded.unsafe_store[width=SIMDW](
                si, y_enc_w.unsafe_load[width=SIMDW](si)
            )
            si += SIMDW
        while si < mc * d_icl:
            y_encoded[si] = y_enc_w[si]
            si += 1
        i = 0
        while i < mc:
            var b = 0
            while b + SIMDW <= d_icl:
                y_encoded.unsafe_store[width=SIMDW](
                    i * d_icl + b,
                    y_encoded.unsafe_load[width=SIMDW](i * d_icl + b)
                    + SIMD[DType.float32, SIMDW](y_enc_b[b]),
                )
                b += SIMDW
            while b < d_icl:
                y_encoded[i * d_icl + b] += y_enc_b[b]
                b += 1
            i += 1

        # ICL self-attention blocks over the class prototypes.
        var icl_cur = y_encoded
        var ihd = cfg.icl_head_dim()
        block_idx = 0
        while block_idx < cfg.icl_num_blocks:
            var off = self._off_icl_blocks + block_idx * self._icl_attn_size
            var p = attn_params_at(
                self.params, off, d_icl, cfg.icl_dim_feedforward(),
                cfg.icl_nhead, ihd, cfg.icl_ssmax_kind,
                cfg.icl_ssmax_n_hidden, cfg.bias_free_ln,
            )
            attention_block_forward(pool, 
                icl_cur, None, icl_cur, mc, mc, d_icl, cfg.icl_nhead, ihd,
                cfg.icl_dim_feedforward(), p, None, False,
            )
            block_idx += 1

        # Reference applies the predictor LayerNorm after tf_icl and before
        # the decoder (torch: src = self.ln(src)).
        layer_norm_affine(
            icl_cur, icl_cur, mc, d_icl,
            self.params + self._off_icl_ln_w,
            self.params + self._off_icl_ln_b,
            True, not cfg.bias_free_ln,
        )

        # Decoder: prototypes -> logits.
        var decoder_w1 = self.params + self._off_decoder_w1
        var decoder_b1 = self.params + self._off_decoder_b1
        var decoder_w2 = self.params + self._off_decoder_w2
        var decoder_b2 = self.params + self._off_decoder_b2

        var decoded = alloc[Float32](mc * d_icl * 2 + 16)
        gemm_nt(pool, mc, d_icl * 2, d_icl, icl_cur, decoder_w1, decoded)
        i = 0
        while i < mc:
            var b = 0
            while b + SIMDW <= d_icl * 2:
                decoded.unsafe_store[width=SIMDW](
                    i * d_icl * 2 + b,
                    decoded.unsafe_load[width=SIMDW](i * d_icl * 2 + b)
                    + SIMD[DType.float32, SIMDW](decoder_b1[b]),
                )
                b += SIMDW
            while b < d_icl * 2:
                decoded[i * d_icl * 2 + b] += decoder_b1[b]
                b += 1
            i += 1
        i = 0
        var dtotal = mc * d_icl * 2
        while i + SIMDW <= dtotal:
            decoded.unsafe_store[width=SIMDW](
                i,
                gelu8(decoded.unsafe_load[width=SIMDW](i)),
            )
            i += SIMDW
        while i < dtotal:
            decoded[i] = gelu_scalar(decoded[i])
            i += 1

        var logits = alloc[Float32](mc * cfg.out_dim + 16)
        gemm_nt(pool, mc, cfg.out_dim, d_icl * 2, decoded, decoder_w2, logits)
        i = 0
        while i < mc:
            var b = 0
            while b + SIMDW <= cfg.out_dim:
                logits.unsafe_store[width=SIMDW](
                    i * cfg.out_dim + b,
                    logits.unsafe_load[width=SIMDW](i * cfg.out_dim + b)
                    + SIMD[DType.float32, SIMDW](decoder_b2[b]),
                )
                b += SIMDW
            while b < cfg.out_dim:
                logits[i * cfg.out_dim + b] += decoder_b2[b]
                b += 1
            i += 1
        decoded.unsafe_free()

        # ---- Prediction ----------------------------------------------------
        # NOTE (documented reduced-graph gap): without CLS aggregation there
        # is no trained projection of test rows into icl_dim; test_proj stays
        # at the LN bias so attention over classes is uniform and the output
        # is the mean of the decoded class logits (bounds-safe placeholder).
        var test_proj = alloc[Float32](n_test * d_icl + 16)
        si = 0
        while si < n_test * d_icl:
            test_proj[si] = 0.0
            si += 1
        i = 0
        while i < n_test:
            var b = 0
            while b < d_icl:
                test_proj[i * d_icl + b] += self.params[self._off_icl_ln_b + b]
                b += 1
            i += 1

        var attn_scores = alloc[Float32](n_test * mc + 16)
        gemm_nt(pool, n_test, mc, d_icl, test_proj, class_embed, attn_scores)

        i = 0
        while i < n_test:
            var mx: Float32 = -3.0e38
            var c = 0
            while c < mc:
                var v = attn_scores[i * mc + c]
                if v > mx:
                    mx = v
                c += 1
            var sum_v: Float32 = 0.0
            c = 0
            while c < mc:
                var ev = exp(attn_scores[i * mc + c] - mx)
                attn_scores[i * mc + c] = ev
                sum_v += ev
                c += 1
            c = 0
            while c < mc:
                attn_scores[i * mc + c] /= sum_v
                c += 1
            i += 1

        gemm_nn(pool, n_test, cfg.out_dim, mc, attn_scores, logits, output)

        col_embed.unsafe_free()
        row_combined.unsafe_free()
        class_embed.unsafe_free()
        y_encoded.unsafe_free()
        logits.unsafe_free()
        test_proj.unsafe_free()
        attn_scores.unsafe_free()

        return 0


    # =====================================================================
    # Stage 1+2: column embedding + row interaction (staged API)
    # =====================================================================

    def stage_col_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        x: Pointer[Float32, MutUntrackedOrigin],       # (n_rows, n_features)
        target: PythonObject,                            # (train_size,) i64/f32
        n_rows: Int,
        train_size: Int,
        n_features: Int,
        col_out: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, G, E) t-major
        col_kv_out: Optional[Pointer[Float32, MutUntrackedOrigin]] = None,
    ) raises -> Int:
        """Column embedding stage (torch ColEmbedding.forward).

        Per position g in [0, G): group features, project with in_linear,
        augment train rows with the target embedding, then run the ISAB
        blocks whose first attention keys are restricted to the train
        prefix. Results are scattered into row-major (t, g, e) order.

        When ``col_kv_out`` is given (cache-build mode), per-position attn2
        K/V are captured post-head-split into
        ``[g][blk][kv][n_inds*e]`` and sentinel positions (g < row_num_cls)
        skip block compute entirely — their outputs are overwritten by CLS
        tokens downstream and their cache slots are never read.
        Returns 0 on success.
        """
        var cfg = self.config
        var e = cfg.embed_dim
        var s = cfg.col_feature_group_size
        var num_cls = cfg.row_num_cls
        var G = num_cls + n_features
        var n_inds = cfg.col_num_inds
        var chd = cfg.col_head_dim()
        var cff = cfg.col_dim_feedforward()
        var is_clf = cfg.max_classes > 0
        var capturing = col_kv_out is not None
        # Both views alias the same buffer; only the view matching the task
        # type is ever dereferenced.
        var target_i64 = ptr_i64(target)
        var target_f32 = ptr_f32(target)

        var pos = alloc[Float32](n_rows * s + 16)
        var cur = alloc[Float32](n_rows * e + 16)
        var hidden = alloc[Float32](n_inds * e + 16)

        var g = 0
        while g < G:
            if g < num_cls:
                if not capturing:
                    # Sentinel positions: SkippableLinear maps fully-masked
                    # input rows to exactly SKIP_VALUE (no projection). In
                    # cache-build mode the block compute is skipped instead;
                    # CLS tokens overwrite these slots downstream either way.
                    var fill = n_rows * s
                    var i0 = 0
                    while i0 < fill:
                        pos[i0] = SKIP_VALUE_F32
                        i0 += 1
                    fill = n_rows * e
                    i0 = 0
                    while i0 < fill:
                        cur[i0] = SKIP_VALUE_F32
                        i0 += 1
            else:
                var h = g - num_cls
                var t = 0
                while t < n_rows:
                    var i = 0
                    while i < s:
                        pos[t * s + i] = x[
                            t * n_features + (h + (1 << i)) % n_features
                        ]
                        i += 1
                    t += 1
                gemm_nt(pool, 
                    n_rows, e, s, pos, self.params + self._off_in_linear_w, cur
                )
                var b = self._off_in_linear_b
                t = 0
                while t < n_rows:
                    var j = 0
                    while j + SIMDW <= e:
                        cur.unsafe_store[width=SIMDW](
                            t * e + j,
                            cur.unsafe_load[width=SIMDW](t * e + j)
                            + self.params.unsafe_load[width=SIMDW](b + j),
                        )
                        j += SIMDW
                    while j < e:
                        cur[t * e + j] += self.params[b + j]
                        j += 1
                    t += 1

            # Target-aware term added to TRAIN-row embeddings only
            # (torch: src[..., :train_size, :] += y_emb).
            if cfg.col_target_aware:
                var yw = self.params + self._off_col_y_enc_w
                var yb = self.params + self._off_col_y_enc_b
                var t = 0
                while t < train_size:
                    if is_clf:
                        # OneHotAndLinear picks weight COLUMN c of the
                        # (e, max_classes) row-major matrix: yw[j*mc + c].
                        var c = Int(target_i64[t])
                        var j = 0
                        while j < e:
                            cur[t * e + j] += yw[j * cfg.max_classes + c] + yb[j]
                            j += 1
                    else:
                        var v = target_f32[t]
                        var j = 0
                        while j < e:
                            cur[t * e + j] += yw[j] * v + yb[j]
                            j += 1
                    t += 1

            # ISAB blocks: attn1 keys restricted to the train prefix.
            var blk = 0
            while blk < cfg.col_num_blocks:
                var boff = self._off_col_blocks + blk * self._col_blk_stride
                var p1 = attn_params_at(
                    self.params, boff + n_inds * e, e, cff,
                    cfg.col_nhead, chd, cfg.col_ssmax_kind,
                    cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
                )
                var p2 = attn_params_at(
                    self.params, boff + n_inds * e + self._col_attn_size,
                    e, cff, cfg.col_nhead, chd, 0,
                    cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
                )
                if capturing:
                    var slot = (g * cfg.col_num_blocks + blk) * 2 * n_inds * e
                    attention_block_forward(pool, 
                        self.params + boff, cur, hidden,
                        n_inds, train_size, e, cfg.col_nhead, chd, cff,
                        p1, None, False,
                    )
                    attention_block_forward(pool, 
                        cur, hidden, cur,
                        n_rows, n_inds, e, cfg.col_nhead, chd, cff,
                        p2, None, False,
                        col_kv_out.value() + slot,
                        col_kv_out.value() + slot + n_inds * e,
                    )
                else:
                    attention_block_forward(pool, 
                        self.params + boff, cur, hidden,
                        n_inds, train_size, e, cfg.col_nhead, chd, cff,
                        p1, None, False,
                    )
                    attention_block_forward(pool, 
                        cur, hidden, cur,
                        n_rows, n_inds, e, cfg.col_nhead, chd, cff,
                        p2, None, False,
                    )
                blk += 1

            # Scatter this position's embeddings into row-major order.
            var t = 0
            while t < n_rows:
                var src_base = t * e
                var dst_base = t * G * e + g * e
                var j = 0
                while j + SIMDW <= e:
                    col_out.unsafe_store[width=SIMDW](
                        dst_base + j,
                        cur.unsafe_load[width=SIMDW](src_base + j),
                    )
                    j += SIMDW
                while j < e:
                    col_out[dst_base + j] = cur[src_base + j]
                    j += 1
                t += 1

            g += 1

        pos.unsafe_free()
        cur.unsafe_free()
        hidden.unsafe_free()
        return 0

    @staticmethod
    def gemm_probe(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """DEBUG: run gemm_nt(m,n,kk) on raw arrays (A (m,kk), B (n,kk))."""
        var a_in = ptr_f32(parts[0])
        var m = iface_dim(parts[0], 0)
        var kk = iface_dim(parts[0], 1)
        var b_in = ptr_f32(parts[1])
        var n = iface_dim(parts[1], 0)
        var a = alloc[Float32](m * kk + 16)
        var b = alloc[Float32](n * kk + 16)
        var c = alloc[Float32](m * n + 16)
        var i = 0
        while i < m * kk:
            a[i] = a_in[i]
            i += 1
        i = 0
        while i < n * kk:
            b[i] = b_in[i]
            i += 1
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        gemm_nt(pool_ptr, m, n, kk, a, b, c)
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        var np = np_module()
        var arr = np.empty(m * n)
        var ob = ptr_f64(arr)
        i = 0
        while i < m * n:
            ob[i] = Float64(c[i])
            i += 1
        return arr

    @staticmethod
    def attn_probe(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """DEBUG: run block-0 attn2 step-by-step on (q (R,e), k (K,e)) and
        return every intermediate, concatenated flat in documented order."""
        var self = self_ptr[]
        var cfg = self.config
        var e = cfg.embed_dim
        var nh = cfg.col_nhead
        var chd = cfg.col_head_dim()
        var cff = cfg.col_dim_feedforward()
        var q_in = ptr_f32(parts[0])
        var R = iface_dim(parts[0], 0)
        var k_in = ptr_f32(parts[1])
        var K = iface_dim(parts[1], 0)
        var boff = (
            self._off_col_blocks + cfg.col_num_inds * e + self._col_attn_size
        )
        if len(parts) > 2 and iface_dim(parts[2], 0) > 0:
            boff = Int(ptr_i64(parts[2])[0])
        var rope_freqs: Optional[Pointer[Float32, MutUntrackedOrigin]] = None
        if len(parts) > 3 and iface_dim(parts[3], 0) > 0:
            rope_freqs = ptr_f32(parts[3])
        var rope_interleaved = True
        if len(parts) > 4 and iface_dim(parts[4], 0) > 0:
            rope_interleaved = Int(ptr_i64(parts[4])[0]) == 1
        var p = attn_params_at(
            self.params, boff, e, cff, nh, chd, 0,
            cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
        )
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)

        var qn = alloc[Float32](R * e + 16)
        layer_norm_affine(q_in, qn, R, e, p.ln1_w, p.ln1_b, True, p.ln_has_bias)
        var kn = alloc[Float32](K * e + 16)
        layer_norm_affine(k_in, kn, K, e, p.ln1_w, p.ln1_b, True, p.ln_has_bias)
        var q_proj = alloc[Float32](R * e + 16)
        var k_proj = alloc[Float32](K * e + 16)
        var v_proj = alloc[Float32](K * e + 16)
        # DEBUG: copy weight/bias blocks to fresh buffers first
        var wcopy = alloc[Float32](3 * e * e + 16)
        var bcopy = alloc[Float32](3 * e + 16)
        var wi = 0
        while wi < 3 * e * e:
            wcopy[wi] = p.w_qkv[wi]
            wi += 1
        var bi2 = 0
        while bi2 < 3 * e:
            bcopy[bi2] = p.b_qkv[bi2]
            bi2 += 1
        gemm_nt(pool_ptr, R, e, e, qn, wcopy, q_proj)
        # dump raw gemm output before bias
        var qproj_raw = alloc[Float32](R * e + 16)
        var ri = 0
        while ri < R * e:
            qproj_raw[ri] = q_proj[ri]
            ri += 1
        _add_row_bias(q_proj, bcopy, R, e)
        gemm_nt(pool_ptr, K, e, e, kn, wcopy + e * e, k_proj)
        _add_row_bias(k_proj, bcopy + e, K, e)
        gemm_nt(pool_ptr, K, e, e, kn, wcopy + 2 * e * e, v_proj)
        _add_row_bias(v_proj, bcopy + 2 * e, K, e)

        var q_a = alloc[Float32](R * nh * chd + 16)
        var k_a = alloc[Float32](K * nh * chd + 16)
        var v_a = alloc[Float32](K * nh * chd + 16)
        var qi = 0
        while qi < R:
            var h = 0
            while h < nh:
                var dd = 0
                while dd < chd:
                    q_a[h * R * chd + qi * chd + dd] = q_proj[qi * e + h * chd + dd]
                    dd += 1
                h += 1
            qi += 1
        var ki = 0
        while ki < K:
            var h = 0
            while h < nh:
                var dd = 0
                while dd < chd:
                    k_a[h * K * chd + ki * chd + dd] = k_proj[ki * e + h * chd + dd]
                    v_a[h * K * chd + ki * chd + dd] = v_proj[
                        ki * e + h * chd + dd
                    ]
                    dd += 1
                h += 1
            ki += 1

        var out_a = alloc[Float32](R * nh * chd + 16)
        if rope_freqs is not None:
            rope_apply(
                q_a, k_a, R, K, nh, chd,
                rope_freqs.value(), rope_interleaved,
            )
        var scores = alloc[Float32](R * K + 16)
        var sc_pre = alloc[Float32](R * K + 16)
        var scale = 1.0 / sqrt(Float32(chd))
        var h2 = 0
        while h2 < nh:
            var q_h = q_a + h2 * R * chd
            var k_h = k_a + h2 * K * chd
            var v_h = v_a + h2 * K * chd
            var o_h = out_a + h2 * R * chd
            gemm_nt(pool_ptr, R, K, chd, q_h, k_h, scores)
            if h2 == 0:
                var cp = 0
                while cp < R * K:
                    sc_pre[cp] = scores[cp]
                    cp += 1
            var ts = 0
            while ts < R:
                var base_s = ts * K
                var mx: Float32 = -3.0e38
                var s = 0
                while s < K:
                    var v = scores[base_s + s] * scale
                    scores[base_s + s] = v
                    if v > mx:
                        mx = v
                    s += 1
                var sum_v: Float32 = 0.0
                s = 0
                while s < K:
                    var ev = exp(scores[base_s + s] - mx)
                    scores[base_s + s] = ev
                    sum_v += ev
                    s += 1
                s = 0
                while s < K:
                    scores[base_s + s] /= sum_v
                    s += 1
                ts += 1
            gemm_nn(pool_ptr, R, chd, K, scores, v_h, o_h)
            h2 += 1

        var merged = alloc[Float32](R * e + 16)
        var mi = 0
        while mi < R:
            var hh = 0
            while hh < nh:
                var dd = 0
                while dd < chd:
                    merged[mi * e + hh * chd + dd] = out_a[
                        hh * R * chd + mi * chd + dd
                    ]
                    dd += 1
                hh += 1
            mi += 1
        var res = alloc[Float32](R * e + 16)
        gemm_nt(pool_ptr, R, e, e, merged, p.w_out, res)
        _add_row_bias(res, p.b_out, R, e)
        var dst = alloc[Float32](R * e + 16)
        mi = 0
        while mi < R * e:
            dst[mi] = q_in[mi] + res[mi]
            mi += 1

        # extra sections: raw params seen by this block + raw q_proj pre-bias
        # + final q_proj snapshot (corruption bisect)
        var w_qkv = p.w_qkv
        var total_params = 3 * e * e + 3 * e + 2 * e + 3 * R * e + R * K
        var pdump = alloc[Float32](total_params)
        var pi = 0
        while pi < 3 * e * e:
            pdump[pi] = w_qkv[pi]
            pi += 1
        var bi = 0
        while bi < 3 * e:
            pdump[pi] = p.b_qkv[bi]
            pi += 1
            bi += 1
        var li = 0
        while li < e:
            pdump[pi] = p.ln1_w[li]
            pi += 1
            li += 1
        li = 0
        while li < e:
            pdump[pi] = p.ln1_b[li]
            pi += 1
            li += 1
        ri = 0
        while ri < R * e:
            pdump[pi] = qproj_raw[ri]
            pi += 1
            ri += 1
        ri = 0
        while ri < R * e:
            pdump[pi] = q_proj[ri]
            pi += 1
            ri += 1
        # DEBUG: marker write between snapshot and dump loop
        ri = 0
        while ri < R * e:
            pdump[pi] = q_proj[ri]
            pi += 1
            ri += 1
        ri = 0
        while ri < R * K:
            pdump[pi] = sc_pre[ri]
            pi += 1
            ri += 1

        # dump: qn, kn, q_proj, k_proj, v_proj, q_a, k_a, v_a, scores,
        #       out_a, merged, res, dst, params
        var np = np_module()
        var total = (
            5 * R * e
            + 3 * K * e
            + 2 * R * nh * chd
            + 2 * K * nh * chd
            + R * K
            + total_params
        )
        var out_arr = np.empty(total)
        var ob = ptr_f64(out_arr)
        var cursor = 0
        var i2 = 0
        while i2 < 14:
            var sz: Int
            var src_ptr: Pointer[Float32, MutUntrackedOrigin]
            if i2 == 0:
                src_ptr = qn
                sz = R * e
            elif i2 == 1:
                src_ptr = kn
                sz = K * e
            elif i2 == 2:
                src_ptr = q_proj
                sz = R * e
            elif i2 == 3:
                src_ptr = k_proj
                sz = K * e
            elif i2 == 4:
                src_ptr = v_proj
                sz = K * e
            elif i2 == 5:
                src_ptr = q_a
                sz = R * nh * chd
            elif i2 == 6:
                src_ptr = k_a
                sz = K * nh * chd
            elif i2 == 7:
                src_ptr = v_a
                sz = K * nh * chd
            elif i2 == 8:
                src_ptr = scores
                sz = R * K
            elif i2 == 9:
                src_ptr = out_a
                sz = R * nh * chd
            elif i2 == 10:
                src_ptr = merged
                sz = R * e
            elif i2 == 11:
                src_ptr = res
                sz = R * e
            else:
                src_ptr = dst
                sz = R * e
            if i2 == 13:
                src_ptr = pdump
                sz = total_params
            var j2 = 0
            while j2 < sz:
                ob[cursor + j2] = Float64(src_ptr[j2])
                j2 += 1
            cursor += sz
            i2 += 1
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return out_arr

    @staticmethod
    def stage_col(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """Run the column embedding stage.

        ``parts = (x, target)`` where ``x`` is ``(n_rows, n_features)``
        float32 and ``target`` is ``(train_size,)`` int64 class labels
        (classification) or float32 scaled targets (regression). Returns
        a flat float32 array of ``n_rows * (row_num_cls + n_features) *
        embed_dim`` values in (row, position, channel) order.
        """
        var self = self_ptr[]
        var x = ptr_f32(parts[0])
        var n_rows = iface_dim(parts[0], 0)
        var n_features = iface_dim(parts[0], 1)
        var train_size = iface_dim(parts[1], 0)
        var np = np_module()
        var total = (
            n_rows * (self.config.row_num_cls + n_features) * self.config.embed_dim
        )
        var out_arr = np.empty(Python.tuple(Int(total)), "float32")
        var col_out = ptr_f32(out_arr)
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        try:
            _ = self.stage_col_impl(
                pool_ptr, x, parts[1], n_rows, train_size, n_features, col_out
            )
        except e:
            pool_shutdown(pool_ptr)
            pool_ptr.free()
            raise e
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return out_arr

    # =====================================================================
    # Stage 2: row interaction (staged API)
    # =====================================================================

    def stage_row_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        col_out: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, G*E)
        n_rows: Int,
        n_groups: Int,
        reps: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, num_cls*E)
    ) raises -> Int:
        """Row interaction stage (torch RowInteraction.forward).

        ``col_out`` holds the column-embedding stage output in
        (row, position, channel) order with position count ``n_groups``.
        The first ``row_num_cls`` positions of every row are replaced by
        the learned CLS tokens, the transformer blocks run over each
        row's ``n_groups`` positions independently (batched over rows;
        RoPE positions restart at 0 within every row), and the final
        block's cross-attention aggregates the CLS queries over the full
        sequence. Results are LayerNormed with ``out_ln`` and returned in
        (row, cls, channel) order. Returns 0 on success.
        """
        var cfg = self.config
        var e = cfg.embed_dim
        var cff = cfg.col_dim_feedforward()
        var rhd = e // cfg.row_nhead
        var num_cls = cfg.row_num_cls
        if cfg.icl_dim != num_cls * e:
            raise Error(
                "row stage output width (row_num_cls * embed_dim) must "
                "equal icl_dim for the staged pipeline"
            )
        if n_groups <= num_cls:
            raise Error("n_groups must exceed row_num_cls")

        var cls_tokens = self.params + self._off_cls_tokens
        var ln_w = self.params + self._off_row_out_ln_w
        var ln_b = self.params + self._off_row_out_ln_b
        var has_bias = not cfg.bias_free_ln

        var ge = n_groups * e
        var ce = num_cls * e
        var cur = alloc[Float32](ge + 16)

        var rope_freqs = self.params + self._off_rope_freqs
        var last_blk = cfg.row_num_blocks - 1
        var t = 0
        while t < n_rows:
            # Load this row's positions and replace the CLS slots.
            var src_base = t * ge
            var i = 0
            while i + SIMDW <= ge:
                cur.unsafe_store[width=SIMDW](
                    i, col_out.unsafe_load[width=SIMDW](src_base + i)
                )
                i += SIMDW
            while i < ge:
                cur[i] = col_out[src_base + i]
                i += 1
            i = 0
            while i + SIMDW <= ce:
                cur.unsafe_store[width=SIMDW](
                    i, cls_tokens.unsafe_load[width=SIMDW](i)
                )
                i += SIMDW
            while i < ce:
                cur[i] = cls_tokens[i]
                i += 1

            # Blocks except the last: full self-attention over positions.
            var blk = 0
            while blk < last_blk:
                var p = attn_params_at(
                    self.params,
                    self._off_row_blocks + blk * self._row_attn_size,
                    e, cff, cfg.row_nhead, rhd, 0,
                    cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
                )
                attention_block_forward(pool, 
                    cur, None, cur, n_groups, n_groups, e,
                    cfg.row_nhead, rhd, cff, p,
                    rope_freqs, cfg.row_rope_interleaved,
                )
                blk += 1

            # Last block: CLS queries attend to the full sequence.
            var plast = attn_params_at(
                self.params,
                self._off_row_blocks + last_blk * self._row_attn_size,
                e, cff, cfg.row_nhead, rhd, 0,
                cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
            )
            attention_block_forward(pool, 
                cur, cur, cur, num_cls, n_groups, e,
                cfg.row_nhead, rhd, cff, plast,
                rope_freqs, cfg.row_rope_interleaved,
            )

            layer_norm_affine(cur, cur, num_cls, e, ln_w, ln_b, True, has_bias)
            var dst_base = t * ce
            i = 0
            while i + SIMDW <= ce:
                reps.unsafe_store[width=SIMDW](
                    dst_base + i, cur.unsafe_load[width=SIMDW](i)
                )
                i += SIMDW
            while i < ce:
                reps[dst_base + i] = cur[i]
                i += 1
            t += 1

        cur.unsafe_free()
        return 0

    @staticmethod
    def stage_row(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """Run the row interaction stage.

        ``parts = (col_out, n_groups)`` where ``col_out`` is the flat
        float32 output of ``stage_col`` ((n_rows, n_groups, embed_dim),
        row-major) and ``n_groups`` is the position count (= row_num_cls
        + n_features). Returns a flat float32 array of ``n_rows *
        row_num_cls * embed_dim`` values in (row, cls, channel) order.
        """
        var self = self_ptr[]
        var col_out = ptr_f32(parts[0])
        var n_rows = iface_dim(parts[0], 0)
        var n_groups = Int(ptr_i64(parts[1])[0])
        var np = np_module()
        var total = n_rows * self.config.row_num_cls * self.config.embed_dim
        var out_arr = np.empty(Python.tuple(Int(total)), "float32")
        var reps = ptr_f32(out_arr)
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        try:
            _ = self.stage_row_impl(pool_ptr, col_out, n_rows, n_groups, reps)
        except e:
            pool_shutdown(pool_ptr)
            pool_ptr.free()
            raise e
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return out_arr

    # =====================================================================
    # Stage 3: ICL prediction from representations (staged API)
    # =====================================================================

    def _y_encode_reps_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        reps: Pointer[Float32, MutUntrackedOrigin],  # (train_size, icl_dim)
        target: PythonObject,
        train_size: Int,
    ) raises -> Int:
        """Add the encoded targets to train-row representations, in place.

        Shared verbatim by ``predict_from_representations_impl`` and
        ``build_cache`` so both augment the train prefix identically.
        Returns 0 on success.
        """
        var cfg = self.config
        var d_icl = cfg.icl_dim
        var is_clf = cfg.max_classes > 0
        var target_i64 = ptr_i64(target)
        var target_f32 = ptr_f32(target)
        var y_enc_w = self.params + self._off_icl_y_encoder_w
        var y_enc_b = self.params + self._off_icl_y_encoder_b
        var t = 0
        while t < train_size:
            var base = t * d_icl
            if is_clf:
                var c = Int(target_i64[t])
                if c < 0 or c >= cfg.max_classes:
                    raise Error("class label out of range")
                var j = 0
                while j < d_icl:
                    reps[base + j] += (
                        y_enc_w[j * cfg.max_classes + c] + y_enc_b[j]
                    )
                    j += 1
            else:
                var v = target_f32[t]
                var j = 0
                while j < d_icl:
                    reps[base + j] += y_enc_w[j] * v + y_enc_b[j]
                    j += 1
            t += 1
        return 0

    def _icl_decode_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        reps: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, icl_dim)
        n_rows: Int,
        dst: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, out_dim)
    ) raises -> Int:
        """Predictor LayerNorm + two-layer decoder head (shared verbatim).

        Writes raw (untempered) logits or quantiles for all ``n_rows``;
        callers slice away the train prefix when running the plain path.
        Returns 0 on success.
        """
        var cfg = self.config
        var d_icl = cfg.icl_dim
        layer_norm_affine(
            reps, reps, n_rows, d_icl,
            self.params + self._off_icl_ln_w,
            self.params + self._off_icl_ln_b,
            True, not cfg.bias_free_ln,
        )

        var decoded = alloc[Float32](n_rows * d_icl * 2 + 16)
        gemm_nt(pool, n_rows, d_icl * 2, d_icl, reps, self.params + self._off_decoder_w1, decoded)
        var t = 0
        while t < n_rows:
            var b = 0
            while b + SIMDW <= d_icl * 2:
                decoded.unsafe_store[width=SIMDW](
                    t * d_icl * 2 + b,
                    decoded.unsafe_load[width=SIMDW](t * d_icl * 2 + b)
                    + self.params.unsafe_load[width=SIMDW](self._off_decoder_b1 + b),
                )
                b += SIMDW
            while b < d_icl * 2:
                decoded[t * d_icl * 2 + b] += self.params[self._off_decoder_b1 + b]
                b += 1
            t += 1
        t = 0
        var ptotal = n_rows * d_icl * 2
        while t + SIMDW <= ptotal:
            decoded.unsafe_store[width=SIMDW](
                t,
                gelu8(decoded.unsafe_load[width=SIMDW](t)),
            )
            t += SIMDW
        while t < ptotal:
            decoded[t] = gelu_scalar(decoded[t])
            t += 1

        gemm_nt(pool, n_rows, cfg.out_dim, d_icl * 2, decoded, self.params + self._off_decoder_w2, dst)
        t = 0
        while t < n_rows:
            var b = 0
            while b + SIMDW <= cfg.out_dim:
                dst.unsafe_store[width=SIMDW](
                    t * cfg.out_dim + b,
                    dst.unsafe_load[width=SIMDW](t * cfg.out_dim + b)
                    + self.params.unsafe_load[width=SIMDW](self._off_decoder_b2 + b),
                )
                b += SIMDW
            while b < cfg.out_dim:
                dst[t * cfg.out_dim + b] += self.params[self._off_decoder_b2 + b]
                b += 1
            t += 1
        decoded.unsafe_free()
        return 0

    def predict_from_representations_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        reps: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, icl_dim)
        target: PythonObject,
        n_rows: Int,
        dst: Pointer[Float32, MutUntrackedOrigin],  # (n_rows, out_dim)
    ) raises -> Int:
        """ICL stage (torch ICLearning.predict_standard internals).

        Train-row representations are augmented with the encoded targets,
        every ICL block attends all rows to the train prefix, and the
        decoder maps the post-LN features to raw (untempered) logits or
        quantiles for ALL rows; callers slice away the train prefix.
        Returns 0 on success.
        """
        var cfg = self.config
        var d_icl = cfg.icl_dim
        var train_size = iface_dim(target, 0)
        if train_size <= 0 or train_size > n_rows:
            raise Error("target length must be within [1, n_rows]")

        # y_encoder augmentation on the train prefix (shared helper).
        _ = self._y_encode_reps_impl(pool, reps, target, train_size)

        # ICL blocks: plain pre-norm attention whose keys/values are
        # restricted to the train prefix (torch Encoder.forward with
        # train_size). Rope/ssmax are unused unless configured.
        var ihd = cfg.icl_head_dim()
        var icl_ff = cfg.icl_dim_feedforward()
        var blk = 0
        while blk < cfg.icl_num_blocks:
            var off = self._off_icl_blocks + blk * self._icl_attn_size
            var p = attn_params_at(
                self.params, off, d_icl, icl_ff,
                cfg.icl_nhead, ihd, cfg.icl_ssmax_kind,
                cfg.icl_ssmax_n_hidden, cfg.bias_free_ln,
            )
            attention_block_forward(pool, 
                reps, reps, reps, n_rows, train_size, d_icl,
                cfg.icl_nhead, ihd, icl_ff, p, None, False,
            )
            blk += 1

        # Predictor LayerNorm + decoder head (shared helper).
        _ = self._icl_decode_impl(pool, reps, n_rows, dst)
        return 0

    @staticmethod
    def predict_from_representations(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """Run the ICL stage on row representations.

        ``parts = (reps, target)`` where ``reps`` is ``(n_rows, icl_dim)``
        float32 (the ``stage_row`` output incl. the train prefix),
        ``target`` is ``(train_size,)`` int64 labels (classification) or
        float32 scaled targets (regression). Returns a flat float32 array
        of ``n_rows * out_dim`` RAW decoder outputs in row-major order;
        callers select the test rows (and class columns) themselves.
        """
        var self = self_ptr[]
        var reps = ptr_f32(parts[0])
        var n_rows = iface_dim(parts[0], 0)
        if iface_dim(parts[0], 1) != self.config.icl_dim:
            raise Error(
                "representations must have icl_dim columns; got width",
            )
        var np = np_module()
        var total = n_rows * self.config.out_dim
        var out_arr = np.empty(Python.tuple(Int(total)), "float32")
        var outp = ptr_f32(out_arr)
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        try:
            _ = self.predict_from_representations_impl(
                pool_ptr, reps, parts[1], n_rows, outp
            )
        except e:
            pool_shutdown(pool_ptr)
            pool_ptr.free()
            raise e
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return out_arr

    # =====================================================================
    # KV-cache: build (train side) and predict (test side)
    # =====================================================================

    def stage_col_cached_impl(
        mut self,
        pool: UnsafePointer[ThreadPool, MutUntrackedOrigin],
        x: Pointer[Float32, MutUntrackedOrigin],       # (n_test, n_features)
        n_test: Int,
        train_size: Int,
        n_features: Int,
        col_out: Pointer[Float32, MutUntrackedOrigin],  # (n_test, G, E) t-major
        col_cache: Pointer[Float32, MutUntrackedOrigin],  # [g][blk][kv][n_inds*e]
    ) raises -> Int:
        """Cached column stage (torch ColEmbedding.predict_with_cache path).

        Skips attn1 entirely — attn2's K/V come from the prebuilt cache —
        and test rows never receive the target augmentation. Sentinel
        positions (g < row_num_cls) stay SKIP_VALUE and skip compute; CLS
        tokens overwrite their outputs downstream. Returns 0 on success.
        """
        var cfg = self.config
        var e = cfg.embed_dim
        var s = cfg.col_feature_group_size
        var num_cls = cfg.row_num_cls
        var G = num_cls + n_features
        var n_inds = cfg.col_num_inds
        var chd = cfg.col_head_dim()
        var cff = cfg.col_dim_feedforward()

        var pos = alloc[Float32](n_test * s + 16)
        var cur = alloc[Float32](n_test * e + 16)

        var g = num_cls
        while g < G:
            if g < num_cls:
                var t = 0
                while t < n_test:
                    var j = 0
                    while j < e:
                        col_out[t * G * e + g * e + j] = SKIP_VALUE_F32
                        j += 1
                    t += 1
            else:
                var h = g - num_cls
                var t = 0
                while t < n_test:
                    var i = 0
                    while i < s:
                        pos[t * s + i] = x[
                            t * n_features + (h + (1 << i)) % n_features
                        ]
                        i += 1
                    t += 1
                gemm_nt(pool, 
                    n_test, e, s, pos, self.params + self._off_in_linear_w, cur
                )
                var b = self._off_in_linear_b
                t = 0
                while t < n_test:
                    var j = 0
                    while j + SIMDW <= e:
                        cur.unsafe_store[width=SIMDW](
                            t * e + j,
                            cur.unsafe_load[width=SIMDW](t * e + j)
                            + self.params.unsafe_load[width=SIMDW](b + j),
                        )
                        j += SIMDW
                    while j < e:
                        cur[t * e + j] += self.params[b + j]
                        j += 1
                    t += 1

                # Cached blocks: only attn2 runs, against the cached K/V.
                var blk = 0
                while blk < cfg.col_num_blocks:
                    var boff = (
                        self._off_col_blocks
                        + blk * self._col_blk_stride
                        + n_inds * e
                        + self._col_attn_size
                    )
                    var p2 = attn_params_at(
                        self.params, boff,
                        e, cff, cfg.col_nhead, chd, 0,
                        cfg.col_ssmax_n_hidden, cfg.bias_free_ln,
                    )
                    var slot = (g * cfg.col_num_blocks + blk) * 2 * n_inds * e
                    attention_block_forward_cached(pool, 
                        cur, cur, n_test, n_inds,
                        e, cfg.col_nhead, chd, cff, p2,
                        col_cache + slot,
                        col_cache + slot + n_inds * e,
                    )
                    blk += 1

                # Scatter this position's embeddings into row-major order.
                t = 0
                while t < n_test:
                    var src_base = t * e
                    var dst_base = t * G * e + g * e
                    var j = 0
                    while j + SIMDW <= e:
                        col_out.unsafe_store[width=SIMDW](
                            dst_base + j,
                            cur.unsafe_load[width=SIMDW](src_base + j),
                        )
                        j += SIMDW
                    while j < e:
                        col_out[dst_base + j] = cur[src_base + j]
                        j += 1
                    t += 1
            g += 1

        pos.unsafe_free()
        cur.unsafe_free()
        return 0

    @staticmethod
    def build_cache(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """Pre-compute the col and ICL K/V caches from training data.

        ``parts = (x, target)`` where ``x`` is ``(train_size, n_features)``
        float32 training rows and ``target`` is ``(train_size,)`` int64
        labels or float32 scaled targets. Returns ``(col_cache,
        icl_cache)``, flat float32 arrays laid out as ``[g][blk][kv][
        n_inds*embed_dim]`` and ``[blk][kv][train_size*icl_dim]``;
        sentinel position slots are zero-filled and never read.
        """
        var self = self_ptr[]
        var x = ptr_f32(parts[0])
        var target = parts[1]
        var n_features = iface_dim(parts[0], 1)
        var train_size = iface_dim(target, 0)
        if train_size < 1:
            raise Error("build_cache needs at least one training row")

        var cfg = self.config
        var e = cfg.embed_dim
        var num_cls = cfg.row_num_cls
        var G = num_cls + n_features
        var n_inds = cfg.col_num_inds
        var d_icl = cfg.icl_dim
        var ihd = cfg.icl_head_dim()
        var icl_ff = cfg.icl_dim_feedforward()
        var ce = num_cls * e

        var np = np_module()
        var col_total = G * cfg.col_num_blocks * 2 * n_inds * e
        var icl_total = cfg.icl_num_blocks * 2 * train_size * d_icl
        var col_arr = np.zeros(Python.tuple(Int(col_total)), "float32")
        var icl_arr = np.zeros(Python.tuple(Int(icl_total)), "float32")
        var col_cache = ptr_f32(col_arr)
        var icl_cache = ptr_f32(icl_arr)

        # Col stage over the training rows with K/V capture.
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        try:
            var col_out = alloc[Float32](train_size * G * e + 16)
            _ = self.stage_col_impl(
                pool_ptr,
                x, target, train_size, train_size, n_features, col_out,
                col_cache
            )

            # Row stage gives the train representations; y-encode them in
            # place exactly like predict_from_representations_impl does.
            var reps = alloc[Float32](train_size * ce + 16)
            _ = self.stage_row_impl(pool_ptr, col_out, train_size, G, reps)
            _ = self._y_encode_reps_impl(pool_ptr, reps, target, train_size)

            # ICL blocks capture K/V restricted to the train prefix.
            var blk = 0
            while blk < cfg.icl_num_blocks:
                var off = self._off_icl_blocks + blk * self._icl_attn_size
                var p = attn_params_at(
                    self.params, off, d_icl, icl_ff,
                    cfg.icl_nhead, ihd, cfg.icl_ssmax_kind,
                    cfg.icl_ssmax_n_hidden, cfg.bias_free_ln,
                )
                var slot = blk * 2 * train_size * d_icl
                attention_block_forward(
                    pool_ptr,
                    reps, reps, reps, train_size, train_size, d_icl,
                    cfg.icl_nhead, ihd, icl_ff, p, None, False,
                    icl_cache + slot,
                    icl_cache + slot + train_size * d_icl,
                )
                blk += 1

            col_out.unsafe_free()
            reps.unsafe_free()
        except e:
            pool_shutdown(pool_ptr)
            pool_ptr.free()
            raise e
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return Python.tuple(col_arr, icl_arr)

    @staticmethod
    def predict_with_cache(
        self_ptr: Pointer[Self, MutAnyOrigin],
        parts: PythonObject,
    ) raises -> PythonObject:
        """Predict test rows against prebuilt caches.

        ``parts = (x_test, target, col_cache, icl_cache)`` where ``x_test``
        is ``(n_test, n_features)`` float32 test rows, ``target`` supplies
        the train size (and label validation for classification), and the
        caches are the flat arrays returned by ``build_cache``. Returns a
        flat float32 array of ``n_test * out_dim`` RAW decoder outputs in
        row-major order (same contract as
        ``predict_from_representations``).
        """
        var self = self_ptr[]
        var x = ptr_f32(parts[0])
        var n_test = iface_dim(parts[0], 0)
        var n_features = iface_dim(parts[0], 1)
        var target = parts[1]
        var train_size = iface_dim(target, 0)
        if train_size < 1:
            raise Error("cache target length must be at least 1")
        if n_test < 1:
            raise Error("predict_with_cache needs at least one test row")

        var cfg = self.config
        var e = cfg.embed_dim
        var num_cls = cfg.row_num_cls
        var G = num_cls + n_features
        var n_inds = cfg.col_num_inds
        var d_icl = cfg.icl_dim
        var ihd = cfg.icl_head_dim()
        var icl_ff = cfg.icl_dim_feedforward()
        var ce = num_cls * e

        var col_cache = ptr_f32(parts[2])
        var icl_cache = ptr_f32(parts[3])
        var col_total = G * cfg.col_num_blocks * 2 * n_inds * e
        var icl_total = cfg.icl_num_blocks * 2 * train_size * d_icl
        if iface_dim(parts[2], 0) != col_total:
            raise Error("col cache size mismatch for this model/features")
        if iface_dim(parts[3], 0) != icl_total:
            raise Error("icl cache size mismatch for this train size")

        var col_out = alloc[Float32](n_test * G * e + 16)
        var np = np_module()
        var total = n_test * cfg.out_dim
        var out_arr = np.empty(Python.tuple(Int(total)), "float32")
        var pool_ptr = UnsafePointer[ThreadPool, MutUntrackedOrigin](
            alloc[ThreadPool](1)
        )
        pool_init(pool_ptr, 0)
        try:
            _ = self.stage_col_cached_impl(
                pool_ptr, x, n_test, train_size, n_features, col_out, col_cache
            )
            var reps = alloc[Float32](n_test * ce + 16)
            _ = self.stage_row_impl(pool_ptr, col_out, n_test, G, reps)

            # ICL blocks attend the test queries to the cached train K/V,
            # then the shared LayerNorm + decoder produces raw outputs.
            var blk = 0
            while blk < cfg.icl_num_blocks:
                var off = self._off_icl_blocks + blk * self._icl_attn_size
                var p = attn_params_at(
                    self.params, off, d_icl, icl_ff,
                    cfg.icl_nhead, ihd, cfg.icl_ssmax_kind,
                    cfg.icl_ssmax_n_hidden, cfg.bias_free_ln,
                )
                var slot = blk * 2 * train_size * d_icl
                attention_block_forward_cached(
                    pool_ptr,
                    reps, reps, n_test, train_size,
                    d_icl, cfg.icl_nhead, ihd, icl_ff, p,
                    icl_cache + slot,
                    icl_cache + slot + train_size * d_icl,
                )
                blk += 1

            _ = self._icl_decode_impl(pool_ptr, reps, n_test, ptr_f32(out_arr))

            col_out.unsafe_free()
            reps.unsafe_free()
        except e:
            pool_shutdown(pool_ptr)
            pool_ptr.free()
            raise e
        pool_shutdown(pool_ptr)
        pool_ptr.free()
        return out_arr


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit__native_tabicl() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native_tabicl")
        _ = (
            m.add_type[TabICLInference]("TabICLInference")
            .def_py_init[TabICLInference.py_init]()
            .def_method[TabICLInference.forward]("forward")
            .def_method[TabICLInference.stage_col]("stage_col")
            .def_method[TabICLInference.stage_row]("stage_row")
            .def_method[TabICLInference.predict_from_representations](
                "predict_from_representations"
            )
            .def_method[TabICLInference.build_cache]("build_cache")
            .def_method[TabICLInference.predict_with_cache]("predict_with_cache")
            .def_method[TabICLInference.attn_probe]("attn_probe")
            .def_method[TabICLInference.gemm_probe]("gemm_probe")
            .def_method[TabICLInference.param_count]("param_count")
            .def_method[TabICLInference.layout_offsets]("layout_offsets")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module _native_tabicl: ", e))
