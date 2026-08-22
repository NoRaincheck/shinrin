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
from std.math import exp, max
from std.memory import alloc
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from shinrin._tk_core import (
    SIMDW,
    gemm_nn,
    gemm_nt,
    gelu8,
    gelu_scalar,
    iface_dim,
    np_module,
    ptr_f32,
    ptr_i64,
)
from shinrin._tk_layers import (
    AttnParams,
    attention_block_forward,
    attention_block_size,
    attn_params_at,
    isab_forward,
    layer_norm_affine,
)



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
        if self.max_classes <= 0:
            raise Error(
                "native TabICL kernel requires classification mode (max_classes > 0)"
            )

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
            self._off_col_y_enc_w = cur
            cur += e * cfg.max_classes
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
        self._off_icl_y_encoder_w = cur
        cur += icl_d * cfg.max_classes
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
        _ = self.forward_impl(
            col_input, row_input, target, n_train, n_test, n_classes, output
        )
        return out_arr

    def forward_impl(
        mut self,
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
        gemm_nt(
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
                    + SIMD[DType.float32, SIMDW](self.params[self._off_in_linear_b + b]),
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
            isab_forward(
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
            attention_block_forward(
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
            attention_block_forward(
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
        gemm_nt(mc, d_icl * 2, d_icl, icl_cur, decoder_w1, decoded)
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
        while i < mc * d_icl * 2:
            decoded[i] = gelu_scalar(decoded[i])
            i += 1

        var logits = alloc[Float32](mc * cfg.out_dim + 16)
        gemm_nt(mc, cfg.out_dim, d_icl * 2, decoded, decoder_w2, logits)
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
        gemm_nt(n_test, mc, d_icl, test_proj, class_embed, attn_scores)

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

        gemm_nn(n_test, cfg.out_dim, mc, attn_scores, logits, output)

        col_embed.unsafe_free()
        row_combined.unsafe_free()
        class_embed.unsafe_free()
        y_encoded.unsafe_free()
        logits.unsafe_free()
        test_proj.unsafe_free()
        attn_scores.unsafe_free()

        return 0


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
            .def_method[TabICLInference.param_count]("param_count")
            .def_method[TabICLInference.layout_offsets]("layout_offsets")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module _native_tabicl: ", e))
