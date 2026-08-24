"""Mojo port of shinrin's native tree extension (Rust: src/lib.rs).

This module compiles to the ``shinrin._native_mojo_core`` Python extension
module. It exposes plain bound types with method-only APIs; the pure-Python
facade in ``shinrin/_native_mojo.py`` wraps them to provide the exact
property/kwargs/pickle surface of the Rust ``shinrin._native`` module.

Original work: scikit-garden contributors (BSD 3-clause); see NOTICE.
"""

from std.os import abort
from std.math import inf, log, exp, sqrt
from std.memory import alloc
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

comptime TREE_LEAF = -1
comptime TREE_UNDEFINED = -2

comptime INF_F32 = inf[DType.float32]()
comptime INF_F64 = inf[DType.float64]()

# =============================================================================
# Small helpers
# =============================================================================


def imin(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


def fmin(a: Float64, b: Float64) -> Float64:
    if a < b:
        return a
    return b


def fmax(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


def f32min(a: Float32, b: Float32) -> Float32:
    if a < b:
        return a
    return b


def f32max(a: Float32, b: Float32) -> Float32:
    if a > b:
        return a
    return b


# =============================================================================
# RNG (exact port of the Rust helpers)
# =============================================================================

comptime RAND_R_MAX: UInt32 = 0x7FFFFFFF


def our_rand_r(mut seed: UInt32) -> UInt32:
    seed ^= seed << 13
    seed ^= seed >> 17
    seed ^= seed << 5
    return seed % (RAND_R_MAX + 1)


def rand_uniform(low: Float64, high: Float64, mut seed: UInt32) -> Float64:
    return (high - low) * Float64(our_rand_r(seed)) / Float64(RAND_R_MAX) + low


def rand_multinomial(pvals: List[Float32], mut seed: UInt32) -> Int:
    var n = len(pvals)
    var cum = List[Float32]()
    for _ in range(n):
        cum.append(0.0)
    cum[0] = pvals[0]
    for j in range(1, n):
        cum[j] = cum[j - 1] + pvals[j]
    var search = rand_uniform(0.0, Float64(cum[n - 1]), seed)
    var f_j = 0
    for j in range(n):
        var lower: Float64 = 0.0
        if j != 0:
            lower = Float64(cum[j - 1])
        if Float64(cum[j]) >= search and lower < search:
            f_j = j
            break
    return f_j


def rand_exponential(rate: Float32, mut seed: UInt32) -> Float64:
    return -log(rand_uniform(0.0, 1.0, seed)) / Float64(rate)


# =============================================================================
# numpy helpers
# =============================================================================


def numpy_mod() raises -> PythonObject:
    return Python.import_module("numpy")


def ptr_f32(arr: PythonObject) raises -> Pointer[Float32, MutUntrackedOrigin]:
    var addr = Int(py=arr.__array_interface__["data"][0])
    return Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def ptr_f64(arr: PythonObject) raises -> Pointer[Float64, MutUntrackedOrigin]:
    var addr = Int(py=arr.__array_interface__["data"][0])
    return Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def ptr_int(arr: PythonObject) raises -> Pointer[Int, MutUntrackedOrigin]:
    var addr = Int(py=arr.__array_interface__["data"][0])
    return Pointer[Int, MutUntrackedOrigin](unsafe_from_address=addr)


def iface_shape(arr: PythonObject) raises -> List[Int]:
    var dims = List[Int]()
    var shape = arr.__array_interface__["shape"]
    for dim in shape:
        dims.append(Int(py=dim))
    return dims^


def np_empty_1d(np: PythonObject, n: Int, dtype: String) raises -> PythonObject:
    return np.empty(Python.tuple(Int(n)), dtype)


def np_empty_2d(np: PythonObject, rows: Int, cols: Int, dtype: String) raises -> PythonObject:
    return np.empty(Python.tuple(Int(rows), Int(cols)), dtype)


def np_empty_3d(np: PythonObject, a: Int, b: Int, c: Int, dtype: String) raises -> PythonObject:
    return np.empty(Python.tuple(Int(a), Int(b), Int(c)), dtype)


def np_array_int_list(np: PythonObject, values: List[Int], dtype: String) raises -> PythonObject:
    var pylist = Python.list()
    for v in values:
        pylist.append(Int(v))
    return np.array(pylist, dtype)


def mk_f64_array_int(np: PythonObject, vals: List[Int]) raises -> PythonObject:
    var arr = np_empty_1d(np, len(vals), "float64")
    var p = ptr_f64(arr)
    for i in range(len(vals)):
        p[i] = Float64(vals[i])
    return arr


def mk_f64_array(np: PythonObject, vals: List[Float64]) raises -> PythonObject:
    var arr = np_empty_1d(np, len(vals), "float64")
    var p = ptr_f64(arr)
    for i in range(len(vals)):
        p[i] = vals[i]
    return arr


def alloc_f64(n: Int) -> Pointer[Float64, MutUntrackedOrigin]:
    var m = n
    if m <= 0:
        m = 1
    return alloc[Float64](m)


def alloc_f32(n: Int) -> Pointer[Float32, MutUntrackedOrigin]:
    var m = n
    if m <= 0:
        m = 1
    return alloc[Float32](m)


# =============================================================================
# Shared fit-time data
# =============================================================================


struct SharedData(Movable, Writable):
    var samples: List[Int]
    var y: Pointer[Float64, MutUntrackedOrigin]
    var y_len: Int
    var y_stride: Int
    var sw: Pointer[Float64, MutUntrackedOrigin]
    var has_sw: Bool

    def __init__(out self):
        self.samples = List[Int]()
        self.y = alloc[Float64](1)
        self.y_len = 0
        self.y_stride = 0
        self.sw = alloc[Float64](1)
        self.has_sw = False

    def weight(self, i: Int) -> Float64:
        if self.has_sw:
            return self.sw[i]
        return 1.0


# =============================================================================
# Criterion
# =============================================================================

comptime CRITERION_REGRESSION = 0
comptime CRITERION_CLASSIFICATION = 1


struct Impurities(ImplicitlyCopyable):
    var left: Float64
    var right: Float64

    def __init__(out self, l: Float64, r: Float64):
        self.left = l
        self.right = r


def make_criterion_regression(n_outputs: Int) -> CoreCriterion:
    var c = CoreCriterion()
    c.kind = CRITERION_REGRESSION
    c.n_outputs = n_outputs
    c.sum_stride = n_outputs
    for _ in range(n_outputs):
        c.sum_total.append(0.0)
        c.sum_left.append(0.0)
        c.sum_right.append(0.0)
    c.n_classes.append(1)
    return c^


def make_criterion_classification(n_outputs: Int, classes_in: List[Int]) -> CoreCriterion:
    var c = CoreCriterion()
    c.kind = CRITERION_CLASSIFICATION
    c.n_outputs = n_outputs
    var stride = 0
    for cl in classes_in:
        c.n_classes.append(cl)
        if cl > stride:
            stride = cl
    c.sum_stride = stride
    for _ in range(n_outputs * stride):
        c.sum_total.append(0.0)
        c.sum_left.append(0.0)
        c.sum_right.append(0.0)
    return c^


struct CoreCriterion(Defaultable, Movable, Writable):
    var kind: Int
    var n_outputs: Int
    var n_node_samples: Int
    var weighted_n_samples: Float64
    var weighted_n_node_samples: Float64
    var weighted_n_left: Float64
    var weighted_n_right: Float64
    var start: Int
    var pos: Int
    var end: Int
    var sum_total: List[Float64]
    var sum_left: List[Float64]
    var sum_right: List[Float64]
    var sq_sum_total: Float64
    var n_classes: List[Int]
    var sum_stride: Int

    def __init__(out self):
        self.kind = CRITERION_REGRESSION
        self.n_outputs = 0
        self.n_node_samples = 0
        self.weighted_n_samples = 0.0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.start = 0
        self.pos = 0
        self.end = 0
        self.sum_total = List[Float64]()
        self.sum_left = List[Float64]()
        self.sum_right = List[Float64]()
        self.sq_sum_total = 0.0
        self.n_classes = List[Int]()
        self.sum_stride = 0


    def init(
        mut self,
        data: SharedData,
        weighted_n_samples: Float64,
        start: Int,
        end: Int,
    ):
        self.start = start
        self.end = end
        self.pos = start
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        if self.kind == CRITERION_REGRESSION:
            for k in range(self.n_outputs):
                self.sum_total[k] = 0.0
            self.sq_sum_total = 0.0
            for p in range(start, end):
                var i = data.samples[p]
                var w = data.weight(i)
                for k in range(self.n_outputs):
                    var y_ik = data.y[i * data.y_stride + k]
                    var w_y_ik = w * y_ik
                    self.sum_total[k] += w_y_ik
                    self.sq_sum_total += w_y_ik * y_ik
                self.weighted_n_node_samples += w
        else:
            for k in range(len(self.sum_total)):
                self.sum_total[k] = 0.0
            for p in range(start, end):
                var i = data.samples[p]
                var w = data.weight(i)
                for k in range(self.n_outputs):
                    var c = Int(data.y[i * data.y_stride + k])
                    self.sum_total[k * self.sum_stride + c] += w
                self.weighted_n_node_samples += w
        self.reset()

    def reset(mut self):
        if self.kind == CRITERION_REGRESSION:
            for k in range(self.n_outputs):
                self.sum_left[k] = 0.0
                self.sum_right[k] = self.sum_total[k]
            self.weighted_n_left = 0.0
            self.weighted_n_right = self.weighted_n_node_samples
            self.pos = self.start
        else:
            for k in range(self.n_outputs):
                var base = k * self.sum_stride
                var n_cls = self.n_classes[k]
                for c in range(n_cls):
                    self.sum_left[base + c] = 0.0
                    self.sum_right[base + c] = self.sum_total[base + c]
            self.weighted_n_left = 0.0
            self.weighted_n_right = self.weighted_n_node_samples
            self.pos = self.start

    def reverse_reset(mut self):
        if self.kind == CRITERION_REGRESSION:
            for k in range(self.n_outputs):
                self.sum_right[k] = 0.0
                self.sum_left[k] = self.sum_total[k]
            self.weighted_n_left = self.weighted_n_node_samples
            self.weighted_n_right = 0.0
            self.pos = self.end
        else:
            for k in range(self.n_outputs):
                var base = k * self.sum_stride
                var n_cls = self.n_classes[k]
                for c in range(n_cls):
                    self.sum_right[base + c] = 0.0
                    self.sum_left[base + c] = self.sum_total[base + c]
            self.weighted_n_left = self.weighted_n_node_samples
            self.weighted_n_right = 0.0
            self.pos = self.end

    def update(mut self, data: SharedData, new_pos: Int):
        if self.kind == CRITERION_REGRESSION:
            var forward = new_pos >= self.pos
            if forward:
                forward = (new_pos - self.pos) <= (self.end - new_pos)
            if forward:
                for p in range(self.pos, new_pos):
                    var i = data.samples[p]
                    var w = data.weight(i)
                    for k in range(self.n_outputs):
                        var y_ik = data.y[i * data.y_stride + k]
                        self.sum_left[k] += w * y_ik
                    self.weighted_n_left += w
            else:
                self.reverse_reset()
                for idx in range(new_pos, self.end):
                    var p = self.end - 1 - (idx - new_pos)
                    var i = data.samples[p]
                    var w = data.weight(i)
                    for k in range(self.n_outputs):
                        var y_ik = data.y[i * data.y_stride + k]
                        self.sum_left[k] -= w * y_ik
                    self.weighted_n_left -= w
            self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
            for k in range(self.n_outputs):
                self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
            self.pos = new_pos
        else:
            var forward = new_pos >= self.pos
            if forward:
                forward = (new_pos - self.pos) <= (self.end - new_pos)
            if forward:
                for p in range(self.pos, new_pos):
                    var i = data.samples[p]
                    var w = data.weight(i)
                    for k in range(self.n_outputs):
                        var label = k * self.sum_stride + Int(data.y[i * data.y_stride + k])
                        self.sum_left[label] += w
                    self.weighted_n_left += w
            else:
                self.reverse_reset()
                for idx in range(new_pos, self.end):
                    var p = self.end - 1 - (idx - new_pos)
                    var i = data.samples[p]
                    var w = data.weight(i)
                    for k in range(self.n_outputs):
                        var label = k * self.sum_stride + Int(data.y[i * data.y_stride + k])
                        self.sum_left[label] -= w
                    self.weighted_n_left -= w
            self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
            for k in range(self.n_outputs):
                var base = k * self.sum_stride
                var n_cls = self.n_classes[k]
                for c in range(n_cls):
                    self.sum_right[base + c] = self.sum_total[base + c] - self.sum_left[base + c]
            self.pos = new_pos

    def node_impurity(self) -> Float64:
        if self.kind == CRITERION_CLASSIFICATION:
            return INF_F64
        var impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            var mean = self.sum_total[k] / self.weighted_n_node_samples
            impurity -= mean * mean
        return impurity / Float64(self.n_outputs)

    def children_impurity(self, data: SharedData) -> Impurities:
        if self.kind == CRITERION_CLASSIFICATION:
            return Impurities(0.0, 0.0)
        var sq_sum_left = 0.0
        for p in range(self.start, self.pos):
            var i = data.samples[p]
            var w = data.weight(i)
            for k in range(self.n_outputs):
                var y_ik = data.y[i * data.y_stride + k]
                sq_sum_left += w * y_ik * y_ik
        var sq_sum_right = self.sq_sum_total - sq_sum_left
        var il = sq_sum_left / self.weighted_n_left
        var ir = sq_sum_right / self.weighted_n_right
        for k in range(self.n_outputs):
            var ml = self.sum_left[k] / self.weighted_n_left
            var mr = self.sum_right[k] / self.weighted_n_right
            il -= ml * ml
            ir -= mr * mr
        return Impurities(il / Float64(self.n_outputs), ir / Float64(self.n_outputs))

    def node_value(self, dest: Pointer[Float64, MutUntrackedOrigin], dest_len: Int):
        if self.kind == CRITERION_REGRESSION:
            var n = imin(self.n_outputs, dest_len)
            for k in range(n):
                dest[k] = self.sum_total[k] / self.weighted_n_node_samples
        else:
            var n = imin(len(self.sum_total), dest_len)
            for k in range(n):
                dest[k] = self.sum_total[k]

    def is_pure(self) -> Bool:
        if self.kind == CRITERION_REGRESSION:
            return self.node_impurity() == 0.0
        var pure = True
        for k in range(self.n_outputs):
            var base = k * self.sum_stride
            var n_cls = self.n_classes[k]
            var output_pure = False
            for c in range(n_cls):
                if self.sum_total[base + c] == Float64(self.n_node_samples):
                    output_pure = True
                    break
            if not output_pure:
                pure = False
                break
        return pure

    @staticmethod
    def py_init(out self: CoreCriterion, args: PythonObject, kwargs: PythonObject) raises:
        var n_outputs = Int(py=args[0])
        if len(args) >= 2:
            var second = args[1]
            var is_int = True
            try:
                _ = Int(py=second)
            except:
                is_int = False
            if is_int:
                self = make_criterion_regression(n_outputs)
                return
            var classes = List[Int]()
            var n = len(second)
            for i in range(n):
                classes.append(Int(py=second[i]))
            self = make_criterion_classification(n_outputs, classes)
        else:
            self = make_criterion_regression(n_outputs)

    @staticmethod
    def get_kind(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].kind)


# =============================================================================
# Splitter
# =============================================================================


struct SplitRecord(ImplicitlyCopyable):
    var feature: Int
    var pos: Int
    var threshold: Float64
    var improvement: Float64
    var impurity_left: Float64
    var impurity_right: Float64
    var e: Float64

    def __init__(out self):
        self.feature = 0
        self.pos = 0
        self.threshold = 0.0
        self.improvement = 0.0
        self.impurity_left = 0.0
        self.impurity_right = 0.0
        self.e = 0.0


struct CoreSplitter(Defaultable, Movable, Writable):
    var criterion: PythonObject
    var random_state: PythonObject
    var has_data: Bool
    var sd: SharedData
    var x: Pointer[Float32, MutUntrackedOrigin]
    var n_features: Int
    var n_samples: Int
    var weighted_n_samples: Float64
    var start: Int
    var end: Int
    var rand_r_state: UInt32
    var lower_bounds: List[Float32]
    var upper_bounds: List[Float32]

    def __init__(out self):
        self.criterion = Python.none()
        self.random_state = Python.none()
        self.has_data = False
        self.sd = SharedData()
        self.x = alloc[Float32](1)
        self.n_features = 0
        self.n_samples = 0
        self.weighted_n_samples = 0.0
        self.start = 0
        self.end = 0
        self.rand_r_state = 0
        self.lower_bounds = List[Float32]()
        self.upper_bounds = List[Float32]()

    def __deinit__(deinit self):
        self.x.free()
        self.sd.y.free()
        if self.sd.has_sw:
            self.sd.sw.free()

    def init_split(
        mut self,
        x: Pointer[Float32, MutUntrackedOrigin],
        n_samples: Int,
        n_features: Int,
        y: Pointer[Float64, MutUntrackedOrigin],
        y_len: Int,
        y_stride: Int,
        sample_weight: Pointer[Float64, MutUntrackedOrigin],
        has_sw: Bool,
        rand_r_state: UInt32,
    ):
        # Free any previous fit's buffers before replacing them.
        self.sd.y.free()
        if self.sd.has_sw:
            self.sd.sw.free()
        self.x.free()

        self.sd.samples.clear()
        var weighted = 0.0
        for i in range(n_samples):
            var zero_weight = False
            if has_sw:
                zero_weight = sample_weight[i] == 0.0
            if not zero_weight:
                self.sd.samples.append(i)
            if has_sw:
                weighted += sample_weight[i]
            else:
                weighted += 1.0
        self.sd.y = y
        self.sd.y_len = y_len
        self.sd.y_stride = y_stride
        self.sd.sw = sample_weight
        self.sd.has_sw = has_sw
        self.has_data = True
        self.x = x

        self.n_features = n_features
        self.rand_r_state = rand_r_state
        self.n_samples = len(self.sd.samples)
        self.weighted_n_samples = weighted

        self.lower_bounds.clear()
        self.upper_bounds.clear()
        for _ in range(n_features):
            self.lower_bounds.append(0.0)
            self.upper_bounds.append(0.0)

    def node_reset(
        mut self, start: Int, end: Int, crit: Pointer[CoreCriterion, MutAnyOrigin]
    ) -> Float64:
        self.start = start
        self.end = end
        crit[].init(self.sd, self.weighted_n_samples, start, end)
        return crit[].weighted_n_node_samples

    def set_bounds(mut self):
        var n_features = self.n_features
        for f in range(n_features):
            var upper_bound = -INF_F32
            var lower_bound = INF_F32
            for p in range(self.start, self.end):
                var i = self.sd.samples[p]
                var current_f = self.x[i * n_features + f]
                if current_f <= lower_bound:
                    lower_bound = current_f
                if current_f > upper_bound:
                    upper_bound = current_f
            self.upper_bounds[f] = upper_bound
            self.lower_bounds[f] = lower_bound

    def node_split(mut self, crit: Pointer[CoreCriterion, MutAnyOrigin]) -> SplitRecord:
        self.set_bounds()
        var n_features = self.n_features
        var pvals = List[Float32]()
        var rate = Float32(0.0)
        for f in range(n_features):
            var diff = self.upper_bounds[f] - self.lower_bounds[f]
            pvals.append(diff)
            rate += diff
        var e = rand_exponential(rate, self.rand_r_state)
        var feature = rand_multinomial(pvals, self.rand_r_state)
        var threshold = rand_uniform(
            Float64(self.lower_bounds[feature]),
            Float64(self.upper_bounds[feature]),
            self.rand_r_state,
        )

        var partition_end = self.end
        var p = self.start
        while p < partition_end:
            var i = self.sd.samples[p]
            if Float64(self.x[i * n_features + feature]) <= threshold:
                p += 1
            else:
                partition_end -= 1
                var tmp = self.sd.samples[p]
                self.sd.samples[p] = self.sd.samples[partition_end]
                self.sd.samples[partition_end] = tmp

        crit[].reset()
        crit[].update(self.sd, p)
        var imps = crit[].children_impurity(self.sd)

        var rec = SplitRecord()
        rec.feature = feature
        rec.pos = p
        rec.threshold = threshold
        rec.impurity_left = imps.left
        rec.impurity_right = imps.right
        rec.e = e
        return rec

    def node_value_into(
        self,
        crit: Pointer[CoreCriterion, MutAnyOrigin],
        dest: Pointer[Float64, MutUntrackedOrigin],
        dest_len: Int,
    ):
        crit[].node_value(dest, dest_len)

    def node_impurity(self, crit: Pointer[CoreCriterion, MutAnyOrigin]) -> Float64:
        return crit[].node_impurity()

    def is_pure(self, crit: Pointer[CoreCriterion, MutAnyOrigin]) -> Bool:
        return crit[].is_pure()

    @staticmethod
    def py_init(out self: CoreSplitter, args: PythonObject, kwargs: PythonObject) raises:
        self = Self()
        if len(args) >= 1:
            self.criterion = args[0]
        if len(args) >= 2:
            self.random_state = args[1]


# =============================================================================
# Tree
# =============================================================================


struct Node(ImplicitlyCopyable, Writable):
    var left_child: Int
    var right_child: Int
    var feature: Int
    var threshold: Float64
    var impurity: Float64
    var n_node_samples: Int
    var weighted_n_node_samples: Float64
    var tau: Float32
    var variance: Float64

    def write_to(mut self, mut writer: Some[Writer]):
        writer.write(
            "Node(",
            self.left_child,
            ",",
            self.right_child,
            ",",
            self.feature,
            ")",
        )

    def __init__(out self):
        self.left_child = 0
        self.right_child = 0
        self.feature = TREE_UNDEFINED
        self.threshold = 0.0
        self.impurity = 0.0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.tau = 0.0
        self.variance = 0.0


def make_tree(n_features: Int, classes_in: List[Int], n_outputs: Int) -> CoreTree:
    var t = CoreTree()
    t.n_features = n_features
    var mx = 0
    for c in classes_in:
        t.n_classes.append(c)
        if c > mx:
            mx = c
    t.max_n_classes = mx
    t.n_outputs = n_outputs
    t.value_stride = n_outputs * mx
    return t^


struct CoreTree(Defaultable, Movable, Writable):
    var n_features: Int
    var n_classes: List[Int]
    var n_outputs: Int
    var max_n_classes: Int
    var max_depth: Int
    var node_count: Int
    var capacity: Int
    var nodes: List[Node]
    var value: List[Float64]
    var value_stride: Int
    var root: Int
    var lb_flat: List[Float32]
    var ub_flat: List[Float32]
    var lb_scratch: List[Float32]
    var ub_scratch: List[Float32]
    var extent_scratch: List[Float32]

    def __init__(out self):
        self.n_features = 0
        self.n_classes = List[Int]()
        self.n_outputs = 0
        self.max_n_classes = 0
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.nodes = List[Node]()
        self.value = List[Float64]()
        self.value_stride = 0
        self.root = 0
        self.lb_flat = List[Float32]()
        self.ub_flat = List[Float32]()
        self.lb_scratch = List[Float32]()
        self.ub_scratch = List[Float32]()
        self.extent_scratch = List[Float32]()


    def resize_c(mut self, capacity_in: Int) -> Bool:
        var capacity = capacity_in
        if capacity == self.capacity and len(self.nodes) != 0:
            return True
        if capacity < 0:
            if self.capacity == 0:
                capacity = 3
            else:
                capacity = 2 * self.capacity
        if capacity < self.node_count:
            self.node_count = capacity
        var cap = capacity
        while len(self.nodes) < cap:
            self.nodes.append(Node())
        while len(self.nodes) > cap:
            _ = self.nodes.pop()
        var new_value_len = cap * self.value_stride
        while len(self.value) < new_value_len:
            self.value.append(0.0)
        while len(self.value) > new_value_len:
            _ = self.value.pop()
        var nf = self.n_features
        while len(self.lb_flat) < cap * nf:
            self.lb_flat.append(0.0)
        while len(self.lb_flat) > cap * nf:
            _ = self.lb_flat.pop()
        while len(self.ub_flat) < cap * nf:
            self.ub_flat.append(0.0)
        while len(self.ub_flat) > cap * nf:
            _ = self.ub_flat.pop()
        self.capacity = capacity
        return True

    def add_node(
        mut self,
        parent: Int,
        is_left: Bool,
        is_leaf: Bool,
        feature: Int,
        threshold: Float64,
        impurity: Float64,
        n_node_samples: Int,
        weighted_n_node_samples: Float64,
        lower_bounds: List[Float32],
        upper_bounds: List[Float32],
        e: Float64,
    ) -> Int:
        var node_id = self.node_count
        if node_id >= self.capacity and not self.resize_c(-1):
            return TREE_LEAF
        if parent != TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        var tau: Float32
        if parent == TREE_UNDEFINED:
            tau = Float32(e)
        elif is_leaf:
            tau = INF_F32
        else:
            tau = Float32(e + Float64(self.nodes[parent].tau))

        var node = Node()
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.tau = tau
        node.variance = impurity
        if is_leaf:
            node.left_child = TREE_LEAF
            node.right_child = TREE_LEAF
            node.feature = TREE_UNDEFINED
            node.threshold = Float64(TREE_UNDEFINED)
        else:
            node.feature = feature
            node.threshold = threshold
        self.nodes[node_id] = node

        var nf = self.n_features
        for f in range(nf):
            self.lb_flat[node_id * nf + f] = lower_bounds[f]
            self.ub_flat[node_id * nf + f] = upper_bounds[f]

        self.node_count += 1
        return node_id

    def set_node_attributes(
        mut self,
        node_ind: Int,
        left_child: Int,
        right_child: Int,
        feature: Int,
        threshold: Float64,
        tau: Float32,
        n_node_samples: Int,
        weighted_n_node_samples: Float64,
        impurity: Float64,
        variance: Float64,
        x: Pointer[Float32, MutUntrackedOrigin],
        x_start: Int,
        y: Pointer[Float64, MutUntrackedOrigin],
        y_start: Int,
        y_stride: Int,
        child_ind: Int,
    ):
        var nf = self.n_features
        if child_ind == -1:
            for f in range(nf):
                var xv = x[x_start + f]
                self.lb_flat[node_ind * nf + f] = xv
                self.ub_flat[node_ind * nf + f] = xv
        else:
            for f in range(nf):
                var xv = x[x_start + f]
                var lb = f32min(xv, self.lb_flat[child_ind * nf + f])
                var ub = f32max(xv, self.ub_flat[child_ind * nf + f])
                self.lb_flat[node_ind * nf + f] = lb
                self.ub_flat[node_ind * nf + f] = ub
        var node = self.nodes[node_ind]
        node.left_child = left_child
        node.right_child = right_child
        node.feature = feature
        node.threshold = threshold
        node.tau = tau
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.impurity = impurity
        node.variance = variance
        self.nodes[node_ind] = node

        var val_ptr = node_ind * self.value_stride
        if child_ind == -1:
            if self.n_classes[0] == 1:
                self.value[val_ptr] = y[y_start * y_stride]
            else:
                var c = Int(y[y_start * y_stride])
                self.value[val_ptr + c] = 1.0

    def update_node_info(
        mut self,
        parent_id: Int,
        child_id: Int,
        y: Pointer[Float64, MutUntrackedOrigin],
        y_start: Int,
        y_stride: Int,
    ):
        var is_regression = self.n_classes[0] == 1
        var child_ptr = child_id * self.value_stride
        var parent_ptr = parent_id * self.value_stride
        var child_n = self.nodes[child_id].n_node_samples
        var y_val = y[y_start * y_stride]
        if is_regression:
            var old_mean = self.value[child_ptr]
            var new_sum = old_mean * Float64(child_n) + y_val
            var new_mean = new_sum / Float64(child_n + 1)
            self.value[parent_ptr] = new_mean
            var ss = (self.nodes[child_id].variance + old_mean * old_mean) * Float64(child_n)
            var parent = self.nodes[parent_id]
            parent.variance = (ss + y_val * y_val) / Float64(child_n + 1) - new_mean * new_mean
            self.nodes[parent_id] = parent
        else:
            self.value[parent_ptr + Int(y_val)] += 1.0
            if child_id != parent_id:
                var n_cls = self.n_classes[0]
                for c in range(n_cls):
                    self.value[parent_ptr + c] += self.value[child_ptr + c]

    def init_root(
        mut self,
        x: Pointer[Float32, MutUntrackedOrigin],
        y: Pointer[Float64, MutUntrackedOrigin],
        y_stride: Int,
    ):
        self.set_node_attributes(
            0,
            TREE_LEAF,
            TREE_LEAF,
            TREE_UNDEFINED,
            Float64(TREE_UNDEFINED),
            INF_F32,
            1,
            1.0,
            0.0,
            0.0,
            x,
            0,
            y,
            0,
            y_stride,
            -1,
        )
        self.node_count += 1

    def extend(
        mut self,
        x: Pointer[Float32, MutUntrackedOrigin],
        x_start: Int,
        y: Pointer[Float64, MutUntrackedOrigin],
        y_start: Int,
        y_stride: Int,
        random_state: UInt32,
        min_samples_split: Int,
    ) raises:
        var rand_r_state = random_state
        var curr_id = self.root
        var parent_id: Int = -1
        var tau_parent = Float32(0.0)
        var nf = self.n_features

        while True:
            ref curr_lb = self.lb_scratch
            ref curr_ub = self.ub_scratch
            ref extent = self.extent_scratch
            curr_lb.clear()
            curr_ub.clear()
            extent.clear()
            for f in range(nf):
                curr_lb.append(self.lb_flat[curr_id * nf + f])
                curr_ub.append(self.ub_flat[curr_id * nf + f])
            var curr_tau = self.nodes[curr_id].tau
            var curr_n = self.nodes[curr_id].n_node_samples
            var curr_wnn = self.nodes[curr_id].weighted_n_node_samples

            var new_rate = Float32(0.0)
            for f in range(nf):
                var x_val = x[x_start + f]
                var e_l = f32max(curr_lb[f] - x_val, Float32(0.0))
                var e_u = f32max(x_val - curr_ub[f], Float32(0.0))
                extent.append(e_l + e_u)
                new_rate += extent[f]
            var e = Float32(rand_exponential(new_rate, rand_r_state))

            if (Float64(tau_parent) + Float64(e)) < Float64(curr_tau) and (curr_n + 1) >= min_samples_split:
                var new_child_id = self.node_count
                var new_parent_id = self.node_count + 1

                var delta = rand_multinomial(extent, rand_r_state)
                var x_val = x[x_start + delta]
                var l_b = curr_lb[delta]
                var u_b = curr_ub[delta]
                var xi: Float32
                if x_val > u_b:
                    xi = Float32(rand_uniform(Float64(u_b), Float64(x_val), rand_r_state))
                else:
                    xi = Float32(rand_uniform(Float64(x_val), Float64(l_b), rand_r_state))

                var left_child: Int
                var right_child: Int
                if x_val < xi:
                    left_child = new_child_id
                    right_child = curr_id
                else:
                    left_child = curr_id
                    right_child = new_child_id

                if not self.resize_c(self.node_count + 2):
                    raise Error("resizing tree failed")

                self.set_node_attributes(
                    new_child_id,
                    TREE_LEAF,
                    TREE_LEAF,
                    TREE_UNDEFINED,
                    Float64(TREE_UNDEFINED),
                    INF_F32,
                    1,
                    1.0,
                    0.0,
                    0.0,
                    x,
                    x_start,
                    y,
                    y_start,
                    y_stride,
                    -1,
                )

                self.set_node_attributes(
                    new_parent_id,
                    left_child,
                    right_child,
                    delta,
                    Float64(xi),
                    Float32(Float64(tau_parent) + Float64(e)),
                    curr_n + 1,
                    curr_wnn + 1.0,
                    0.0,
                    0.0,
                    x,
                    x_start,
                    y,
                    y_start,
                    y_stride,
                    curr_id,
                )
                self.update_node_info(new_parent_id, curr_id, y, y_start, y_stride)

                if curr_id == self.root:
                    self.root = new_parent_id
                else:
                    if self.nodes[parent_id].left_child == curr_id:
                        self.nodes[parent_id].left_child = new_parent_id
                    else:
                        self.nodes[parent_id].right_child = new_parent_id
                self.max_depth += 1
                self.node_count += 2
                break
            else:
                for f in range(nf):
                    var x_val2 = x[x_start + f]
                    var lb = f32min(x_val2, self.lb_flat[curr_id * nf + f])
                    var ub = f32max(x_val2, self.ub_flat[curr_id * nf + f])
                    self.lb_flat[curr_id * nf + f] = lb
                    self.ub_flat[curr_id * nf + f] = ub
                self.update_node_info(curr_id, curr_id, y, y_start, y_stride)
                var grown = self.nodes[curr_id]
                grown.n_node_samples += 1
                grown.weighted_n_node_samples += 1.0
                self.nodes[curr_id] = grown
                if grown.left_child == TREE_LEAF:
                    break
                parent_id = curr_id
                if Float64(x[x_start + grown.feature]) < grown.threshold:
                    curr_id = grown.left_child
                else:
                    curr_id = grown.right_child
                tau_parent = self.nodes[parent_id].tau

    # ---- Python-facing getters/methods ----

    @staticmethod
    def py_init(out self: CoreTree, args: PythonObject, kwargs: PythonObject) raises:
        var n_features = Int(py=args[0])
        var classes = List[Int]()
        var second = args[1]
        var n = len(second)
        for i in range(n):
            classes.append(Int(py=second[i]))
        var n_outputs = Int(py=args[2])
        self = make_tree(n_features, classes, n_outputs)

    @staticmethod
    def g_node_count(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].node_count)

    @staticmethod
    def g_capacity(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].capacity)

    @staticmethod
    def g_max_depth(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].max_depth)

    @staticmethod
    def g_n_features(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].n_features)

    @staticmethod
    def g_n_outputs(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].n_outputs)

    @staticmethod
    def g_max_n_classes(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].max_n_classes)

    @staticmethod
    def g_root(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].root)

    @staticmethod
    def g_n_classes(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        return np_array_int_list(np, self_ptr[].n_classes, "intp")

    @staticmethod
    def g_children_left(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "intp")
        var out = ptr_int(arr)
        for i in range(n):
            out[i] = t.nodes[i].left_child
        return arr

    @staticmethod
    def g_children_right(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "intp")
        var out = ptr_int(arr)
        for i in range(n):
            out[i] = t.nodes[i].right_child
        return arr

    @staticmethod
    def g_feature(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "intp")
        var out = ptr_int(arr)
        for i in range(n):
            out[i] = t.nodes[i].feature
        return arr

    @staticmethod
    def g_threshold(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "float64")
        var out = ptr_f64(arr)
        for i in range(n):
            out[i] = t.nodes[i].threshold
        return arr

    @staticmethod
    def g_impurity(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "float64")
        var out = ptr_f64(arr)
        for i in range(n):
            out[i] = t.nodes[i].impurity
        return arr

    @staticmethod
    def g_n_node_samples(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "intp")
        var out = ptr_int(arr)
        for i in range(n):
            out[i] = t.nodes[i].n_node_samples
        return arr

    @staticmethod
    def g_weighted_n_node_samples(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "float64")
        var out = ptr_f64(arr)
        for i in range(n):
            out[i] = t.nodes[i].weighted_n_node_samples
        return arr

    @staticmethod
    def g_tau(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "float32")
        var out = ptr_f32(arr)
        for i in range(n):
            out[i] = t.nodes[i].tau
        return arr

    @staticmethod
    def g_lower_bounds(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var nf = t.n_features
        var arr = np_empty_2d(np, n, nf, "float32")
        var out = ptr_f32(arr)
        for i in range(n):
            for f in range(nf):
                out[i * nf + f] = t.lb_flat[i * nf + f]
        return arr

    @staticmethod
    def g_upper_bounds(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var nf = t.n_features
        var arr = np_empty_2d(np, n, nf, "float32")
        var out = ptr_f32(arr)
        for i in range(n):
            for f in range(nf):
                out[i * nf + f] = t.ub_flat[i * nf + f]
        return arr

    @staticmethod
    def g_variance(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var arr = np_empty_1d(np, n, "float64")
        var out = ptr_f64(arr)
        for i in range(n):
            out[i] = t.nodes[i].variance
        return arr

    @staticmethod
    def g_mean(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var total = t.node_count * t.value_stride
        var arr = np_empty_1d(np, total, "float64")
        var out = ptr_f64(arr)
        for i in range(total):
            out[i] = t.value[i]
        return arr

    @staticmethod
    def g_base_value(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var root = t.root
        var arr = np_empty_1d(np, t.value_stride, "float64")
        var out = ptr_f64(arr)
        for k in range(t.value_stride):
            out[k] = t.value[root * t.value_stride + k]
        return arr

    @staticmethod
    def g_value(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var no = t.n_outputs
        var m = t.max_n_classes
        var arr = np_empty_3d(np, n, no, m, "float64")
        var out = ptr_f64(arr)
        for i in range(n):
            for k in range(no):
                for c in range(m):
                    out[(i * no + k) * m + c] = t.value[i * t.value_stride + k * m + c]
        return arr

    @staticmethod
    def apply(self_ptr: Pointer[Self, MutAnyOrigin], x_obj: PythonObject) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var shape = iface_shape(x_obj)
        var n_samples = shape[0]
        var xp = ptr_f32(x_obj)
        var arr = np_empty_1d(np, n_samples, "intp")
        var out = ptr_int(arr)
        for i in range(n_samples):
            var curr = t.root
            while t.nodes[curr].left_child != TREE_LEAF:
                if Float64(xp[i * t.n_features + t.nodes[curr].feature]) <= t.nodes[curr].threshold:
                    curr = t.nodes[curr].left_child
                else:
                    curr = t.nodes[curr].right_child
            out[i] = curr
        return arr

    @staticmethod
    def predict(
        self_ptr: Pointer[Self, MutAnyOrigin],
        x_obj: PythonObject,
        return_std_obj: PythonObject,
        is_regression_obj: PythonObject,
        path_smoothing_obj: PythonObject,
    ) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var shape = iface_shape(x_obj)
        var n_samples = shape[0]
        var n_features = shape[1]
        var xp = ptr_f32(x_obj)
        var return_std = Bool(py=return_std_obj)
        var is_regression = Bool(py=is_regression_obj)
        var path_smoothing = Bool(py=path_smoothing_obj)

        var n_classes = t.max_n_classes
        var value_stride = t.value_stride

        var mean_arr = np_empty_1d(np, n_samples, "float32")
        var mean_p = ptr_f32(mean_arr)
        var std_arr = np_empty_1d(np, n_samples, "float32")
        var std_p = ptr_f32(std_arr)
        var proba_arr = np_empty_2d(np, n_samples, n_classes, "float32")
        var proba_p = ptr_f32(proba_arr)
        for i in range(n_samples):
            mean_p[i] = 0.0
            std_p[i] = 0.0
            for c in range(n_classes):
                proba_p[i * n_classes + c] = 0.0

        for i in range(n_samples):
            var parent_tau = 0.0
            var p_nsy = 1.0
            var node_id = t.root
            while True:
                var node = t.nodes[node_id]
                var delta = Float64(node.tau) - parent_tau
                parent_tau = Float64(node.tau)

                var eta = 0.0
                if path_smoothing:
                    for f in range(n_features):
                        var x_val = Float64(xp[i * n_features + f])
                        var d1 = x_val - Float64(t.ub_flat[node_id * t.n_features + f])
                        var d2 = Float64(t.lb_flat[node_id * t.n_features + f]) - x_val
                        eta += fmax(d1, 0.0) + fmax(d2, 0.0)

                # With path smoothing every visited node contributes
                # w_j = p_nsy * p_js; in constant mode only the leaf
                # contributes (p_nsy stays 1.0 because p_js is 0).
                var w_j: Float64
                var p_js: Float64
                if node.left_child == TREE_LEAF:
                    w_j = p_nsy
                    p_js = 0.0
                elif path_smoothing:
                    p_js = 1.0 - exp(-delta * eta)
                    w_j = p_nsy * p_js
                else:
                    w_j = 0.0
                    p_js = 0.0

                if is_regression:
                    mean_p[i] = Float32(
                        Float64(mean_p[i]) + w_j * t.value[node_id * value_stride]
                    )
                else:
                    for c in range(n_classes):
                        var val = w_j * (
                            t.value[node_id * value_stride + c]
                            / Float64(t.nodes[node_id].n_node_samples)
                        )
                        proba_p[i * n_classes + c] = Float32(
                            Float64(proba_p[i * n_classes + c]) + val
                        )
                if return_std:
                    var v0 = t.value[node_id * value_stride]
                    var val = w_j * (v0 * v0 + node.variance)
                    std_p[i] = Float32(Float64(std_p[i]) + val)

                if node.left_child == TREE_LEAF:
                    break
                p_nsy *= 1.0 - p_js

                if Float64(xp[i * n_features + node.feature]) <= node.threshold:
                    node_id = node.left_child
                else:
                    node_id = node.right_child
            if return_std:
                var m = Float64(mean_p[i])
                var s = fmax(Float64(std_p[i]) - m * m, 0.0)
                std_p[i] = Float32(sqrt(s))

        if is_regression:
            if return_std:
                return Python.tuple(mean_arr, std_arr)
            return Python.tuple(mean_arr)
        return Python.tuple(proba_arr)

    @staticmethod
    def decision_path(self_ptr: Pointer[Self, MutAnyOrigin], x_obj: PythonObject) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var shape = iface_shape(x_obj)
        var n_samples = shape[0]
        var xp = ptr_f32(x_obj)
        var node_count = t.node_count

        var indptr = List[Int]()
        var indices = List[Int]()
        indptr.append(0)
        for i in range(n_samples):
            var curr = t.root
            while curr != TREE_LEAF:
                indices.append(curr)
                if t.nodes[curr].left_child == TREE_LEAF:
                    break
                if Float64(xp[i * t.n_features + t.nodes[curr].feature]) <= t.nodes[curr].threshold:
                    curr = t.nodes[curr].left_child
                else:
                    curr = t.nodes[curr].right_child
            indptr.append(len(indices))

        var data_arr = np_empty_1d(np, len(indices), "intp")
        var data_p = ptr_int(data_arr)
        var indices_arr = np_empty_1d(np, len(indices), "intp")
        var indices_p = ptr_int(indices_arr)
        var indptr_arr = np_empty_1d(np, len(indptr), "intp")
        var indptr_p = ptr_int(indptr_arr)
        for j in range(len(indices)):
            data_p[j] = 1
            indices_p[j] = indices[j]
        for j in range(len(indptr)):
            indptr_p[j] = indptr[j]

        var scipy = Python.import_module("scipy.sparse")
        var tup = Python.tuple(data_arr, indices_arr, indptr_arr)
        var shp = Python.tuple(Int(n_samples), Int(node_count))
        return scipy.csr_matrix(tup, shape=shp)

    @staticmethod
    def isolation_path_length(self_ptr: Pointer[Self, MutAnyOrigin], x_obj: PythonObject) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var shape = iface_shape(x_obj)
        var n_samples = shape[0]
        var xp = ptr_f32(x_obj)
        var arr = np_empty_1d(np, n_samples, "float64")
        var out = ptr_f64(arr)
        for i in range(n_samples):
            var curr = t.root
            var depth = 0
            while t.nodes[curr].left_child != TREE_LEAF:
                depth += 1
                if Float64(xp[i * t.n_features + t.nodes[curr].feature]) <= t.nodes[curr].threshold:
                    curr = t.nodes[curr].left_child
                else:
                    curr = t.nodes[curr].right_child
            out[i] = Float64(depth)
        return arr

    @staticmethod
    def weighted_decision_path(self_ptr: Pointer[Self, MutAnyOrigin], x_obj: PythonObject) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var shape = iface_shape(x_obj)
        var n_samples = shape[0]
        var n_features = shape[1]
        var xp = ptr_f32(x_obj)
        var node_count = t.node_count

        var indptr = List[Int]()
        var indices = List[Int]()
        var values = List[Float32]()
        indptr.append(0)
        for i in range(n_samples):
            var p_nsy = Float32(1.0)
            var parent_tau = Float32(0.0)
            var curr = t.root
            while True:
                var node = t.nodes[curr]
                if node.left_child != TREE_LEAF:
                    var delta = node.tau - parent_tau
                    parent_tau = node.tau
                    var eta = Float32(0.0)
                    for f in range(n_features):
                        var x_val = xp[i * n_features + f]
                        var d1 = x_val - t.ub_flat[curr * t.n_features + f]
                        var d2 = t.lb_flat[curr * t.n_features + f] - x_val
                        eta += f32max(d1, Float32(0.0)) + f32max(d2, Float32(0.0))
                    var p_s = Float32(1.0) - Float32(exp(-Float64(delta * eta)))
                    if p_s > 0.0:
                        indices.append(curr)
                        values.append(p_s * p_nsy)
                    p_nsy *= Float32(1.0) - p_s
                    if Float64(xp[i * n_features + node.feature]) <= node.threshold:
                        curr = node.left_child
                    else:
                        curr = node.right_child
                else:
                    indices.append(curr)
                    values.append(p_nsy)
                    break
            indptr.append(len(indices))

        var values_arr = np_empty_1d(np, len(values), "float64")
        var values_p = ptr_f64(values_arr)
        var indices_arr = np_empty_1d(np, len(indices), "intp")
        var indices_p = ptr_int(indices_arr)
        var indptr_arr = np_empty_1d(np, len(indptr), "intp")
        var indptr_p = ptr_int(indptr_arr)
        for j in range(len(values)):
            values_p[j] = Float64(values[j])
        for j in range(len(indices)):
            indices_p[j] = indices[j]
        for j in range(len(indptr)):
            indptr_p[j] = indptr[j]

        var scipy = Python.import_module("scipy.sparse")
        var tup = Python.tuple(values_arr, indices_arr, indptr_arr)
        var shp = Python.tuple(Int(n_samples), Int(node_count))
        return scipy.csr_matrix(tup, shape=shp)

    @staticmethod
    def shap_values(self_ptr: Pointer[Self, MutAnyOrigin], x_obj: PythonObject) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var shape = iface_shape(x_obj)
        var n_samples = shape[0]
        var n_features = shape[1]
        var xp = ptr_f32(x_obj)
        var is_regression = t.n_classes[0] == 1
        var max_n_classes = t.max_n_classes

        var out_arr = np_empty_2d(np, n_samples, n_features, "float64")
        var out = ptr_f64(out_arr)
        for i in range(n_samples * n_features):
            out[i] = 0.0

        for i in range(n_samples):
            var path = List[Int]()
            var curr = t.root
            while True:
                path.append(curr)
                if t.nodes[curr].left_child == TREE_LEAF:
                    break
                if Float64(xp[i * n_features + t.nodes[curr].feature]) <= t.nodes[curr].threshold:
                    curr = t.nodes[curr].left_child
                else:
                    curr = t.nodes[curr].right_child

            for pi in range(len(path)):
                var node_idx = path[pi]
                var node = t.nodes[node_idx]
                if node.left_child != TREE_LEAF:
                    var feat = node.feature
                    if feat >= n_features:
                        continue
                    var x_val = Float64(xp[i * n_features + feat])
                    var threshold = node.threshold

                    var val_parent = node_pred(self_ptr, node_idx, is_regression, max_n_classes)
                    var val_left = node_pred(self_ptr, node.left_child, is_regression, max_n_classes)
                    var val_right = node_pred(self_ptr, node.right_child, is_regression, max_n_classes)

                    var shap_val: Float64
                    if x_val <= threshold:
                        shap_val = val_left - val_parent
                    else:
                        shap_val = val_right - val_parent
                    out[i * n_features + feat] = shap_val
        return out_arr

    @staticmethod
    def populate_from_arrays(
        self_ptr: Pointer[Self, MutAnyOrigin], parts: PythonObject
    ) raises -> PythonObject:
        ref t = self_ptr[]
        var left_child = parts[0]
        var right_child = parts[1]
        var feature = parts[2]
        var threshold = parts[3]
        var n_node_samples = parts[4]
        var value = parts[5]
        var tau = parts[6]
        var lower_bounds = parts[7]
        var upper_bounds = parts[8]

        var n_nodes = Int(py=len(left_child))
        if not t.resize_c(n_nodes):
            raise Error("failed to resize tree")

        var lc = ptr_f64(left_child)
        var rc = ptr_f64(right_child)
        var feat = ptr_f64(feature)
        var thresh = ptr_f64(threshold)
        var samples = ptr_f64(n_node_samples)
        var val = ptr_f64(value)
        var val_len = Int(py=len(value))
        var tau_arr = ptr_f32(tau)
        var lb_arr = ptr_f32(lower_bounds)
        var ub_arr = ptr_f32(upper_bounds)
        var lb_shape = iface_shape(lower_bounds)
        var lb_rows = lb_shape[0]
        var lb_cols = lb_shape[1]
        var nf = t.n_features

        t.root = 0
        t.node_count = n_nodes

        for i in range(n_nodes):
            var node = Node()
            node.left_child = Int(lc[i])
            node.right_child = Int(rc[i])
            node.feature = Int(feat[i])
            node.threshold = thresh[i]
            node.tau = tau_arr[i]
            node.n_node_samples = Int(samples[i])
            node.weighted_n_node_samples = samples[i]
            node.impurity = 0.0
            node.variance = 0.0
            t.nodes[i] = node

            for f in range(nf):
                if i < lb_rows and f < lb_cols:
                    t.lb_flat[i * nf + f] = lb_arr[i * lb_cols + f]
                    t.ub_flat[i * nf + f] = ub_arr[i * lb_cols + f]
                else:
                    t.lb_flat[i * nf + f] = 0.0
                    t.ub_flat[i * nf + f] = 0.0

            var val_offset = i * t.value_stride
            var val_end = val_offset + t.value_stride
            if val_end <= val_len:
                for k in range(val_offset, val_end):
                    t.value[k] = val[k]
        return Python.none()

    @staticmethod
    def getstate(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var np = numpy_mod()
        ref t = self_ptr[]
        var n = t.node_count
        var nf = t.n_features
        var state = Python.dict()

        var lc = List[Int]()
        var rc = List[Int]()
        var ft = List[Int]()
        var ns = List[Int]()
        var th = List[Float64]()
        var im = List[Float64]()
        var wn = List[Float64]()
        var ta = List[Float32]()
        var vr = List[Float64]()
        for i in range(n):
            lc.append(t.nodes[i].left_child)
            rc.append(t.nodes[i].right_child)
            ft.append(t.nodes[i].feature)
            ns.append(t.nodes[i].n_node_samples)
            th.append(t.nodes[i].threshold)
            im.append(t.nodes[i].impurity)
            wn.append(t.nodes[i].weighted_n_node_samples)
            ta.append(t.nodes[i].tau)
            vr.append(t.nodes[i].variance)

        state["max_depth"] = PythonObject(t.max_depth)
        state["node_count"] = PythonObject(t.node_count)
        state["root"] = PythonObject(t.root)
        state["left_child"] = mk_f64_array_int(np, lc)
        state["right_child"] = mk_f64_array_int(np, rc)
        state["feature"] = mk_f64_array_int(np, ft)
        state["n_node_samples"] = mk_f64_array_int(np, ns)

        state["threshold"] = mk_f64_array(np, th)
        state["impurity"] = mk_f64_array(np, im)
        state["weighted_n_node_samples"] = mk_f64_array(np, wn)
        state["variance"] = mk_f64_array(np, vr)

        var ta_arr = np_empty_1d(np, len(ta), "float32")
        var ta_ptr = ptr_f32(ta_arr)
        for i in range(len(ta)):
            ta_ptr[i] = ta[i]
        state["tau"] = ta_arr

        var lb_arr = np_empty_2d(np, n, nf, "float32")
        var lb_ptr = ptr_f32(lb_arr)
        var ub_arr = np_empty_2d(np, n, nf, "float32")
        var ub_ptr = ptr_f32(ub_arr)
        for i in range(n):
            for f in range(nf):
                lb_ptr[i * nf + f] = t.lb_flat[i * nf + f]
                ub_ptr[i * nf + f] = t.ub_flat[i * nf + f]
        state["lower_bounds"] = lb_arr
        state["upper_bounds"] = ub_arr

        var total = n * t.value_stride
        var val_arr = np_empty_1d(np, total, "float64")
        var val_ptr = ptr_f64(val_arr)
        for i in range(total):
            val_ptr[i] = t.value[i]
        state["values"] = val_arr
        return state

    @staticmethod
    def setstate(self_ptr: Pointer[Self, MutAnyOrigin], d: PythonObject) raises -> PythonObject:
        ref t = self_ptr[]
        var max_depth = Int(py=d["max_depth"])
        var node_count = Int(py=d["node_count"])
        var root = Int(py=d["root"])

        var lc = ptr_f64(d["left_child"])
        var rc = ptr_f64(d["right_child"])
        var ft = ptr_f64(d["feature"])
        var th = ptr_f64(d["threshold"])
        var im = ptr_f64(d["impurity"])
        var ns = ptr_f64(d["n_node_samples"])
        var wn = ptr_f64(d["weighted_n_node_samples"])
        var ta = ptr_f32(d["tau"])
        var vr = ptr_f64(d["variance"])
        var lb = ptr_f32(d["lower_bounds"])
        var ub = ptr_f32(d["upper_bounds"])
        var values = ptr_f64(d["values"])

        var nf = t.n_features

        t.max_depth = max_depth
        t.node_count = node_count
        t.root = root
        t.capacity = node_count
        t.nodes.clear()
        t.lb_flat.clear()
        t.ub_flat.clear()
        for i in range(node_count):
            var node = Node()
            node.left_child = Int(lc[i])
            node.right_child = Int(rc[i])
            node.feature = Int(ft[i])
            node.threshold = th[i]
            node.impurity = im[i]
            node.n_node_samples = Int(ns[i])
            node.weighted_n_node_samples = wn[i]
            node.tau = ta[i]
            node.variance = vr[i]
            t.nodes.append(node)
            for f in range(nf):
                t.lb_flat.append(lb[i * nf + f])
                t.ub_flat.append(ub[i * nf + f])

        t.value.clear()
        var total = node_count * t.value_stride
        for i in range(total):
            t.value.append(values[i])
        return Python.none()


def node_pred(tptr: Pointer[CoreTree, MutAnyOrigin], idx: Int, is_regression: Bool, max_n_classes: Int) -> Float64:
    if is_regression:
        return tptr[].value[idx * tptr[].value_stride]
    var total = tptr[].nodes[idx].n_node_samples
    if total > 0:
        var s = 0.0
        for c in range(max_n_classes):
            s += tptr[].value[idx * tptr[].value_stride + c]
        return s / Float64(total)
    return 0.0


# =============================================================================
# Builders
# =============================================================================


struct StackRecord(ImplicitlyCopyable):
    var start: Int
    var end: Int
    var depth: Int
    var parent: Int
    var is_left: Bool
    var impurity: Float64
    var n_constant_features: Int

    def __init__(out self):
        self.start = 0
        self.end = 0
        self.depth = 0
        self.parent = 0
        self.is_left = False
        self.impurity = 0.0
        self.n_constant_features = 0


struct CoreDepthFirstBuilder(Defaultable, Movable, Writable):
    var splitter: PythonObject
    var min_samples_split: Int
    var max_depth: Int

    def __init__(out self):
        self.splitter = Python.none()
        self.min_samples_split = 0
        self.max_depth = 0

    @staticmethod
    def py_init(out self: CoreDepthFirstBuilder, args: PythonObject, kwargs: PythonObject) raises:
        self = Self()
        self.splitter = args[0]
        self.min_samples_split = Int(py=args[1])
        self.max_depth = Int(py=args[2])

    @staticmethod
    def build(
        self_ptr: Pointer[Self, MutAnyOrigin],
        tree_obj: PythonObject,
        x_obj: PythonObject,
        y_obj: PythonObject,
        sample_weight_obj: PythonObject,
        x_idx_sorted_obj: PythonObject,
    ) raises -> PythonObject:
        _ = x_idx_sorted_obj
        var tptr = tree_obj.downcast_value_ptr[CoreTree]()
        var sptr = self_ptr[].splitter.downcast_value_ptr[CoreSplitter]()

        var x_shape = iface_shape(x_obj)
        var n_samples = x_shape[0]
        var n_features = x_shape[1]
        var y_shape = iface_shape(y_obj)
        var y_n = y_shape[0]
        var y_stride = y_shape[1]
        if y_n != n_samples:
            raise Error(
                "Number of labels="
                + String(y_n)
                + " does not match number of samples="
                + String(n_samples)
            )

        var xp = ptr_f32(x_obj)
        var x_copy = alloc_f32(n_samples * n_features)
        for i in range(n_samples * n_features):
            x_copy[i] = xp[i]
        var yp = ptr_f64(y_obj)
        var y_copy = alloc_f64(y_n * y_stride)
        for i in range(y_n * y_stride):
            y_copy[i] = yp[i]

        var sw = alloc[Float64](1)
        var has_sw = False
        try:
            var sw_len = Int(py=len(sample_weight_obj))
            if sw_len > 0:
                sw.free()
                sw = alloc[Float64](sw_len)
                var swp = ptr_f64(sample_weight_obj)
                for i in range(sw_len):
                    sw[i] = swp[i]
                has_sw = True
        except e:
            _ = e

        var seed_obj = sptr[].random_state.randint(0, 2147483647)
        var rand_seed = UInt32(Int(py=seed_obj))

        sptr[].init_split(
            x_copy,
            n_samples,
            n_features,
            y_copy,
            y_n * y_stride,
            y_stride,
            sw,
            has_sw,
            rand_seed,
        )

        var init_capacity: Int
        if tptr[].max_depth <= 10:
            init_capacity = (1 << (tptr[].max_depth + 1)) - 1
        else:
            init_capacity = 2047
        tptr[].resize_c(init_capacity)

        var crit_ptr = sptr[].criterion.downcast_value_ptr[CoreCriterion]()

        var stack = List[StackRecord]()
        var first_rec = StackRecord()
        first_rec.start = 0
        first_rec.end = sptr[].n_samples
        first_rec.depth = 0
        first_rec.parent = TREE_UNDEFINED
        first_rec.is_left = False
        first_rec.impurity = INF_F64
        first_rec.n_constant_features = 0
        stack.append(first_rec)

        var first = True
        var max_depth_seen: Int = -1

        while len(stack) > 0:
            var rec = stack.pop()
            var start = rec.start
            var end = rec.end
            var depth = rec.depth
            var parent = rec.parent
            var is_left = rec.is_left
            var impurity = rec.impurity
            var n_constant_features = rec.n_constant_features
            var n_node_samples = end - start

            var weighted_n_node_samples = sptr[].node_reset(start, end, crit_ptr)

            if first:
                impurity = sptr[].node_impurity(crit_ptr)
                first = False

            var is_leaf = depth >= self_ptr[].max_depth or n_node_samples < self_ptr[].min_samples_split
            var split = SplitRecord()
            if not is_leaf:
                split = sptr[].node_split(crit_ptr)
                is_leaf = split.pos >= end
            else:
                sptr[].set_bounds()
            is_leaf = is_leaf or sptr[].is_pure(crit_ptr)

            var node_id = tptr[].add_node(
                parent,
                is_left,
                is_leaf,
                split.feature,
                split.threshold,
                impurity,
                n_node_samples,
                weighted_n_node_samples,
                sptr[].lower_bounds,
                sptr[].upper_bounds,
                split.e,
            )
            if node_id == TREE_LEAF:
                raise Error("failed to allocate node")

            var offset = node_id * tptr[].value_stride
            var dest = alloc_f64(tptr[].value_stride)
            sptr[].node_value_into(crit_ptr, dest, tptr[].value_stride)
            for k in range(tptr[].value_stride):
                tptr[].value[offset + k] = dest[k]

            if not is_leaf:
                var right_rec = StackRecord()
                right_rec.start = split.pos
                right_rec.end = end
                right_rec.depth = depth + 1
                right_rec.parent = node_id
                right_rec.is_left = False
                right_rec.impurity = split.impurity_right
                right_rec.n_constant_features = n_constant_features
                stack.append(right_rec)
                var left_rec = StackRecord()
                left_rec.start = start
                left_rec.end = split.pos
                left_rec.depth = depth + 1
                left_rec.parent = node_id
                left_rec.is_left = True
                left_rec.impurity = split.impurity_left
                left_rec.n_constant_features = n_constant_features
                stack.append(left_rec)

            if depth > max_depth_seen:
                max_depth_seen = depth

        var node_count = tptr[].node_count
        tptr[].resize_c(node_count)
        tptr[].max_depth = max_depth_seen
        return Python.none()


struct CorePartialFitBuilder(Defaultable, Movable, Writable):
    var min_samples_split: Int
    var max_depth: Int
    var random_state: PythonObject

    def __init__(out self):
        self.min_samples_split = 0
        self.max_depth = 0
        self.random_state = Python.none()

    @staticmethod
    def py_init(out self: CorePartialFitBuilder, args: PythonObject, kwargs: PythonObject) raises:
        self = Self()
        self.min_samples_split = Int(py=args[0])
        self.max_depth = Int(py=args[1])
        self.random_state = args[2]

    @staticmethod
    def build(
        self_ptr: Pointer[Self, MutAnyOrigin],
        tree_obj: PythonObject,
        x_obj: PythonObject,
        y_obj: PythonObject,
    ) raises -> PythonObject:
        var tptr = tree_obj.downcast_value_ptr[CoreTree]()

        var seed_obj = self_ptr[].random_state.randint(0, 2147483647)
        var rand_r_state = UInt32(Int(py=seed_obj))
        var x_shape = iface_shape(x_obj)
        var n_samples = x_shape[0]
        var n_features = x_shape[1]
        var y_shape = iface_shape(y_obj)
        var y_stride = y_shape[1]

        var xp = ptr_f32(x_obj)
        var x_copy = alloc_f32(n_samples * n_features)
        for i in range(n_samples * n_features):
            x_copy[i] = xp[i]
        var yp = ptr_f64(y_obj)
        var y_copy = alloc_f64(n_samples * y_stride)
        for i in range(n_samples * y_stride):
            y_copy[i] = yp[i]

        if tptr[].max_depth <= 10:
            tptr[].resize_c((1 << (tptr[].max_depth + 1)) - 1)
        var node_count = tptr[].node_count
        if node_count == 0:
            tptr[].init_root(x_copy, y_copy, y_stride)
        var start = 0
        if node_count == 0:
            start = 1
        for sample_ind in range(start, n_samples):
            tptr[].extend(
                x_copy,
                sample_ind * n_features,
                y_copy,
                sample_ind * y_stride,
                y_stride,
                rand_r_state,
                self_ptr[].min_samples_split,
            )
        return Python.none()


# =============================================================================
# Module
# =============================================================================


@export
def PyInit__native_mojo_core() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native_mojo_core")
        _ = (
            m.add_type[CoreCriterion]("CoreCriterion")
            .def_py_init[CoreCriterion.py_init]()
            .def_method[CoreCriterion.get_kind]("kind")
        )
        _ = m.add_type[CoreSplitter]("CoreSplitter").def_py_init[CoreSplitter.py_init]()
        _ = (
            m.add_type[CoreTree]("CoreTree")
            .def_py_init[CoreTree.py_init]()
            .def_method[CoreTree.g_node_count]("node_count")
            .def_method[CoreTree.g_capacity]("capacity")
            .def_method[CoreTree.g_max_depth]("max_depth")
            .def_method[CoreTree.g_n_features]("n_features")
            .def_method[CoreTree.g_n_outputs]("n_outputs")
            .def_method[CoreTree.g_max_n_classes]("max_n_classes")
            .def_method[CoreTree.g_root]("root")
            .def_method[CoreTree.g_n_classes]("n_classes")
            .def_method[CoreTree.g_children_left]("children_left")
            .def_method[CoreTree.g_children_right]("children_right")
            .def_method[CoreTree.g_feature]("feature")
            .def_method[CoreTree.g_threshold]("threshold")
            .def_method[CoreTree.g_impurity]("impurity")
            .def_method[CoreTree.g_n_node_samples]("n_node_samples")
            .def_method[CoreTree.g_weighted_n_node_samples]("weighted_n_node_samples")
            .def_method[CoreTree.g_tau]("tau")
            .def_method[CoreTree.g_lower_bounds]("lower_bounds")
            .def_method[CoreTree.g_upper_bounds]("upper_bounds")
            .def_method[CoreTree.g_variance]("variance")
            .def_method[CoreTree.g_mean]("mean")
            .def_method[CoreTree.g_base_value]("base_value")
            .def_method[CoreTree.g_value]("value")
            .def_method[CoreTree.apply]("apply")
            .def_method[CoreTree.predict]("predict")
            .def_method[CoreTree.decision_path]("decision_path")
            .def_method[CoreTree.isolation_path_length]("isolation_path_length")
            .def_method[CoreTree.weighted_decision_path]("weighted_decision_path")
            .def_method[CoreTree.shap_values]("shap_values")
            .def_method[CoreTree.populate_from_arrays]("populate_from_arrays")
            .def_method[CoreTree.getstate]("getstate")
            .def_method[CoreTree.setstate]("setstate")
        )
        _ = (
            m.add_type[CoreDepthFirstBuilder]("CoreDepthFirstBuilder")
            .def_py_init[CoreDepthFirstBuilder.py_init]()
            .def_method[CoreDepthFirstBuilder.build]("build")
        )
        _ = (
            m.add_type[CorePartialFitBuilder]("CorePartialFitBuilder")
            .def_py_init[CorePartialFitBuilder.py_init]()
            .def_method[CorePartialFitBuilder.build]("build")
        )
        var mod = m.finalize()

        var np = Python.import_module("numpy")
        var builtins = Python.import_module("builtins")
        builtins.setattr(mod, "DTYPE", np.dtype(np.float32))
        builtins.setattr(mod, "DOUBLE", np.dtype(np.float64))
        var type_names = List[String]()
        type_names.append("CoreCriterion")
        type_names.append("CoreSplitter")
        type_names.append("CoreTree")
        type_names.append("CoreDepthFirstBuilder")
        type_names.append("CorePartialFitBuilder")
        for name in type_names:
            var t = mod.__dict__[name]
            builtins.setattr(t, "__module__", "shinrin._native_mojo_core")
        return mod
    except e:
        abort(String("failed to create module _native_mojo_core: ", e))
