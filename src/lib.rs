// Rust port of scikit-garden's `skgarden.mondrian.tree` Cython extensions
// (_tree.pyx, _splitter.pyx, _criterion.pyx, _utils.pyx).
//
// Original work: scikit-garden contributors (BSD 3-clause). This file is a
// faithful port of the same algorithms; see NOTICE for provenance.
//
// The Python-facing API mirrors the Cython classes: Tree, DepthFirstTreeBuilder,
// PartialFitTreeBuilder, Splitter, BaseDenseSplitter, MondrianSplitter,
// Criterion, MSE, ClassificationCriterion.

use std::cell::RefCell;
use std::rc::Rc;

use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

// =============================================================================
// Constants and helpers
// =============================================================================

const TREE_LEAF: i64 = -1;
const TREE_UNDEFINED: i64 = -2;
const RAND_R_MAX: u32 = 0x7fff_ffff;

fn our_rand_r(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed % (RAND_R_MAX + 1)
}

fn rand_uniform(low: f64, high: f64, seed: &mut u32) -> f64 {
    (high - low) * our_rand_r(seed) as f64 / RAND_R_MAX as f64 + low
}

fn rand_multinomial(pvals: &[f32], seed: &mut u32) -> i64 {
    let n = pvals.len();
    let mut cum = vec![0f32; n];
    cum[0] = pvals[0];
    for j in 1..n {
        cum[j] = cum[j - 1] + pvals[j];
    }
    let search = rand_uniform(0.0, cum[n - 1] as f64, seed);
    let mut f_j = 0usize;
    for j in 0..n {
        let lower = if j == 0 { 0.0 } else { cum[j - 1] as f64 };
        if cum[j] as f64 >= search && lower < search {
            f_j = j;
            break;
        }
    }
    f_j as i64
}

fn rand_exponential(rate: f32, seed: &mut u32) -> f64 {
    -rand_uniform(0.0, 1.0, seed).ln() / rate as f64
}

fn reduce_tuple<'py>(
    py: Python<'py>,
    class: PyObject,
    args: PyObject,
    state: PyObject,
) -> PyResult<PyObject> {
    let items: Vec<PyObject> = vec![class, args, state];
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

macro_rules! get_dict_item {
    ($d:expr, $k:expr) => {
        $d.get_item($k)?.ok_or_else(|| PyKeyError::new_err($k))?
    };
}

// =============================================================================
// Shared fit-time data
// =============================================================================

#[derive(Clone)]
struct SharedData {
    samples: Rc<RefCell<Vec<i64>>>,
    y: Rc<Vec<f64>>,
    y_stride: usize,
    sample_weight: Rc<Option<Vec<f64>>>,
}

impl SharedData {
    fn weight(&self, i: usize) -> f64 {
        match self.sample_weight.as_ref() {
            Some(sw) => sw[i],
            None => 1.0,
        }
    }
}

// =============================================================================
// Criterion
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum CriterionKind {
    Regression,
    Classification,
}

struct CriterionState {
    kind: CriterionKind,
    n_outputs: usize,
    n_node_samples: usize,
    weighted_n_samples: f64,
    weighted_n_node_samples: f64,
    weighted_n_left: f64,
    weighted_n_right: f64,
    start: usize,
    pos: usize,
    end: usize,
    data: Option<SharedData>,
    sum_total: Vec<f64>,
    sum_left: Vec<f64>,
    sum_right: Vec<f64>,
    sq_sum_total: f64,
    n_classes: Vec<i64>,
    sum_stride: usize,
}

impl CriterionState {
    fn regression(n_outputs: usize, _n_samples: usize) -> Self {
        CriterionState {
            kind: CriterionKind::Regression,
            n_outputs,
            n_node_samples: 0,
            weighted_n_samples: 0.0,
            weighted_n_node_samples: 0.0,
            weighted_n_left: 0.0,
            weighted_n_right: 0.0,
            start: 0,
            pos: 0,
            end: 0,
            data: None,
            sum_total: vec![0.0; n_outputs],
            sum_left: vec![0.0; n_outputs],
            sum_right: vec![0.0; n_outputs],
            sq_sum_total: 0.0,
            n_classes: vec![1],
            sum_stride: n_outputs,
        }
    }

    fn classification(n_outputs: usize, n_classes: Vec<i64>) -> Self {
        let sum_stride = n_classes.iter().copied().max().unwrap_or(0) as usize;
        CriterionState {
            kind: CriterionKind::Classification,
            n_outputs,
            n_node_samples: 0,
            weighted_n_samples: 0.0,
            weighted_n_node_samples: 0.0,
            weighted_n_left: 0.0,
            weighted_n_right: 0.0,
            start: 0,
            pos: 0,
            end: 0,
            data: None,
            sum_total: vec![0.0; n_outputs * sum_stride],
            sum_left: vec![0.0; n_outputs * sum_stride],
            sum_right: vec![0.0; n_outputs * sum_stride],
            sq_sum_total: 0.0,
            n_classes,
            sum_stride,
        }
    }

    fn init(&mut self, data: &SharedData, weighted_n_samples: f64, start: usize, end: usize) {
        self.data = Some(data.clone());
        self.start = start;
        self.end = end;
        self.pos = start;
        self.n_node_samples = end - start;
        self.weighted_n_samples = weighted_n_samples;
        self.weighted_n_node_samples = 0.0;
        let samples = data.samples.borrow();
        match self.kind {
            CriterionKind::Regression => {
                for v in self.sum_total.iter_mut() {
                    *v = 0.0;
                }
                self.sq_sum_total = 0.0;
                for p in start..end {
                    let i = samples[p] as usize;
                    let w = data.weight(i);
                    for k in 0..self.n_outputs {
                        let y_ik = data.y[i * data.y_stride + k];
                        let w_y_ik = w * y_ik;
                        self.sum_total[k] += w_y_ik;
                        self.sq_sum_total += w_y_ik * y_ik;
                    }
                    self.weighted_n_node_samples += w;
                }
            }
            CriterionKind::Classification => {
                for v in self.sum_total.iter_mut() {
                    *v = 0.0;
                }
                for p in start..end {
                    let i = samples[p] as usize;
                    let w = data.weight(i);
                    for k in 0..self.n_outputs {
                        let c = data.y[i * data.y_stride + k] as usize;
                        self.sum_total[k * self.sum_stride + c] += w;
                    }
                    self.weighted_n_node_samples += w;
                }
            }
        }
        self.reset();
    }

    fn reset(&mut self) {
        match self.kind {
            CriterionKind::Regression => {
                for v in self.sum_left.iter_mut() {
                    *v = 0.0;
                }
                self.sum_right.copy_from_slice(&self.sum_total);
                self.weighted_n_left = 0.0;
                self.weighted_n_right = self.weighted_n_node_samples;
                self.pos = self.start;
            }
            CriterionKind::Classification => {
                for (k, &n_cls) in self.n_classes.iter().enumerate() {
                    let base = k * self.sum_stride;
                    self.sum_left[base..base + n_cls as usize].fill(0.0);
                    self.sum_right[base..base + n_cls as usize]
                        .copy_from_slice(&self.sum_total[base..base + n_cls as usize]);
                }
                self.weighted_n_left = 0.0;
                self.weighted_n_right = self.weighted_n_node_samples;
                self.pos = self.start;
            }
        }
    }

    fn reverse_reset(&mut self) {
        match self.kind {
            CriterionKind::Regression => {
                for v in self.sum_right.iter_mut() {
                    *v = 0.0;
                }
                self.sum_left.copy_from_slice(&self.sum_total);
                self.weighted_n_left = self.weighted_n_node_samples;
                self.weighted_n_right = 0.0;
                self.pos = self.end;
            }
            CriterionKind::Classification => {
                for (k, &n_cls) in self.n_classes.iter().enumerate() {
                    let base = k * self.sum_stride;
                    self.sum_right[base..base + n_cls as usize].fill(0.0);
                    self.sum_left[base..base + n_cls as usize]
                        .copy_from_slice(&self.sum_total[base..base + n_cls as usize]);
                }
                self.weighted_n_left = self.weighted_n_node_samples;
                self.weighted_n_right = 0.0;
                self.pos = self.end;
            }
        }
    }

    fn update(&mut self, new_pos: usize) {
        let data = self.data.clone().expect("criterion not initialized");
        let samples = data.samples.borrow();
        match self.kind {
            CriterionKind::Regression => {
                if new_pos.checked_sub(self.pos).unwrap_or(usize::MAX) <= self.end - new_pos {
                    for p in self.pos..new_pos {
                        let i = samples[p] as usize;
                        let w = data.weight(i);
                        for k in 0..self.n_outputs {
                            let y_ik = data.y[i * data.y_stride + k];
                            self.sum_left[k] += w * y_ik;
                        }
                        self.weighted_n_left += w;
                    }
                } else {
                    self.reverse_reset();
                    for p in (new_pos..self.end).rev() {
                        let i = samples[p] as usize;
                        let w = data.weight(i);
                        for k in 0..self.n_outputs {
                            let y_ik = data.y[i * data.y_stride + k];
                            self.sum_left[k] -= w * y_ik;
                        }
                        self.weighted_n_left -= w;
                    }
                }
                self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left;
                for k in 0..self.n_outputs {
                    self.sum_right[k] = self.sum_total[k] - self.sum_left[k];
                }
                self.pos = new_pos;
            }
            CriterionKind::Classification => {
                if new_pos.checked_sub(self.pos).unwrap_or(usize::MAX) <= self.end - new_pos {
                    for p in self.pos..new_pos {
                        let i = samples[p] as usize;
                        let w = data.weight(i);
                        for k in 0..self.n_outputs {
                            let label =
                                k * self.sum_stride + data.y[i * data.y_stride + k] as usize;
                            self.sum_left[label] += w;
                        }
                        self.weighted_n_left += w;
                    }
                } else {
                    self.reverse_reset();
                    for p in (new_pos..self.end).rev() {
                        let i = samples[p] as usize;
                        let w = data.weight(i);
                        for k in 0..self.n_outputs {
                            let label =
                                k * self.sum_stride + data.y[i * data.y_stride + k] as usize;
                            self.sum_left[label] -= w;
                        }
                        self.weighted_n_left -= w;
                    }
                }
                self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left;
                for (k, &n_cls) in self.n_classes.iter().enumerate() {
                    let base = k * self.sum_stride;
                    for c in 0..n_cls as usize {
                        self.sum_right[base + c] =
                            self.sum_total[base + c] - self.sum_left[base + c];
                    }
                }
                self.pos = new_pos;
            }
        }
    }

    fn node_impurity(&self) -> f64 {
        match self.kind {
            CriterionKind::Regression => {
                let mut impurity = self.sq_sum_total / self.weighted_n_node_samples;
                for k in 0..self.n_outputs {
                    let mean = self.sum_total[k] / self.weighted_n_node_samples;
                    impurity -= mean * mean;
                }
                impurity / self.n_outputs as f64
            }
            CriterionKind::Classification => f64::INFINITY,
        }
    }

    fn children_impurity(&self, impurity_left: &mut f64, impurity_right: &mut f64) {
        match self.kind {
            CriterionKind::Regression => {
                let data = self.data.clone().expect("criterion not initialized");
                let samples = data.samples.borrow();
                let mut sq_sum_left = 0.0;
                for p in self.start..self.pos {
                    let i = samples[p] as usize;
                    let w = data.weight(i);
                    for k in 0..self.n_outputs {
                        let y_ik = data.y[i * data.y_stride + k];
                        sq_sum_left += w * y_ik * y_ik;
                    }
                }
                let sq_sum_right = self.sq_sum_total - sq_sum_left;
                let mut il = sq_sum_left / self.weighted_n_left;
                let mut ir = sq_sum_right / self.weighted_n_right;
                for k in 0..self.n_outputs {
                    let ml = self.sum_left[k] / self.weighted_n_left;
                    let mr = self.sum_right[k] / self.weighted_n_right;
                    il -= ml * ml;
                    ir -= mr * mr;
                }
                *impurity_left = il / self.n_outputs as f64;
                *impurity_right = ir / self.n_outputs as f64;
            }
            CriterionKind::Classification => {
                *impurity_left = 0.0;
                *impurity_right = 0.0;
            }
        }
    }

    fn node_value(&self, dest: &mut [f64]) {
        match self.kind {
            CriterionKind::Regression => {
                let n = self.n_outputs.min(dest.len());
                for (d, s) in dest.iter_mut().zip(&self.sum_total).take(n) {
                    *d = s / self.weighted_n_node_samples;
                }
            }
            CriterionKind::Classification => {
                let n = self.sum_total.len().min(dest.len());
                dest[..n].copy_from_slice(&self.sum_total[..n]);
            }
        }
    }

    fn is_pure(&self) -> bool {
        match self.kind {
            CriterionKind::Regression => self.node_impurity() == 0.0,
            CriterionKind::Classification => {
                let mut pure = true;
                for (k, &n_cls) in self.n_classes.iter().enumerate() {
                    let base = k * self.sum_stride;
                    let mut output_pure = false;
                    for c in 0..n_cls as usize {
                        if self.sum_total[base + c] == self.n_node_samples as f64 {
                            output_pure = true;
                            break;
                        }
                    }
                    if !output_pure {
                        pure = false;
                        break;
                    }
                }
                pure
            }
        }
    }
}

#[pyclass(unsendable, subclass, module = "shinrin._native", name = "Criterion")]
struct PyCriterion {
    inner: RefCell<CriterionState>,
}

#[pyclass(unsendable, extends=PyCriterion, module="shinrin._native", name="MSE")]
struct PyMSE {
    n_outputs: usize,
    n_samples: usize,
}

#[pyclass(unsendable, extends=PyCriterion, module="shinrin._native", name="ClassificationCriterion")]
struct PyClassificationCriterion {
    n_outputs: usize,
    n_classes: Vec<isize>,
}

#[pymethods]
impl PyCriterion {
    fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    fn __setstate__(&self, _d: &Bound<'_, PyDict>) {}
}

#[pymethods]
impl PyMSE {
    #[new]
    fn new(n_outputs: usize, n_samples: usize) -> (Self, PyCriterion) {
        (
            PyMSE {
                n_outputs,
                n_samples,
            },
            PyCriterion {
                inner: RefCell::new(CriterionState::regression(n_outputs, n_samples)),
            },
        )
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let args = PyTuple::new(py, [self.n_outputs, self.n_samples])?
            .into_any()
            .unbind();
        let state = PyDict::new(py).into_any().unbind();
        reduce_tuple(
            py,
            PyType::new::<PyMSE>(py).into_any().unbind(),
            args,
            state,
        )
    }
}

#[pymethods]
impl PyClassificationCriterion {
    #[new]
    fn new(n_outputs: usize, n_classes: PyReadonlyArray1<isize>) -> (Self, PyCriterion) {
        let n_classes_vec: Vec<isize> = n_classes
            .as_slice()
            .map_or_else(|_| Vec::new(), |s| s.to_vec());
        (
            PyClassificationCriterion {
                n_outputs,
                n_classes: n_classes_vec.clone(),
            },
            PyCriterion {
                inner: RefCell::new(CriterionState::classification(
                    n_outputs,
                    n_classes_vec.iter().map(|&v| v as i64).collect(),
                )),
            },
        )
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let n_classes_arr = PyArray1::from_vec(py, self.n_classes.clone());
        let args = PyTuple::new(
            py,
            [
                self.n_outputs.into_pyobject(py)?.into_any().unbind(),
                n_classes_arr.into_any().unbind(),
            ],
        )?
        .into_any()
        .unbind();
        let state = PyDict::new(py).into_any().unbind();
        reduce_tuple(
            py,
            PyType::new::<PyClassificationCriterion>(py)
                .into_any()
                .unbind(),
            args,
            state,
        )
    }
}

// =============================================================================
// Splitter
// =============================================================================

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct SplitRecord {
    feature: i64,
    pos: usize,
    threshold: f64,
    improvement: f64,
    impurity_left: f64,
    impurity_right: f64,
    e: f64,
}

impl Default for SplitRecord {
    fn default() -> Self {
        SplitRecord {
            feature: 0,
            pos: 0,
            threshold: 0.0,
            improvement: 0.0,
            impurity_left: 0.0,
            impurity_right: 0.0,
            e: 0.0,
        }
    }
}

struct SplitterState {
    data: Option<SharedData>,
    x: Vec<f32>,
    n_features: usize,
    n_samples: usize,
    weighted_n_samples: f64,
    features: Vec<i64>,
    constant_features: Vec<i64>,
    feature_values: Vec<f32>,
    start: usize,
    end: usize,
    rand_r_state: u32,
    lower_bounds: Vec<f32>,
    upper_bounds: Vec<f32>,
}

impl SplitterState {
    fn new() -> Self {
        SplitterState {
            data: None,
            x: Vec::new(),
            n_features: 0,
            n_samples: 0,
            weighted_n_samples: 0.0,
            features: Vec::new(),
            constant_features: Vec::new(),
            feature_values: Vec::new(),
            start: 0,
            end: 0,
            rand_r_state: 0,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn init(
        &mut self,
        x: Vec<f32>,
        n_samples: usize,
        n_features: usize,
        y: Vec<f64>,
        y_stride: usize,
        sample_weight: Option<Vec<f64>>,
        rand_r_state: u32,
    ) {
        self.x = x;
        self.n_features = n_features;
        self.rand_r_state = rand_r_state;

        let mut samples = Vec::with_capacity(n_samples);
        let mut weighted_n_samples = 0.0;
        for i in 0..n_samples {
            let zero_weight = sample_weight.as_ref().is_some_and(|sw| sw[i] == 0.0);
            if !zero_weight {
                samples.push(i as i64);
            }
            weighted_n_samples += sample_weight.as_ref().map_or(1.0, |sw| sw[i]);
        }
        self.n_samples = samples.len();
        self.weighted_n_samples = weighted_n_samples;

        self.features = (0..n_features as i64).collect();
        self.constant_features = vec![0; n_features];
        self.feature_values = vec![0.0; n_samples];
        self.lower_bounds = vec![0.0; n_features];
        self.upper_bounds = vec![0.0; n_features];

        self.data = Some(SharedData {
            samples: Rc::new(RefCell::new(samples)),
            y: Rc::new(y),
            y_stride,
            sample_weight: Rc::new(sample_weight),
        });
    }

    fn node_reset(&mut self, start: usize, end: usize, criterion: &PyCriterion) -> f64 {
        self.start = start;
        self.end = end;
        let data = self.data.clone().expect("splitter not initialized");
        let mut crit = criterion.inner.borrow_mut();
        crit.init(&data, self.weighted_n_samples, start, end);
        crit.weighted_n_node_samples
    }

    fn set_bounds(&mut self) {
        let n_features = self.n_features;
        let data = self.data.clone().expect("splitter not initialized");
        let samples = data.samples.borrow();
        let x = &self.x;
        for f in 0..n_features {
            let mut upper_bound = f32::NEG_INFINITY;
            let mut lower_bound = f32::INFINITY;
            for p in self.start..self.end {
                let i = samples[p] as usize;
                let current_f = x[i * n_features + f];
                if current_f <= lower_bound {
                    lower_bound = current_f;
                }
                if current_f > upper_bound {
                    upper_bound = current_f;
                }
            }
            self.upper_bounds[f] = upper_bound;
            self.lower_bounds[f] = lower_bound;
        }
    }

    fn node_split(&mut self, criterion: &PyCriterion) -> SplitRecord {
        self.set_bounds();
        let n_features = self.n_features;
        let mut pvals = vec![0f32; n_features];
        let mut rate = 0f32;
        for (f, pv) in pvals.iter_mut().enumerate() {
            let ub = self.upper_bounds[f];
            let lb = self.lower_bounds[f];
            *pv = ub - lb;
            rate += ub - lb;
        }
        let e = rand_exponential(rate, &mut self.rand_r_state);
        let feature = rand_multinomial(&pvals, &mut self.rand_r_state);
        let threshold = rand_uniform(
            self.lower_bounds[feature as usize] as f64,
            self.upper_bounds[feature as usize] as f64,
            &mut self.rand_r_state,
        );

        let data = self.data.clone().expect("splitter not initialized");
        let mut samples = data.samples.borrow_mut();
        let x = &self.x;
        let mut partition_end = self.end;
        let mut p = self.start;
        while p < partition_end {
            let i = samples[p] as usize;
            if (x[i * n_features + feature as usize] as f64) <= threshold {
                p += 1;
            } else {
                partition_end -= 1;
                samples.swap(p, partition_end);
            }
        }
        drop(samples);

        let mut crit = criterion.inner.borrow_mut();
        crit.reset();
        crit.update(p);
        let mut impurity_left = 0.0;
        let mut impurity_right = 0.0;
        crit.children_impurity(&mut impurity_left, &mut impurity_right);
        drop(crit);

        SplitRecord {
            feature,
            pos: p,
            threshold,
            impurity_left,
            impurity_right,
            e,
            ..Default::default()
        }
    }

    fn node_value_into(&self, criterion: &PyCriterion, dest: &mut [f64]) {
        let crit = criterion.inner.borrow();
        crit.node_value(dest);
    }

    fn node_impurity(&self, criterion: &PyCriterion) -> f64 {
        let crit = criterion.inner.borrow();
        crit.node_impurity()
    }

    fn is_pure(&self, criterion: &PyCriterion) -> bool {
        let crit = criterion.inner.borrow();
        crit.is_pure()
    }
}

#[pyclass(unsendable, subclass, module = "shinrin._native", name = "Splitter")]
struct PySplitter {
    inner: RefCell<SplitterState>,
    criterion: Py<PyAny>,
    random_state: Py<PyAny>,
}

#[pyclass(unsendable, extends=PySplitter, module="shinrin._native", name="BaseDenseSplitter")]
struct PyBaseDenseSplitter;

#[pyclass(unsendable, extends=PySplitter, module="shinrin._native", name="MondrianSplitter")]
struct PyMondrianSplitter;

#[pymethods]
impl PySplitter {
    #[new]
    fn new(criterion: Py<PyAny>, random_state: Py<PyAny>) -> Self {
        PySplitter {
            inner: RefCell::new(SplitterState::new()),
            criterion,
            random_state,
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    fn __setstate__(&self, _d: &Bound<'_, PyDict>) {}
}

#[pymethods]
impl PyMondrianSplitter {
    #[new]
    fn new(criterion: Py<PyAny>, random_state: Py<PyAny>) -> PyResult<(Self, PySplitter)> {
        Ok((
            PyMondrianSplitter,
            PySplitter {
                inner: RefCell::new(SplitterState::new()),
                criterion,
                random_state,
            },
        ))
    }

    fn __reduce__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyObject> {
        let base = slf.into_super();
        let args = PyTuple::new(
            py,
            [
                base.criterion
                    .clone_ref(py)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                base.random_state
                    .clone_ref(py)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            ],
        )?
        .into_any()
        .unbind();
        let state = PyDict::new(py).into_any().unbind();
        reduce_tuple(
            py,
            PyType::new::<PyMondrianSplitter>(py).into_any().unbind(),
            args,
            state,
        )
    }
}

// =============================================================================
// Tree
// =============================================================================

#[derive(Clone)]
struct Node {
    left_child: i64,
    right_child: i64,
    feature: i64,
    threshold: f64,
    impurity: f64,
    n_node_samples: i64,
    weighted_n_node_samples: f64,
    lower_bounds: Vec<f32>,
    upper_bounds: Vec<f32>,
    tau: f32,
    variance: f64,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            left_child: 0,
            right_child: 0,
            feature: TREE_UNDEFINED,
            threshold: 0.0,
            impurity: 0.0,
            n_node_samples: 0,
            weighted_n_node_samples: 0.0,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            tau: 0.0,
            variance: 0.0,
        }
    }
}

struct TreeState {
    n_features: usize,
    n_classes: Vec<i64>,
    n_outputs: usize,
    max_n_classes: i64,
    max_depth: i64,
    node_count: i64,
    capacity: i64,
    nodes: Vec<Node>,
    value: Vec<f64>,
    value_stride: usize,
    root: i64,
}

impl TreeState {
    fn new(n_features: usize, n_classes: Vec<i64>, n_outputs: usize) -> Self {
        let max_n_classes = n_classes.iter().copied().max().unwrap_or(0);
        TreeState {
            n_features,
            n_classes,
            n_outputs,
            max_n_classes,
            max_depth: 0,
            node_count: 0,
            capacity: 0,
            nodes: Vec::new(),
            value: Vec::new(),
            value_stride: n_outputs * max_n_classes as usize,
            root: 0,
        }
    }

    fn resize_c(&mut self, mut capacity: i64) -> bool {
        if capacity == self.capacity && !self.nodes.is_empty() {
            return true;
        }
        if capacity < 0 {
            capacity = if self.capacity == 0 {
                3
            } else {
                2 * self.capacity
            };
        }
        if capacity < self.node_count {
            self.node_count = capacity;
        }
        let cap = capacity as usize;
        if cap != self.nodes.len() {
            self.nodes.resize(cap, Node::default());
        }
        let new_value_len = cap * self.value_stride;
        if new_value_len != self.value.len() {
            self.value.resize(new_value_len, 0.0);
        }
        self.capacity = capacity;
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn add_node(
        &mut self,
        parent: i64,
        is_left: bool,
        is_leaf: bool,
        feature: i64,
        threshold: f64,
        impurity: f64,
        n_node_samples: i64,
        weighted_n_node_samples: f64,
        lower_bounds: Vec<f32>,
        upper_bounds: Vec<f32>,
        e: f64,
    ) -> i64 {
        let node_id = self.node_count;
        if node_id >= self.capacity && !self.resize_c(-1) {
            return TREE_LEAF;
        }
        if parent != TREE_UNDEFINED {
            if is_left {
                self.nodes[parent as usize].left_child = node_id;
            } else {
                self.nodes[parent as usize].right_child = node_id;
            }
        }

        let tau = if parent == TREE_UNDEFINED {
            e as f32
        } else if is_leaf {
            f32::INFINITY
        } else {
            (e + self.nodes[parent as usize].tau as f64) as f32
        };

        let mut node = Node {
            impurity,
            n_node_samples,
            weighted_n_node_samples,
            lower_bounds,
            upper_bounds,
            tau,
            variance: impurity,
            ..Default::default()
        };
        if is_leaf {
            node.left_child = TREE_LEAF;
            node.right_child = TREE_LEAF;
            node.feature = TREE_UNDEFINED;
            node.threshold = TREE_UNDEFINED as f64;
        } else {
            node.feature = feature;
            node.threshold = threshold;
        }
        self.nodes[node_id as usize] = node;
        self.node_count += 1;
        node_id
    }

    #[allow(clippy::too_many_arguments)]
    fn set_node_attributes(
        &mut self,
        node_ind: usize,
        left_child: i64,
        right_child: i64,
        feature: i64,
        threshold: f64,
        tau: f32,
        n_node_samples: i64,
        weighted_n_node_samples: f64,
        impurity: f64,
        variance: f64,
        x: &[f32],
        x_start: usize,
        n_features: usize,
        y: &[f64],
        y_start: usize,
        y_stride: usize,
        child_ind: i64,
    ) {
        let bounds = if child_ind == -1 {
            let lb: Vec<f32> = (0..n_features).map(|f| x[x_start + f]).collect();
            let ub = lb.clone();
            (lb, ub)
        } else {
            self.bounds_for_node(child_ind, x, x_start, n_features)
        };
        {
            let node = &mut self.nodes[node_ind];
            node.left_child = left_child;
            node.right_child = right_child;
            node.feature = feature;
            node.threshold = threshold;
            node.tau = tau;
            node.n_node_samples = n_node_samples;
            node.weighted_n_node_samples = weighted_n_node_samples;
            node.impurity = impurity;
            node.variance = variance;
            node.lower_bounds = bounds.0;
            node.upper_bounds = bounds.1;
        }
        let val_ptr = node_ind * self.value_stride;
        if child_ind == -1 {
            if self.n_classes[0] == 1 {
                self.value[val_ptr] = y[y_start * y_stride];
            } else {
                let c = y[y_start * y_stride] as usize;
                self.value[val_ptr + c] = 1.0;
            }
        }
    }

    fn bounds_for_node(
        &self,
        child_ind: i64,
        x: &[f32],
        x_start: usize,
        n_features: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let child = &self.nodes[child_ind as usize];
        let mut lb = Vec::with_capacity(n_features);
        let mut ub = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let x_val = x[x_start + f];
            lb.push(x_val.min(child.lower_bounds[f]));
            ub.push(x_val.max(child.upper_bounds[f]));
        }
        (lb, ub)
    }

    fn update_node_info(
        &mut self,
        parent_id: usize,
        child_id: usize,
        y: &[f64],
        y_start: usize,
        y_stride: usize,
    ) {
        let is_regression = self.n_classes[0] == 1;
        let child_ptr = child_id * self.value_stride;
        let parent_ptr = parent_id * self.value_stride;
        let child_n = self.nodes[child_id].n_node_samples;
        let y_val = y[y_start * y_stride];
        if is_regression {
            let old_mean = self.value[child_ptr];
            let new_sum = old_mean * child_n as f64 + y_val;
            let new_mean = new_sum / (child_n + 1) as f64;
            self.value[parent_ptr] = new_mean;
            let ss = (self.nodes[child_id].variance + old_mean * old_mean) * child_n as f64;
            let parent = &mut self.nodes[parent_id];
            parent.variance = (ss + y_val * y_val) / (child_n + 1) as f64 - new_mean * new_mean;
        } else {
            self.value[parent_ptr + y_val as usize] += 1.0;
            if child_id != parent_id {
                let n_cls = self.n_classes[0] as usize;
                for c in 0..n_cls {
                    self.value[parent_ptr + c] += self.value[child_ptr + c];
                }
            }
        }
    }

    fn init(&mut self, x: &[f32], n_features: usize, y: &[f64], y_stride: usize) {
        self.set_node_attributes(
            0,
            TREE_LEAF,
            TREE_LEAF,
            TREE_UNDEFINED,
            TREE_UNDEFINED as f64,
            f32::INFINITY,
            1,
            1.0,
            0.0,
            0.0,
            x,
            0,
            n_features,
            y,
            0,
            y_stride,
            -1,
        );
        self.node_count += 1;
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        &mut self,
        x: &[f32],
        x_start: usize,
        n_features: usize,
        y: &[f64],
        y_start: usize,
        y_stride: usize,
        random_state: u32,
        min_samples_split: i64,
    ) -> PyResult<()> {
        let mut rand_r_state = random_state;
        let mut curr_id = self.root;
        let mut parent_id: i64 = -1;
        let mut tau_parent = 0.0f32;

        loop {
            let curr_lb = self.nodes[curr_id as usize].lower_bounds.clone();
            let curr_ub = self.nodes[curr_id as usize].upper_bounds.clone();
            let curr_tau = self.nodes[curr_id as usize].tau;
            let curr_n = self.nodes[curr_id as usize].n_node_samples;
            let curr_wnn = self.nodes[curr_id as usize].weighted_n_node_samples;

            let mut new_rate = 0f32;
            let mut extent = vec![0f32; n_features];
            for f in 0..n_features {
                let x_val = x[x_start + f];
                let e_l = (curr_lb[f] - x_val).max(0.0);
                let e_u = (x_val - curr_ub[f]).max(0.0);
                extent[f] = e_l + e_u;
                new_rate += extent[f];
            }
            let e = rand_exponential(new_rate, &mut rand_r_state) as f32;

            if (tau_parent + e) < curr_tau && (curr_n + 1) >= min_samples_split {
                let new_child_id = self.node_count;
                let new_parent_id = self.node_count + 1;

                let delta = rand_multinomial(&extent, &mut rand_r_state);
                let x_val = x[x_start + delta as usize];
                let l_b = curr_lb[delta as usize];
                let u_b = curr_ub[delta as usize];
                let xi = if x_val > u_b {
                    rand_uniform(u_b as f64, x_val as f64, &mut rand_r_state) as f32
                } else {
                    rand_uniform(x_val as f64, l_b as f64, &mut rand_r_state) as f32
                };

                let (left_child, right_child) = if x_val < xi {
                    (new_child_id, curr_id)
                } else {
                    (curr_id, new_child_id)
                };

                if !self.resize_c(self.node_count + 2) {
                    return Err(PyMemoryError::new_err("resizing tree failed"));
                }

                self.set_node_attributes(
                    new_child_id as usize,
                    TREE_LEAF,
                    TREE_LEAF,
                    TREE_UNDEFINED,
                    TREE_UNDEFINED as f64,
                    f32::INFINITY,
                    1,
                    1.0,
                    0.0,
                    0.0,
                    x,
                    x_start,
                    n_features,
                    y,
                    y_start,
                    y_stride,
                    -1,
                );

                self.set_node_attributes(
                    new_parent_id as usize,
                    left_child,
                    right_child,
                    delta,
                    xi as f64,
                    tau_parent + e,
                    curr_n + 1,
                    curr_wnn + 1.0,
                    0.0,
                    0.0,
                    x,
                    x_start,
                    n_features,
                    y,
                    y_start,
                    y_stride,
                    curr_id,
                );
                self.update_node_info(
                    new_parent_id as usize,
                    curr_id as usize,
                    y,
                    y_start,
                    y_stride,
                );

                if curr_id == self.root {
                    self.root = new_parent_id;
                } else {
                    if self.nodes[parent_id as usize].left_child == curr_id {
                        self.nodes[parent_id as usize].left_child = new_parent_id;
                    } else {
                        self.nodes[parent_id as usize].right_child = new_parent_id;
                    }
                }
                self.max_depth += 1;
                self.node_count += 2;
                break;
            } else {
                self.update_extent_inplace(curr_id as usize, x, x_start, n_features);
                self.update_node_info(curr_id as usize, curr_id as usize, y, y_start, y_stride);
                self.nodes[curr_id as usize].n_node_samples += 1;
                self.nodes[curr_id as usize].weighted_n_node_samples += 1.0;
                if self.nodes[curr_id as usize].left_child == TREE_LEAF {
                    break;
                }
                parent_id = curr_id;
                if (x[x_start + self.nodes[curr_id as usize].feature as usize] as f64)
                    < self.nodes[curr_id as usize].threshold
                {
                    curr_id = self.nodes[curr_id as usize].left_child;
                } else {
                    curr_id = self.nodes[curr_id as usize].right_child;
                }
                tau_parent = self.nodes[parent_id as usize].tau;
            }
        }
        Ok(())
    }

    fn update_extent_inplace(
        &mut self,
        node_ind: usize,
        x: &[f32],
        x_start: usize,
        n_features: usize,
    ) {
        for f in 0..n_features {
            let x_val = x[x_start + f];
            let node = &mut self.nodes[node_ind];
            node.lower_bounds[f] = x_val.min(node.lower_bounds[f]);
            node.upper_bounds[f] = x_val.max(node.upper_bounds[f]);
        }
    }
}

#[pyclass(unsendable, module = "shinrin._native", name = "Tree")]
struct PyTree {
    inner: RefCell<TreeState>,
}

#[pymethods]
impl PyTree {
    #[new]
    fn new(
        n_features: usize,
        n_classes: PyReadonlyArray1<isize>,
        n_outputs: usize,
    ) -> PyResult<Self> {
        let n_classes_vec: Vec<i64> = n_classes
            .as_slice()
            .map_err(|_| PyValueError::new_err("n_classes must be a 1d intp array"))?
            .iter()
            .map(|&v| v as i64)
            .collect();
        Ok(PyTree {
            inner: RefCell::new(TreeState::new(n_features, n_classes_vec, n_outputs)),
        })
    }

    #[getter]
    fn node_count(&self) -> i64 {
        self.inner.borrow().node_count
    }

    #[getter]
    fn capacity(&self) -> i64 {
        self.inner.borrow().capacity
    }

    #[getter]
    fn max_depth(&self) -> i64 {
        self.inner.borrow().max_depth
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.borrow().n_features
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.borrow().n_outputs
    }

    #[getter]
    fn max_n_classes(&self) -> i64 {
        self.inner.borrow().max_n_classes
    }

    #[getter]
    fn root(&self) -> i64 {
        self.inner.borrow().root
    }

    #[getter]
    fn n_classes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<isize>> {
        let inner = self.inner.borrow();
        PyArray1::from_vec(py, inner.n_classes.iter().map(|&v| v as isize).collect())
    }

    #[getter]
    fn children_left<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<isize>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(
            py,
            (0..n).map(|i| inner.nodes[i].left_child as isize).collect(),
        )
    }

    #[getter]
    fn children_right<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<isize>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(
            py,
            (0..n)
                .map(|i| inner.nodes[i].right_child as isize)
                .collect(),
        )
    }

    #[getter]
    fn feature<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<isize>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(
            py,
            (0..n).map(|i| inner.nodes[i].feature as isize).collect(),
        )
    }

    #[getter]
    fn threshold<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(py, (0..n).map(|i| inner.nodes[i].threshold).collect())
    }

    #[getter]
    fn impurity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(py, (0..n).map(|i| inner.nodes[i].impurity).collect())
    }

    #[getter]
    fn n_node_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<isize>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(
            py,
            (0..n)
                .map(|i| inner.nodes[i].n_node_samples as isize)
                .collect(),
        )
    }

    #[getter]
    fn weighted_n_node_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(
            py,
            (0..n)
                .map(|i| inner.nodes[i].weighted_n_node_samples)
                .collect(),
        )
    }

    #[getter]
    fn tau<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(py, (0..n).map(|i| inner.nodes[i].tau).collect())
    }

    #[getter]
    fn lower_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        let nf = inner.n_features;
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let lb = &inner.nodes[i].lower_bounds;
                (0..nf).map(|f| lb.get(f).copied().unwrap_or(0.0)).collect()
            })
            .collect();
        PyArray2::from_vec2(py, &rows).map_err(PyErr::from)
    }

    #[getter]
    fn upper_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        let nf = inner.n_features;
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let ub = &inner.nodes[i].upper_bounds;
                (0..nf).map(|f| ub.get(f).copied().unwrap_or(0.0)).collect()
            })
            .collect();
        PyArray2::from_vec2(py, &rows).map_err(PyErr::from)
    }

    #[getter]
    fn variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        PyArray1::from_vec(py, (0..n).map(|i| inner.nodes[i].variance).collect())
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        let total = n * inner.value_stride;
        PyArray1::from_vec(py, inner.value[..total].to_vec())
    }

    #[getter]
    fn base_value<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.borrow();
        // base_value is the root node value (prediction at root, before any splits)
        let root = inner.root as usize;
        let val = &inner.value[root * inner.value_stride..(root + 1) * inner.value_stride];
        PyArray1::from_vec(py, val.to_vec())
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        let n_outputs = inner.n_outputs;
        let m = inner.max_n_classes as usize;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut outputs = Vec::with_capacity(n_outputs);
            for k in 0..n_outputs {
                let start = i * inner.value_stride + k * m;
                outputs.push(inner.value[start..start + m].to_vec());
            }
            rows.push(outputs);
        }
        Ok(PyArray3::from_vec3(py, &rows)?)
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<isize>>> {
        let inner = self.inner.borrow();
        let shape = x.shape();
        let n_samples = shape[0];
        let x_readonly = x.readonly();
        let view = x_readonly.as_array();
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut curr = inner.root;
            loop {
                let node = &inner.nodes[curr as usize];
                if node.left_child == TREE_LEAF {
                    break;
                }
                if (view[[i, node.feature as usize]] as f64) <= node.threshold {
                    curr = node.left_child;
                } else {
                    curr = node.right_child;
                }
            }
            out.push(curr as isize);
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[pyo3(signature = (x, return_std=false, is_regression=true))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyArray2<f32>>,
        return_std: bool,
        is_regression: bool,
    ) -> PyResult<PyObject> {
        let inner = self.inner.borrow();
        let shape = x.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let x_readonly = x.readonly();
        let view = x_readonly.as_array();

        let node_count = inner.node_count as usize;
        let max_n_classes = inner.max_n_classes as usize;
        let value_stride = inner.value_stride;
        // node_values = value[:, 0, :]
        let node_values: Vec<f64> = (0..node_count)
            .flat_map(|i| inner.value[i * value_stride..i * value_stride + max_n_classes].to_vec())
            .collect();

        let n_classes = max_n_classes;
        let n_node_samples: Vec<f64> = (0..node_count)
            .map(|i| inner.nodes[i].n_node_samples as f64)
            .collect();

        let mut mean = vec![0f32; n_samples];
        let mut std = vec![0f32; n_samples];
        let mut proba = vec![0f32; n_samples * n_classes];

        for i in 0..n_samples {
            let mut parent_tau = 0.0f64;
            let mut p_nsy = 1.0f64;
            let mut node_id = inner.root as usize;
            loop {
                let node = &inner.nodes[node_id];
                let delta = node.tau as f64 - parent_tau;
                parent_tau = node.tau as f64;

                let mut eta = 0.0f64;
                for f in 0..n_features {
                    let x_val = view[[i, f]] as f64;
                    eta += (x_val - node.upper_bounds[f] as f64).max(0.0)
                        + (node.lower_bounds[f] as f64 - x_val).max(0.0);
                }

                let (w_j, p_js) = if node.left_child == TREE_LEAF {
                    (p_nsy, 0.0f64)
                } else {
                    let p_js = 1.0 - (-delta * eta).exp();
                    (p_nsy * p_js, p_js)
                };

                if is_regression {
                    mean[i] = (mean[i] as f64 + w_j * node_values[node_id * n_classes]) as f32;
                } else {
                    for c in 0..n_classes {
                        let val =
                            w_j * (node_values[node_id * n_classes + c] / n_node_samples[node_id]);
                        proba[i * n_classes + c] = (proba[i * n_classes + c] as f64 + val) as f32;
                    }
                }
                if return_std {
                    let val = w_j * (node_values[node_id * n_classes].powi(2) + node.variance);
                    std[i] = (std[i] as f64 + val) as f32;
                }

                if node.left_child == TREE_LEAF {
                    break;
                }
                p_nsy *= 1.0 - p_js;

                if view[[i, node.feature as usize]] as f64 <= node.threshold {
                    node_id = node.left_child as usize;
                } else {
                    node_id = node.right_child as usize;
                }
            }
            if return_std {
                let m = mean[i] as f64;
                let s = (std[i] as f64 - m * m).max(0.0).sqrt();
                std[i] = s as f32;
            }
        }

        if is_regression {
            let mean_arr = PyArray1::from_vec(py, mean);
            if return_std {
                let std_arr = PyArray1::from_vec(py, std);
                Ok(PyTuple::new(py, [mean_arr, std_arr])?.into_any().unbind())
            } else {
                Ok(PyTuple::new(py, [mean_arr])?.into_any().unbind())
            }
        } else {
            let rows: Vec<Vec<f32>> = (0..n_samples)
                .map(|i| proba[i * n_classes..(i + 1) * n_classes].to_vec())
                .collect();
            let proba_arr = PyArray2::from_vec2(py, &rows)?;
            Ok(PyTuple::new(py, [proba_arr])?.into_any().unbind())
        }
    }

    fn decision_path<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<PyObject> {
        let inner = self.inner.borrow();
        let shape = x.shape();
        let n_samples = shape[0];
        let x_readonly = x.readonly();
        let view = x_readonly.as_array();
        let node_count = inner.node_count;

        let mut indptr = Vec::with_capacity(n_samples + 1);
        let mut indices = Vec::with_capacity(n_samples * (1 + inner.max_depth as usize));
        indptr.push(0isize);
        for i in 0..n_samples {
            let mut curr = inner.root;
            while curr != TREE_LEAF {
                let node = &inner.nodes[curr as usize];
                indices.push(curr as isize);
                if node.left_child == TREE_LEAF {
                    break;
                }
                if (view[[i, node.feature as usize]] as f64) <= node.threshold {
                    curr = node.left_child;
                } else {
                    curr = node.right_child;
                }
            }
            indptr.push(indices.len() as isize);
        }

        let data = vec![1isize; indices.len()];
        let indptr_arr = PyArray1::from_vec(py, indptr);
        let indices_arr = PyArray1::from_vec(py, indices);
        let data_arr = PyArray1::from_vec(py, data);
        build_csr(
            py,
            data_arr.into_any().unbind(),
            indices_arr,
            indptr_arr,
            n_samples,
            node_count,
        )
    }

    fn isolation_path_length<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let inner = self.inner.borrow();
        let shape = x.shape();
        let n_samples = shape[0];
        let x_readonly = x.readonly();
        let view = x_readonly.as_array();
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut curr = inner.root;
            let mut depth = 0usize;
            loop {
                let node = &inner.nodes[curr as usize];
                if node.left_child == TREE_LEAF {
                    break;
                }
                depth += 1;
                if (view[[i, node.feature as usize]] as f64) <= node.threshold {
                    curr = node.left_child;
                } else {
                    curr = node.right_child;
                }
            }
            out.push(depth as f64);
        }
        Ok(PyArray1::from_vec(py, out))
    }

    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = self.inner.borrow();
        let shape = x.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let x_readonly = x.readonly();
        let view = x_readonly.as_array();
        let node_count = inner.node_count as usize;
        let is_regression = inner.n_classes[0] == 1;
        let max_n_classes = inner.max_n_classes as usize;
        let value_stride = inner.value_stride;

        // Precompute node values for regression (leaf mean) and classification (class counts)
        let node_values: Vec<f64> = (0..node_count)
            .flat_map(|i| inner.value[i * value_stride..i * value_stride + max_n_classes].to_vec())
            .collect();
        let n_node_samples: Vec<f64> = (0..node_count)
            .map(|i| inner.nodes[i].n_node_samples as f64)
            .collect();

        // For each sample, traverse from root to leaf and compute SHAP values
        // using the tree SHAP approximation (Lundberg et al. 2020).
        let mut out = vec![0f64; n_samples * n_features];

        for i in 0..n_samples {
            // Traverse path
            let mut path: Vec<usize> = Vec::new();
            let mut curr = inner.root as usize;
            loop {
                path.push(curr);
                let node = &inner.nodes[curr];
                if node.left_child == TREE_LEAF {
                    break;
                }
                if view[[i, node.feature as usize]] as f64 <= node.threshold {
                    curr = node.left_child as usize;
                } else {
                    curr = node.right_child as usize;
                }
            }

            // For each node on the path, compute SHAP value for the split feature
            for &node_idx in &path {
                let node = &inner.nodes[node_idx];
                if node.left_child != TREE_LEAF {
                    // Internal node: compute SHAP value for the split feature
                    let feat = node.feature as usize;
                    if feat >= n_features {
                        continue;
                    }
                    let x_val = view[[i, feat]] as f64;
                    let threshold = node.threshold;

                    // Compute conditional expectations E[Y | X_S] for S = features on path before this node
                    // Approximation: use the node value as E[Y | path so far],
                    // and the average of children values as E[Y | path without this feature]
                    let val_parent = if is_regression {
                        node_values[node_idx * max_n_classes]
                    } else {
                        // For classification, use the predicted class probability
                        // Sum of node values normalized by node samples
                        let total = n_node_samples[node_idx];
                        if total > 0.0 {
                            let sum: f64 = (0..max_n_classes)
                                .map(|c| node_values[node_idx * max_n_classes + c])
                                .sum();
                            sum / total
                        } else {
                            0.0
                        }
                    };

                    let val_left = if is_regression {
                        let left_idx = node.left_child as usize;
                        node_values[left_idx * max_n_classes]
                    } else {
                        let left_idx = node.left_child as usize;
                        let total = n_node_samples[left_idx];
                        if total > 0.0 {
                            let sum: f64 = (0..max_n_classes)
                                .map(|c| node_values[left_idx * max_n_classes + c])
                                .sum();
                            sum / total
                        } else {
                            0.0
                        }
                    };

                    let val_right = if is_regression {
                        let right_idx = node.right_child as usize;
                        node_values[right_idx * max_n_classes]
                    } else {
                        let right_idx = node.right_child as usize;
                        let total = n_node_samples[right_idx];
                        if total > 0.0 {
                            let sum: f64 = (0..max_n_classes)
                                .map(|c| node_values[right_idx * max_n_classes + c])
                                .sum();
                            sum / total
                        } else {
                            0.0
                        }
                    };

                    // SHAP value: if x goes left, contribution = val_left - val_parent
                    // if x goes right, contribution = val_right - val_parent
                    // This is the simplified TreeSHAP (additive approximation)
                    let shap_val = if x_val <= threshold {
                        val_left - val_parent
                    } else {
                        val_right - val_parent
                    };

                    out[i * n_features + feat] = shap_val;
                }
            }
        }

        let rows: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| out[i * n_features..(i + 1) * n_features].to_vec())
            .collect();
        PyArray2::from_vec2(py, &rows).map_err(PyErr::from)
    }

    fn weighted_decision_path<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<PyObject> {
        let inner = self.inner.borrow();
        let shape = x.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let x_readonly = x.readonly();
        let view = x_readonly.as_array();
        let node_count = inner.node_count;

        let mut indptr = Vec::with_capacity(n_samples + 1);
        let mut indices = Vec::with_capacity(n_samples * (1 + inner.max_depth as usize));
        let mut values = Vec::with_capacity(n_samples * (1 + inner.max_depth as usize));
        indptr.push(0isize);
        for i in 0..n_samples {
            let mut p_nsy = 1.0f32;
            let mut parent_tau = 0.0f32;
            let mut curr = inner.root;
            loop {
                let node = &inner.nodes[curr as usize];
                if node.left_child != TREE_LEAF {
                    let delta = node.tau - parent_tau;
                    parent_tau = node.tau;
                    let mut eta = 0f32;
                    for f in 0..n_features {
                        let x_val = view[[i, f]];
                        eta += (x_val - node.upper_bounds[f]).max(0.0)
                            + (node.lower_bounds[f] - x_val).max(0.0);
                    }
                    let p_s = 1.0 - (-(delta * eta) as f64).exp() as f32;
                    if p_s > 0.0 {
                        indices.push(curr as isize);
                        values.push(p_s * p_nsy);
                    }
                    p_nsy *= 1.0 - p_s;
                    if view[[i, node.feature as usize]] as f64 <= node.threshold {
                        curr = node.left_child;
                    } else {
                        curr = node.right_child;
                    }
                } else {
                    indices.push(curr as isize);
                    values.push(p_nsy);
                    break;
                }
            }
            indptr.push(indices.len() as isize);
        }

        let indptr_arr = PyArray1::from_vec(py, indptr);
        let indices_arr = PyArray1::from_vec(py, indices);
        let values_arr = PyArray1::from_vec(py, values);
        build_csr(
            py,
            values_arr.into_any().unbind(),
            indices_arr,
            indptr_arr,
            n_samples,
            node_count,
        )
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let inner = self.inner.borrow();
        let args = PyTuple::new(
            py,
            [
                inner.n_features.into_pyobject(py)?.into_any().unbind(),
                PyArray1::from_vec(
                    py,
                    inner
                        .n_classes
                        .iter()
                        .map(|&v| v as isize)
                        .collect::<Vec<_>>(),
                )
                .into_any()
                .unbind(),
                inner.n_outputs.into_pyobject(py)?.into_any().unbind(),
            ],
        )?
        .into_any()
        .unbind();
        let state = self.__getstate__(py)?;
        reduce_tuple(
            py,
            PyType::new::<PyTree>(py).into_any().unbind(),
            args,
            state,
        )
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let inner = self.inner.borrow();
        let n = inner.node_count as usize;
        let d = PyDict::new(py);
        d.set_item("max_depth", inner.max_depth)?;
        d.set_item("node_count", inner.node_count)?;
        d.set_item("root", inner.root)?;
        d.set_item(
            "left_child",
            PyArray1::from_vec(
                py,
                (0..n)
                    .map(|i| inner.nodes[i].left_child as isize)
                    .collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "right_child",
            PyArray1::from_vec(
                py,
                (0..n)
                    .map(|i| inner.nodes[i].right_child as isize)
                    .collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "feature",
            PyArray1::from_vec(
                py,
                (0..n)
                    .map(|i| inner.nodes[i].feature as isize)
                    .collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "threshold",
            PyArray1::from_vec(
                py,
                (0..n).map(|i| inner.nodes[i].threshold).collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "impurity",
            PyArray1::from_vec(
                py,
                (0..n).map(|i| inner.nodes[i].impurity).collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "n_node_samples",
            PyArray1::from_vec(
                py,
                (0..n)
                    .map(|i| inner.nodes[i].n_node_samples as isize)
                    .collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "weighted_n_node_samples",
            PyArray1::from_vec(
                py,
                (0..n)
                    .map(|i| inner.nodes[i].weighted_n_node_samples)
                    .collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "tau",
            PyArray1::from_vec(py, (0..n).map(|i| inner.nodes[i].tau).collect::<Vec<_>>()),
        )?;
        d.set_item(
            "variance",
            PyArray1::from_vec(
                py,
                (0..n).map(|i| inner.nodes[i].variance).collect::<Vec<_>>(),
            ),
        )?;
        d.set_item(
            "lower_bounds",
            PyArray2::from_vec2(
                py,
                &(0..n)
                    .map(|i| inner.nodes[i].lower_bounds.clone())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        d.set_item(
            "upper_bounds",
            PyArray2::from_vec2(
                py,
                &(0..n)
                    .map(|i| inner.nodes[i].upper_bounds.clone())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        let total = n * inner.value_stride;
        d.set_item(
            "values",
            PyArray1::from_vec(py, inner.value[..total].to_vec()),
        )?;
        Ok(d.into_any().unbind())
    }

    fn __setstate__<'py>(&self, py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<()> {
        let max_depth: i64 = get_dict_item!(d, "max_depth").extract()?;
        let node_count: i64 = get_dict_item!(d, "node_count").extract()?;
        let root: i64 = get_dict_item!(d, "root").extract()?;
        let left = get_dict_item!(d, "left_child").extract::<PyReadonlyArray1<isize>>()?;
        let right = get_dict_item!(d, "right_child").extract::<PyReadonlyArray1<isize>>()?;
        let feature = get_dict_item!(d, "feature").extract::<PyReadonlyArray1<isize>>()?;
        let threshold = get_dict_item!(d, "threshold").extract::<PyReadonlyArray1<f64>>()?;
        let impurity = get_dict_item!(d, "impurity").extract::<PyReadonlyArray1<f64>>()?;
        let n_node_samples =
            get_dict_item!(d, "n_node_samples").extract::<PyReadonlyArray1<isize>>()?;
        let weighted =
            get_dict_item!(d, "weighted_n_node_samples").extract::<PyReadonlyArray1<f64>>()?;
        let tau = get_dict_item!(d, "tau").extract::<PyReadonlyArray1<f32>>()?;
        let variance = get_dict_item!(d, "variance").extract::<PyReadonlyArray1<f64>>()?;
        let lower = get_dict_item!(d, "lower_bounds").extract::<PyReadonlyArray2<f32>>()?;
        let upper = get_dict_item!(d, "upper_bounds").extract::<PyReadonlyArray2<f32>>()?;
        let values = get_dict_item!(d, "values").extract::<PyReadonlyArray1<f64>>()?;

        let left = left.as_slice()?.to_vec();
        let right = right.as_slice()?.to_vec();
        let feature = feature.as_slice()?.to_vec();
        let threshold = threshold.as_slice()?.to_vec();
        let impurity = impurity.as_slice()?.to_vec();
        let n_node_samples = n_node_samples.as_slice()?.to_vec();
        let weighted = weighted.as_slice()?.to_vec();
        let tau = tau.as_slice()?.to_vec();
        let variance = variance.as_slice()?.to_vec();
        let values = values.as_slice()?.to_vec();
        let lower: Vec<Vec<f32>> = lower
            .as_array()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        let upper: Vec<Vec<f32>> = upper
            .as_array()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();

        let mut inner = self.inner.borrow_mut();
        inner.max_depth = max_depth;
        inner.node_count = node_count;
        inner.root = root;
        inner.capacity = node_count;
        inner.nodes.clear();
        inner.nodes.reserve(node_count as usize);
        for i in 0..node_count as usize {
            inner.nodes.push(Node {
                left_child: left[i] as i64,
                right_child: right[i] as i64,
                feature: feature[i] as i64,
                threshold: threshold[i],
                impurity: impurity[i],
                n_node_samples: n_node_samples[i] as i64,
                weighted_n_node_samples: weighted[i],
                lower_bounds: lower[i].clone(),
                upper_bounds: upper[i].clone(),
                tau: tau[i],
                variance: variance[i],
            });
        }
        inner.value.clear();
        inner.value.extend_from_slice(&values);
        let _ = py;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (left_child, right_child, feature, threshold, n_node_samples, value, tau, lower_bounds, upper_bounds))]
    fn populate_from_arrays(
        &self,
        left_child: Bound<'_, PyArray1<isize>>,
        right_child: Bound<'_, PyArray1<isize>>,
        feature: Bound<'_, PyArray1<isize>>,
        threshold: Bound<'_, PyArray1<f64>>,
        n_node_samples: Bound<'_, PyArray1<isize>>,
        value: Bound<'_, PyArray1<f64>>,
        tau: Bound<'_, PyArray1<f32>>,
        lower_bounds: Bound<'_, PyArray2<f32>>,
        upper_bounds: Bound<'_, PyArray2<f32>>,
    ) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();

        let n_nodes = left_child.len();

        // Resize the tree to hold all nodes
        if !inner.resize_c(n_nodes as i64) {
            return Err(PyMemoryError::new_err("failed to resize tree"));
        }

        // Read arrays
        let lc = left_child.readonly().as_slice()?.to_vec();
        let rc = right_child.readonly().as_slice()?.to_vec();
        let feat = feature.readonly().as_slice()?.to_vec();
        let thresh = threshold.readonly().as_slice()?.to_vec();
        let samples = n_node_samples.readonly().as_slice()?.to_vec();
        let val = value.readonly().as_slice()?.to_vec();
        let tau_arr = tau.readonly().as_slice()?.to_vec();
        let lb_readonly = lower_bounds.readonly();
        let ub_readonly = upper_bounds.readonly();
        let lb_arr = lb_readonly.as_array();
        let ub_arr = ub_readonly.as_array();
        let n_features = inner.n_features;
        let lb_rows = lb_arr.shape()[0];
        let lb_cols = lb_arr.shape()[1];
        // Copy 2D arrays into owned vectors to avoid lifetime issues
        let lb: Vec<Vec<f32>> = (0..lb_rows)
            .map(|i| (0..lb_cols).map(|j| lb_arr[[i, j]]).collect())
            .collect();
        let ub: Vec<Vec<f32>> = (0..lb_rows)
            .map(|i| (0..lb_cols).map(|j| ub_arr[[i, j]]).collect())
            .collect();

        // Set root
        inner.root = 0;
        inner.node_count = n_nodes as i64;

        // Populate nodes
        for i in 0..n_nodes {
            let lb_vec: Vec<f32> = if i < lb_rows {
                (0..n_features)
                    .map(|f| if f < lb_cols { lb[i][f] } else { 0.0 })
                    .collect()
            } else {
                vec![0.0; n_features]
            };
            let ub_vec: Vec<f32> = if i < lb_rows {
                (0..n_features)
                    .map(|f| if f < lb_cols { ub[i][f] } else { 0.0 })
                    .collect()
            } else {
                vec![0.0; n_features]
            };
            let val_offset = i * inner.value_stride;
            let val_end = val_offset + inner.value_stride;
            let node = Node {
                left_child: lc[i] as i64,
                right_child: rc[i] as i64,
                feature: feat[i] as i64,
                threshold: thresh[i],
                tau: tau_arr[i],
                n_node_samples: samples[i] as i64,
                weighted_n_node_samples: samples[i] as f64,
                impurity: 0.0,
                variance: 0.0,
                lower_bounds: lb_vec,
                upper_bounds: ub_vec,
            };

            // Value is stored in a flat array in inner.value
            if val_end <= val.len() {
                inner.value[val_offset..val_end].copy_from_slice(&val[val_offset..val_end]);
            }

            inner.nodes[i] = node;
        }

        Ok(())
    }
}

// =============================================================================
// Tree builders
// =============================================================================

fn build_csr<'py>(
    py: Python<'py>,
    data: PyObject,
    indices: Bound<'py, PyArray1<isize>>,
    indptr: Bound<'py, PyArray1<isize>>,
    n_samples: usize,
    node_count: i64,
) -> PyResult<PyObject> {
    let scipy = py.import("scipy.sparse")?;
    let shape = PyTuple::new(py, [n_samples, node_count as usize])?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    let args = PyTuple::new(
        py,
        [
            data,
            indices.into_any().unbind(),
            indptr.into_any().unbind(),
        ],
    )?;
    let csr = scipy.getattr("csr_matrix")?.call((args,), Some(&kwargs))?;
    Ok(csr.unbind())
}

#[pyclass(unsendable, module = "shinrin._native", name = "DepthFirstTreeBuilder")]
struct PyDepthFirstTreeBuilder {
    splitter: Py<PyAny>,
    min_samples_split: i64,
    max_depth: i64,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyDepthFirstTreeBuilder {
    #[new]
    fn new(splitter: Py<PyAny>, min_samples_split: i64, max_depth: i64) -> Self {
        PyDepthFirstTreeBuilder {
            splitter,
            min_samples_split,
            max_depth,
        }
    }

    #[pyo3(signature = (tree, X, y, sample_weight=None, X_idx_sorted=None))]
    fn build<'py>(
        &self,
        py: Python<'py>,
        tree: Bound<'py, PyTree>,
        X: Bound<'py, PyArray2<f32>>,
        y: Bound<'py, PyArray2<f64>>,
        sample_weight: Option<Bound<'py, PyArray1<f64>>>,
        X_idx_sorted: Option<PyObject>,
    ) -> PyResult<()> {
        let _ = X_idx_sorted;
        let shape = X.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let y_shape = y.shape();
        let y_n = y_shape[0];
        let y_stride = y_shape[1];
        if y_n != n_samples {
            return Err(PyValueError::new_err(format!(
                "Number of labels={} does not match number of samples={}",
                y_n, n_samples
            )));
        }
        let x_readonly = X.readonly();
        let x_view = x_readonly.as_array();
        let mut x = vec![0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                x[i * n_features + j] = x_view[[i, j]];
            }
        }
        let y_readonly = y.readonly();
        let y_view = y_readonly.as_array();
        let mut y_flat = vec![0f64; y_n * y_stride];
        for i in 0..y_n {
            for k in 0..y_stride {
                y_flat[i * y_stride + k] = y_view[[i, k]];
            }
        }
        let sw: Option<Vec<f64>> = match &sample_weight {
            Some(arr) => Some(arr.readonly().as_slice()?.to_vec()),
            None => None,
        };

        let splitter_obj = self.splitter.bind(py);
        let splitter = splitter_obj.downcast::<PySplitter>()?;
        let splitter_ref = splitter.borrow();
        let criterion_obj = splitter_ref.criterion.bind(py);
        let criterion = criterion_obj.downcast::<PyCriterion>()?;

        let rand_r_state: u32 = splitter_ref
            .random_state
            .bind(py)
            .call_method1("randint", (0, RAND_R_MAX))?
            .extract()?;

        let mut splitter_inner = splitter_ref.inner.borrow_mut();
        splitter_inner.init(x, n_samples, n_features, y_flat, y_stride, sw, rand_r_state);

        let tree_inner = tree.borrow_mut();
        {
            let tstate = tree_inner.inner.borrow();
            let init_capacity = if tstate.max_depth <= 10 {
                2i64.pow((tstate.max_depth + 1) as u32) - 1
            } else {
                2047
            };
            drop(tstate);
            tree_inner.inner.borrow_mut().resize_c(init_capacity);
        }

        let n_node_samples_total = splitter_inner.n_samples;
        let max_depth = self.max_depth;
        let min_samples_split = self.min_samples_split;

        #[derive(Clone, Copy)]
        struct StackRecord {
            start: usize,
            end: usize,
            depth: i64,
            parent: i64,
            is_left: bool,
            impurity: f64,
            n_constant_features: i64,
        }

        let mut stack: Vec<StackRecord> = Vec::with_capacity(10);
        stack.push(StackRecord {
            start: 0,
            end: n_node_samples_total,
            depth: 0,
            parent: TREE_UNDEFINED,
            is_left: false,
            impurity: f64::INFINITY,
            n_constant_features: 0,
        });

        let mut first = true;
        let mut max_depth_seen: i64 = -1;

        while let Some(rec) = stack.pop() {
            let start = rec.start;
            let end = rec.end;
            let depth = rec.depth;
            let parent = rec.parent;
            let is_left = rec.is_left;
            let mut impurity = rec.impurity;
            let n_constant_features = rec.n_constant_features;
            let n_node_samples = (end - start) as i64;

            let weighted_n_node_samples =
                splitter_inner.node_reset(start, end, &criterion.borrow());

            if first {
                impurity = splitter_inner.node_impurity(&criterion.borrow());
                first = false;
            }

            let mut is_leaf = depth >= max_depth || n_node_samples < min_samples_split;
            let mut split = SplitRecord::default();
            if !is_leaf {
                split = splitter_inner.node_split(&criterion.borrow());
                is_leaf = split.pos >= end;
            } else {
                splitter_inner.set_bounds();
            }
            is_leaf = is_leaf || splitter_inner.is_pure(&criterion.borrow());

            let node_id = {
                let mut t = tree_inner.inner.borrow_mut();
                t.add_node(
                    parent,
                    is_left,
                    is_leaf,
                    split.feature,
                    split.threshold,
                    impurity,
                    n_node_samples,
                    weighted_n_node_samples,
                    splitter_inner.lower_bounds.clone(),
                    splitter_inner.upper_bounds.clone(),
                    split.e,
                )
            };
            if node_id == TREE_LEAF {
                return Err(PyMemoryError::new_err("failed to allocate node"));
            }

            {
                let mut t = tree_inner.inner.borrow_mut();
                let offset = node_id as usize * t.value_stride;
                let n = t.value_stride;
                let mut dest = vec![0.0; n];
                splitter_inner.node_value_into(&criterion.borrow(), &mut dest);
                t.value[offset..offset + n].copy_from_slice(&dest);
            }

            if !is_leaf {
                stack.push(StackRecord {
                    start: split.pos,
                    end,
                    depth: depth + 1,
                    parent: node_id,
                    is_left: false,
                    impurity: split.impurity_right,
                    n_constant_features,
                });
                stack.push(StackRecord {
                    start,
                    end: split.pos,
                    depth: depth + 1,
                    parent: node_id,
                    is_left: true,
                    impurity: split.impurity_left,
                    n_constant_features,
                });
            }

            if depth > max_depth_seen {
                max_depth_seen = depth;
            }
        }

        {
            let mut t = tree_inner.inner.borrow_mut();
            let node_count = t.node_count;
            t.resize_c(node_count);
            t.max_depth = max_depth_seen;
        }
        Ok(())
    }
}

#[pyclass(unsendable, module = "shinrin._native", name = "PartialFitTreeBuilder")]
struct PyPartialFitTreeBuilder {
    min_samples_split: i64,
    max_depth: i64,
    random_state: Py<PyAny>,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyPartialFitTreeBuilder {
    #[new]
    fn new(min_samples_split: i64, max_depth: i64, random_state: Py<PyAny>) -> Self {
        PyPartialFitTreeBuilder {
            min_samples_split,
            max_depth,
            random_state,
        }
    }

    #[pyo3(signature = (tree, X, y, sample_weight=None, X_idx_sorted=None))]
    fn build<'py>(
        &self,
        py: Python<'py>,
        tree: Bound<'py, PyTree>,
        X: Bound<'py, PyArray2<f32>>,
        y: Bound<'py, PyArray2<f64>>,
        sample_weight: Option<Bound<'py, PyArray1<f64>>>,
        X_idx_sorted: Option<PyObject>,
    ) -> PyResult<()> {
        let _ = sample_weight;
        let _ = X_idx_sorted;
        let rand_r_state: u32 = self
            .random_state
            .bind(py)
            .call_method1("randint", (0, RAND_R_MAX))?
            .extract()?;
        let shape = X.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let y_shape = y.shape();
        let y_stride = y_shape[1];
        let x_readonly = X.readonly();
        let x_view = x_readonly.as_array();
        let mut x = vec![0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                x[i * n_features + j] = x_view[[i, j]];
            }
        }
        let y_readonly = y.readonly();
        let y_view = y_readonly.as_array();
        let mut y_flat = vec![0f64; n_samples * y_stride];
        for i in 0..n_samples {
            for k in 0..y_stride {
                y_flat[i * y_stride + k] = y_view[[i, k]];
            }
        }

        let tree_inner = tree.borrow_mut();
        {
            let mut t = tree_inner.inner.borrow_mut();
            if t.max_depth <= 10 {
                let init_capacity = 2i64.pow((t.max_depth + 1) as u32) - 1;
                t.resize_c(init_capacity);
            }
            let (max_depth, min_samples_split) = (self.max_depth, self.min_samples_split);
            let node_count = t.node_count;
            drop(t);
            if node_count == 0 {
                tree_inner
                    .inner
                    .borrow_mut()
                    .init(&x, n_features, &y_flat, y_stride);
            }
            let start = if node_count == 0 { 1 } else { 0 };
            for sample_ind in start..n_samples {
                tree_inner.inner.borrow_mut().extend(
                    &x,
                    sample_ind * n_features,
                    n_features,
                    &y_flat,
                    sample_ind * y_stride,
                    y_stride,
                    rand_r_state,
                    min_samples_split,
                )?;
            }
            let _ = max_depth;
        }
        Ok(())
    }
}

// =============================================================================
// Module
// =============================================================================

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCriterion>()?;
    m.add_class::<PyMSE>()?;
    m.add_class::<PyClassificationCriterion>()?;
    m.add_class::<PySplitter>()?;
    m.add_class::<PyBaseDenseSplitter>()?;
    m.add_class::<PyMondrianSplitter>()?;
    m.add_class::<PyTree>()?;
    m.add_class::<PyDepthFirstTreeBuilder>()?;
    m.add_class::<PyPartialFitTreeBuilder>()?;
    m.add("DTYPE", numpy::dtype::<f32>(m.py()))?;
    m.add("DOUBLE", numpy::dtype::<f64>(m.py()))?;
    Ok(())
}
