//! A minimal, high-performance immutable KDTree for k-nearest-neighbor
//! queries, exposed to free-threaded CPython as `kdtree._core`.
//!
//! # Module map
//!
//! The crate is layered so each module owns exactly one concern:
//!
//! - `metric.rs` — the `L^p` algebra over *reduced distances*, the domain
//!   every distance in the crate is carried in.
//! - `simd.rs` — architecture-neutral SIMD primitives (lane width, lane-wise
//!   min/max, horizontal reductions).
//! - `kernel.rs` — every bulk loop over point rows: bounding boxes for the
//!   build, and the reduced-distance kernels for queries.
//! - `tree.rs` — the storage layout (`Tree`, `Node`) and read access.
//! - `build.rs` — the only writer of that layout.
//! - `query.rs` — the branch-and-bound descent and the k-best set.
//! - this file — the Python boundary: argument coercion in, ndarray out.
#![feature(portable_simd)]

mod build;
pub mod error;
mod kernel;
mod metric;
mod query;
mod simd;
pub mod tree;

use ndarray::{Array2, ArrayViewD};
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArrayDyn, PyUntypedArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::error::KDTreeError;
use crate::tree::Tree;

fn kd_error(err: KDTreeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// View an ndarray as `f64`, promoting other dtypes.
///
/// The API takes `numpy.ndarray` and nothing else, so this never has to build
/// an array out of a sequence: the non-`f64` case is one `astype` on an array
/// that already exists, and anything that is not an array is a `TypeError`
/// rather than a silent (and, for a ragged list, surprising) conversion.
fn as_f64_array<'py>(obj: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, f64>> {
    // Fast path: already f64, so extract exactly once and copy nothing.
    if let Ok(readonly) = obj.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        return Ok(readonly);
    }
    let array = obj.cast::<PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err(format!(
            "expected a numpy.ndarray, got {}",
            obj.get_type()
                .name()
                .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string())
        ))
    })?;
    Ok(array.call_method1("astype", ("float64",))?.extract()?)
}

/// Copy a view into one row-major `Vec` — the layout the core expects — with
/// a straight memcpy whenever the input is already contiguous.
fn row_major(view: &ArrayViewD<'_, f64>) -> Vec<f64> {
    match view.as_slice() {
        Some(slice) => slice.to_vec(),
        None => view.iter().copied().collect(),
    }
}

/// A `(n_rows, ndim)` point matrix normalized out of Python. `single` records
/// that the caller passed one bare point, so `query` can mirror that shape
/// back in its result.
struct Points {
    values: Vec<f64>,
    n_rows: usize,
    ndim: usize,
    single: bool,
}

/// Normalize an argument of `what` into a point matrix, accepting a single
/// `(ndim,)` point only when `allow_single`.
fn as_points(obj: &Bound<'_, PyAny>, what: &'static str, allow_single: bool) -> PyResult<Points> {
    let readonly = as_f64_array(obj)?;
    let view = readonly.as_array();
    let shape_err = || kd_error(KDTreeError::InvalidShape(what));
    match view.ndim() {
        1 if allow_single => Ok(Points {
            n_rows: 1,
            ndim: view.len(),
            values: row_major(&view),
            single: true,
        }),
        2 => Ok(Points {
            n_rows: view.shape()[0],
            ndim: view.shape()[1],
            values: row_major(&view),
            single: false,
        }),
        _ => Err(shape_err()),
    }
}

impl Points {
    /// Require rows of exactly `ndim` coordinates.
    fn require_ndim(self, ndim: usize) -> PyResult<Self> {
        if self.ndim != ndim {
            return Err(kd_error(KDTreeError::DimensionMismatch {
                expected: ndim,
                got: self.ndim,
            }));
        }
        Ok(self)
    }
}

/// Box one result buffer as a numpy array of `shape`, or as a bare 1-D
/// vector when `shape` is `None` (the single-point query result).
fn to_numpy<T: Element>(
    py: Python<'_>,
    values: Vec<T>,
    shape: Option<(usize, usize)>,
) -> Py<PyAny> {
    match shape {
        None => PyArray1::from_vec(py, values).into_any().unbind(),
        Some(shape) => {
            let array = Array2::from_shape_vec(shape, values).expect("shape should match");
            PyArray2::from_owned_array(py, array).into_any().unbind()
        }
    }
}

#[pyclass(module = "kdtree._core", frozen)]
struct KDTree {
    tree: Tree,
}

#[pymethods]
impl KDTree {
    #[new]
    #[pyo3(signature = (data, *, leafsize = 32, parallel = false))]
    fn new(
        py: Python<'_>,
        data: Bound<'_, PyAny>,
        leafsize: usize,
        parallel: bool,
    ) -> PyResult<Self> {
        let points = as_points(&data, "data must be a two-dimensional array", false)?;
        // The borrow of the caller's array ended with `as_points`, so the
        // build below owns its data and can run detached from the GIL.
        let tree = py
            .detach(|| Tree::new(points.values, points.ndim, leafsize, parallel))
            .map_err(kd_error)?;
        Ok(Self { tree })
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.tree.ndim()
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.tree.n_points()
    }

    #[getter]
    fn leafsize(&self) -> usize {
        self.tree.leafsize()
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let array = Array2::from_shape_vec(
            (self.tree.n_points(), self.tree.ndim()),
            self.tree.original_data(),
        )
        .expect("tree data should be rectangular");
        PyArray2::from_owned_array(py, array)
    }

    fn __len__(&self) -> usize {
        self.tree.n_points()
    }

    fn __repr__(&self) -> String {
        format!(
            "KDTree(n_points={}, ndim={}, leafsize={})",
            self.tree.n_points(),
            self.tree.ndim(),
            self.tree.leafsize()
        )
    }

    #[pyo3(signature = (x, *, k = 1, p = 2.0, max_distance = None, eps = 0.0, parallel = false))]
    #[allow(clippy::too_many_arguments)] // mirrors the Python keyword API
    fn query<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        k: usize,
        p: f64,
        max_distance: Option<f64>,
        eps: f64,
        parallel: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let queries = as_points(&x, "query must be one- or two-dimensional", true)?
            .require_ndim(self.tree.ndim())?;
        let tree = &self.tree;
        let (distances, indices) = py
            .detach(|| tree.query(&queries.values, k, p, max_distance, eps, parallel))
            .map_err(kd_error)?;

        // A single-point query answers with bare `(k,)` vectors; a batch
        // answers with `(n_queries, k)` matrices.
        let shape = (!queries.single).then_some((queries.n_rows, k));
        Ok((to_numpy(py, distances, shape), to_numpy(py, indices, shape)))
    }
}

#[pymodule(gil_used = false)]
#[pyo3(name = "_core")]
fn kdtree_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KDTree>()?;
    Ok(())
}
