#![feature(portable_simd)]

mod build;
pub mod error;
mod kernel;
mod layout;
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

fn as_f64_array<'py>(obj: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, f64>> {
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

fn row_major(view: &ArrayViewD<'_, f64>) -> Vec<f64> {
    match view.as_slice() {
        Some(slice) => slice.to_vec(),
        None => view.iter().copied().collect(),
    }
}

struct Points {
    values: Vec<f64>,
    n_rows: usize,
    ndim: usize,
    single: bool,
}

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
        // The copy in `as_points` is what lets the build run detached from the GIL.
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
