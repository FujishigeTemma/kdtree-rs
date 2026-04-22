mod error;
mod metric;
mod node;
mod query;
mod tree;

use ndarray::{Array2, Ix1, Ix2};
use numpy::{PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyModule};

use crate::error::KDTreeError;
use crate::tree::Tree;

fn kd_error(err: KDTreeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn as_numpy_f64<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = [("dtype", numpy.getattr("float64")?)].into_py_dict(py)?;
    numpy.call_method("asarray", (obj,), Some(&kwargs))
}

fn parse_queries<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    expected_ndim: usize,
) -> PyResult<(Vec<f64>, usize, bool)> {
    let array = as_numpy_f64(py, obj)?;
    let readonly = array.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let view = readonly.as_array();
    match view.ndim() {
        1 => {
            let query = view
                .into_dimensionality::<Ix1>()
                .map_err(|_| kd_error(KDTreeError::InvalidShape("query must be one- or two-dimensional")))?;
            if query.len() != expected_ndim {
                return Err(kd_error(KDTreeError::DimensionMismatch {
                    expected: expected_ndim,
                    got: query.len(),
                }));
            }
            Ok((query.to_vec(), 1, true))
        }
        2 => {
            let queries = view
                .into_dimensionality::<Ix2>()
                .map_err(|_| kd_error(KDTreeError::InvalidShape("query must be one- or two-dimensional")))?;
            if queries.shape()[1] != expected_ndim {
                return Err(kd_error(KDTreeError::DimensionMismatch {
                    expected: expected_ndim,
                    got: queries.shape()[1],
                }));
            }
            Ok((queries.iter().copied().collect(), queries.nrows(), false))
        }
        _ => Err(kd_error(KDTreeError::InvalidShape(
            "query must be one- or two-dimensional",
        ))),
    }
}

fn auto_parallel(parallel: Option<bool>, n_queries: usize) -> bool {
    parallel.unwrap_or(n_queries >= 256)
}

#[pyclass(module = "kdtree._core", frozen)]
struct KDTree {
    tree: Tree,
    data: Vec<f64>,
}

#[pymethods]
impl KDTree {
    #[new]
    #[pyo3(signature = (data, *, leafsize = 32, copy_data = false))]
    fn new(
        py: Python<'_>,
        data: Bound<'_, PyAny>,
        leafsize: usize,
        copy_data: bool,
    ) -> PyResult<Self> {
        let _ = copy_data;
        let array = as_numpy_f64(py, &data)?;
        let readonly = array.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let view = readonly.as_array();
        if view.ndim() != 2 {
            return Err(kd_error(KDTreeError::InvalidShape(
                "data must be a two-dimensional array",
            )));
        }
        let matrix = view
            .into_dimensionality::<Ix2>()
            .map_err(|_| kd_error(KDTreeError::InvalidShape("data must be a two-dimensional array")))?;
        let original: Vec<f64> = matrix.iter().copied().collect();
        let tree = Tree::new(matrix, leafsize).map_err(kd_error)?;
        Ok(Self { tree, data: original })
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
        let array =
            Array2::from_shape_vec((self.tree.n_points(), self.tree.ndim()), self.data.clone())
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

    #[pyo3(signature = (x, *, k = 1, p = 2.0, max_distance = None, eps = 0.0, parallel = None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        k: usize,
        p: f64,
        max_distance: Option<f64>,
        eps: f64,
        parallel: Option<bool>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (queries, n_queries, single) = parse_queries(py, &x, self.tree.ndim())?;
        let parallel = auto_parallel(parallel, n_queries);
        let (distances, indices) = self
            .tree
            .query(&queries, k, p, max_distance, eps, parallel)
            .map_err(kd_error)?;

        if single {
            let py_distances = PyArray1::from_vec(py, distances).into_any().unbind();
            let py_indices = PyArray1::from_vec(py, indices.into_iter().map(|i| i as i64).collect())
                .into_any()
                .unbind();
            Ok((py_distances, py_indices))
        } else {
            let py_distances = PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n_queries, k), distances).expect("shape should match"),
            )
            .into_any()
            .unbind();
            let converted = indices.into_iter().map(|i| i as i64).collect::<Vec<_>>();
            let py_indices = PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n_queries, k), converted).expect("shape should match"),
            )
            .into_any()
            .unbind();
            Ok((py_distances, py_indices))
        }
    }
}

#[pymodule(gil_used = false)]
#[pyo3(name = "_core")]
fn kdtree_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KDTree>()?;
    Ok(())
}
