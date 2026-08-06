from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from kdtree import KDTree
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree as SciPyKDTree


class Problem(NamedTuple):
    data: NDArray[np.float64]
    queries: NDArray[np.float64]
    k: int
    p: float
    max_distance: float | None
    leafsize: int
    parallel: bool


class Case(NamedTuple):
    data: ArrayLike
    queries: ArrayLike
    k: int = 1
    p: float = 2.0
    max_distance: float | None = None
    leafsize: int = 32
    parallel: bool = False


@st.composite
def problems(draw):
    dims = draw(st.sampled_from((1, 2, 3, 8, 16)))
    n_points = draw(st.integers(1, 256))
    n_queries = draw(st.integers(1, 32))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))

    queries = rng.normal(size=(n_queries, dims))
    if draw(st.booleans()):
        queries = queries[0]

    return Problem(
        data=rng.normal(size=(n_points, dims)),
        queries=queries,
        k=draw(st.integers(1, 16)),
        p=draw(st.sampled_from((1.0, 2.0, 3.0, np.inf))),
        max_distance=draw(st.one_of(st.none(), st.floats(0.0, 4.0))),
        leafsize=draw(st.sampled_from((1, 8, 16, 32, 64))),
        parallel=draw(st.booleans()),
    )


def check_query(
    data: ArrayLike,
    queries: ArrayLike,
    k: int,
    p: float,
    max_distance: float | None,
    leafsize: int,
    parallel: bool,
) -> None:
    actual_tree = KDTree(data, leafsize=leafsize)
    actual = actual_tree.query(
        queries,
        k=k,
        p=p,
        max_distance=max_distance,
        parallel=parallel,
    )

    expected_tree = SciPyKDTree(data, leafsize=leafsize)
    expected = expected_tree.query(
        queries,
        k=k,
        p=p,
        distance_upper_bound=np.inf if max_distance is None else max_distance,
        workers=-1 if parallel else 1,
    )

    data_array = np.asarray(data, dtype=np.float64)
    assert actual_tree.ndim == data_array.shape[1]
    assert actual_tree.n_points == data_array.shape[0]
    assert actual_tree.leafsize == leafsize
    assert len(actual_tree) == data_array.shape[0]
    np.testing.assert_array_equal(actual_tree.data, data_array)

    expected_distances = np.asarray(expected[0]).reshape(actual[0].shape)
    expected_indices = np.asarray(expected[1]).reshape(actual[1].shape)
    np.testing.assert_allclose(actual[0], expected_distances, atol=1e-12)
    np.testing.assert_array_equal(actual[1], expected_indices)


@settings(max_examples=100, deadline=None)
@given(problem=problems())
def test_compatibility(problem: Problem) -> None:
    check_query(*problem)


PARALLEL_RNG = np.random.default_rng(42)
PARALLEL_DATA = PARALLEL_RNG.normal(size=(2_000, 8))
PARALLEL_QUERIES = PARALLEL_RNG.normal(size=(256, 8))

VALID: dict[str, Case] = {
    "array-like": Case(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [0.2, 0.0],
        k=2,
    ),
    "single-query": Case(
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
        [0.2, 0.0],
        k=2,
    ),
    "batch-query": Case(
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
        [[0.2, 0.0], [3.8, 0.0]],
        k=2,
    ),
    "max-distance-padding": Case(
        [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
        [0.0, 0.0],
        k=3,
        max_distance=1.0,
    ),
    "infinite-norm": Case(
        [[0.0, 0.0], [2.0, 1.0], [1.0, 2.0]],
        [[0.1, 0.8]],
        k=3,
        p=np.inf,
        leafsize=1,
    ),
    "k-greater-than-points": Case(
        [[0.0], [2.0]],
        [[0.5]],
        k=4,
    ),
    "large-parallel-batch": Case(
        PARALLEL_DATA,
        PARALLEL_QUERIES,
        k=4,
        parallel=True,
    ),
}

INVALID: dict[str, Case] = {
    "query-wrong-dimensions": Case([[0.0, 0.0], [1.0, 0.0]], [0.0]),
    "zero-k": Case([[0.0, 0.0], [1.0, 0.0]], [0.0, 0.0], k=0),
}


@pytest.mark.parametrize("case", VALID.values(), ids=VALID.keys())
def test_valid(case: Case) -> None:
    check_query(*case)


@pytest.mark.parametrize("case", INVALID.values(), ids=INVALID.keys())
def test_invalid(case: Case) -> None:
    tree = KDTree(case.data, leafsize=case.leafsize)
    with pytest.raises(ValueError):
        tree.query(
            case.queries,
            k=case.k,
            p=case.p,
            max_distance=case.max_distance,
            parallel=case.parallel,
        )


def test_threaded_queries_are_safe() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(1_000, 4))
    queries = rng.normal(size=(32, 4))
    actual_tree = KDTree(data)
    expected_tree = SciPyKDTree(data)

    def run(offset: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        return actual_tree.query(queries[offset : offset + 8], k=3, parallel=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(run, range(0, 32, 8)))

    for offset, result in zip(range(0, 32, 8), actual, strict=True):
        expected = expected_tree.query(queries[offset : offset + 8], k=3, workers=1)
        np.testing.assert_allclose(result[0], expected[0], atol=1e-12)
        np.testing.assert_array_equal(result[1], expected[1])
