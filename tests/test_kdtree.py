from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from kdtree import KDTree
from scipy.spatial import KDTree as SciPyKDTree


class Problem(NamedTuple):
    data: np.ndarray
    queries: np.ndarray
    k: int
    p: float
    max_distance: float | None
    leafsize: int
    parallel: bool


class Case(NamedTuple):
    data: np.ndarray
    queries: np.ndarray
    k: int = 1
    p: float = 2.0
    max_distance: float | None = None
    leafsize: int = 32
    parallel: bool = False


@st.composite
def problems(draw):
    dims = draw(st.integers(1, 20))
    n_points = draw(st.integers(1, 10000))
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
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    p: float,
    max_distance: float | None,
    leafsize: int,
    parallel: bool,
) -> None:
    actual_tree = KDTree(data, leafsize=leafsize, parallel=parallel)
    actual = actual_tree.query(queries, k=k, p=p, max_distance=max_distance, parallel=parallel)

    expected_tree = SciPyKDTree(data, leafsize=leafsize)
    expected = expected_tree.query(
        queries,
        k=k,
        p=p,
        distance_upper_bound=max_distance if max_distance is not None else np.inf,
        workers=-1 if parallel else 1,
    )

    assert actual_tree.ndim == data.shape[1]
    assert actual_tree.n_points == data.shape[0]
    assert actual_tree.leafsize == leafsize
    assert len(actual_tree) == data.shape[0]
    np.testing.assert_array_equal(actual_tree.data, data)

    expected_shape = (k,) if queries.ndim == 1 else (queries.shape[0], k)
    assert actual[0].shape == expected_shape
    assert actual[1].shape == expected_shape

    expected_distances = np.asarray(expected[0]).reshape(expected_shape)
    expected_indices = np.asarray(expected[1]).reshape(expected_shape)
    np.testing.assert_allclose(actual[0], expected_distances, atol=1e-12)
    np.testing.assert_array_equal(actual[1], expected_indices)


@settings(max_examples=100, deadline=None)
@given(problem=problems())
def test_compatibility(problem: Problem) -> None:
    check_query(*problem)


RNG = np.random.default_rng(0)
PARALLEL_QUERIES = RNG.normal(size=(256, 8))

VALID: dict[str, Case] = {
    "single-query": Case(
        np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        np.array([0.2, 0.0]),
        k=2,
    ),
    "batch-query": Case(
        np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        np.array([[0.2, 0.0], [3.8, 0.0]]),
        k=2,
    ),
    "max-distance-padding": Case(
        np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]),
        np.array([0.0, 0.0]),
        k=3,
        max_distance=1.0,
    ),
    "infinite-norm": Case(
        np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 2.0]]),
        np.array([[0.1, 0.8]]),
        k=3,
        p=np.inf,
        leafsize=1,
    ),
    "single-query-k1": Case(
        np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        np.array([0.2, 0.0]),
    ),
    "batch-query-k1": Case(
        np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        np.array([[0.2, 0.0], [3.8, 0.0]]),
    ),
    "k-greater-than-points": Case(
        np.array([[0.0], [2.0]]),
        np.array([[0.5]]),
        k=4,
    ),
    "promoted-dtype": Case(
        np.array([[0, 0], [1, 0], [4, 0]], dtype=np.int64),
        np.array([[0.2, 0.0]], dtype=np.float32),
        k=2,
    ),
    # Fortran order and a strided column view both make `as_slice()` fail,
    # which is the only way `row_major`'s element-wise branch is reached.
    "non-contiguous": Case(
        np.asfortranarray([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        np.array([[0.2, 9.0, 0.0], [3.8, 9.0, 0.0]])[:, ::2],
        k=2,
    ),
    "large-parallel-batch": Case(
        RNG.normal(size=(2_000, 8)),
        PARALLEL_QUERIES,
        k=4,
        parallel=True,
    ),
    **{
        f"row-width-{dims:02d}-p{'inf' if np.isinf(p) else int(p)}": Case(
            np.random.default_rng(dims).normal(size=(512, dims)),
            np.random.default_rng(1_000 + dims).normal(size=(64, dims)),
            k=3,
            p=p,
            leafsize=6,
        )
        for dims in range(1, 18)
        for p in (1.0, 2.0, np.inf)
    },
}

INVALID: dict[str, Case] = {
    "query-wrong-dimensions": Case(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.0])),
    "zero-k": Case(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.0, 0.0]), k=0),
}


@pytest.mark.parametrize("case", VALID.values(), ids=VALID.keys())
def test_valid(case: Case) -> None:
    check_query(*case)


@pytest.mark.parametrize("case", INVALID.values(), ids=INVALID.keys())
def test_invalid(case: Case) -> None:
    with pytest.raises(ValueError):
        KDTree(case.data, leafsize=case.leafsize).query(case.queries, k=case.k, p=case.p)
