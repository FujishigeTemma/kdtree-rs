from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from kdtree import KDTree


def test_constructor_accepts_array_like() -> None:
    tree = KDTree([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert tree.ndim == 2
    assert tree.n_points == 3
    assert tree.leafsize == 32
    assert len(tree) == 3


def test_query_single_returns_rank_1_arrays() -> None:
    tree = KDTree([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])

    distances, indices = tree.query([0.2, 0.0], k=2)

    assert distances.shape == (2,)
    assert indices.shape == (2,)
    np.testing.assert_allclose(distances, [0.2, 0.8])
    np.testing.assert_array_equal(indices, [0, 1])


def test_query_batch_returns_rank_2_arrays() -> None:
    tree = KDTree([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])

    distances, indices = tree.query([[0.2, 0.0], [3.8, 0.0]], k=2)

    assert distances.shape == (2, 2)
    assert indices.shape == (2, 2)
    np.testing.assert_allclose(distances[0], [0.2, 0.8])
    np.testing.assert_allclose(distances[1], [0.2, 2.8])
    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_array_equal(indices[1], [2, 1])


def test_query_radius_single_returns_index_array() -> None:
    tree = KDTree([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])

    indices = tree.query_radius([0.1, 0.0], 1.1, sort=True)

    np.testing.assert_array_equal(indices, [0, 1])


def test_query_radius_batch_returns_lists_of_arrays() -> None:
    tree = KDTree([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])

    indices = tree.query_radius([[0.1, 0.0], [3.9, 0.0]], 1.1, sort=True)

    assert len(indices) == 2
    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_array_equal(indices[1], [2])


def test_query_pairs_returns_compact_pair_array() -> None:
    tree = KDTree([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0], [2.3, 0.0]])

    pairs = tree.query_pairs(0.6)

    np.testing.assert_array_equal(pairs, [[0, 1], [2, 3]])


def test_invalid_inputs_raise_value_error() -> None:
    tree = KDTree([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError):
        tree.query([0.0], k=1)

    with pytest.raises(ValueError):
        tree.query([0.0, 0.0], k=0)

    with pytest.raises(ValueError):
        tree.query_radius([0.0, 0.0], -1.0)


def test_parallel_query_matches_serial_query() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2_000, 8))
    queries = rng.normal(size=(256, 8))
    tree = KDTree(data)

    serial = tree.query(queries, k=4, parallel=False)
    parallel = tree.query(queries, k=4, parallel=True)

    np.testing.assert_allclose(serial[0], parallel[0])
    np.testing.assert_array_equal(serial[1], parallel[1])


def test_threaded_queries_are_safe() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(1_000, 4))
    queries = rng.normal(size=(64, 4))
    tree = KDTree(data)

    def run(offset: int) -> tuple[np.ndarray, np.ndarray]:
        query = queries[offset : offset + 8]
        return tree.query(query, k=3, parallel=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(0, 32, 8)))

    assert len(results) == 4
    for distances, indices in results:
        assert distances.shape == (8, 3)
        assert indices.shape == (8, 3)
