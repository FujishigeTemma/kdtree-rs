from __future__ import annotations

import itertools

import numpy as np
from kdtree import KDTree


def brute_force_query(
    data: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
    p: float,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isinf(p):
        distances = np.max(np.abs(data - query), axis=1)
    else:
        distances = np.sum(np.abs(data - query) ** p, axis=1) ** (1.0 / p)
    order = np.argsort(distances, kind="stable")
    if max_distance is not None:
        order = order[distances[order] <= max_distance]
    order = order[:k]

    out_distances = np.full(k, np.inf, dtype=np.float64)
    out_indices = np.full(k, data.shape[0], dtype=np.int64)
    out_distances[: len(order)] = distances[order]
    out_indices[: len(order)] = order
    return out_distances, out_indices


def brute_force_radius(
    data: np.ndarray,
    query: np.ndarray,
    *,
    radius: float,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isinf(p):
        distances = np.max(np.abs(data - query), axis=1)
    else:
        distances = np.sum(np.abs(data - query) ** p, axis=1) ** (1.0 / p)
    mask = distances <= radius
    indices = np.nonzero(mask)[0]
    return indices.astype(np.int64), distances[indices]


def test_query_matches_bruteforce_across_metrics() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(128, 6))
    queries = rng.normal(size=(16, 6))
    tree = KDTree(data, leafsize=16)

    for p in (1.0, 2.0, np.inf, 3.0):
        distances, indices = tree.query(queries, k=5, p=p)
        for row in range(queries.shape[0]):
            expected_distances, expected_indices = brute_force_query(data, queries[row], k=5, p=p)
            np.testing.assert_allclose(distances[row], expected_distances)
            np.testing.assert_array_equal(indices[row], expected_indices)


def test_query_respects_max_distance() -> None:
    data = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    tree = KDTree(data)

    distances, indices = tree.query([0.0, 0.0], k=3, max_distance=1.0)

    np.testing.assert_allclose(distances, [0.0, np.inf, np.inf])
    np.testing.assert_array_equal(indices, [0, 3, 3])


def test_query_radius_matches_bruteforce() -> None:
    rng = np.random.default_rng(456)
    data = rng.normal(size=(96, 4))
    queries = rng.normal(size=(8, 4))
    tree = KDTree(data, leafsize=8)

    for p in (1.0, 2.0, np.inf, 3.0):
        result = tree.query_radius(queries, 1.75, p=p, return_distance=True, sort=True)
        assert isinstance(result, tuple)
        indices_list, distances_list = result
        for row, (indices, distances) in enumerate(zip(indices_list, distances_list, strict=True)):
            expected_indices, expected_distances = brute_force_radius(
                data,
                queries[row],
                radius=1.75,
                p=p,
            )
            order = np.argsort(expected_distances, kind="stable")
            np.testing.assert_array_equal(indices, expected_indices[order])
            np.testing.assert_allclose(distances, expected_distances[order])


def test_query_pairs_matches_bruteforce() -> None:
    rng = np.random.default_rng(99)
    data = rng.normal(size=(64, 3))
    tree = KDTree(data, leafsize=8)

    pairs = tree.query_pairs(0.5, p=2.0)

    expected = []
    for left, right in itertools.combinations(range(data.shape[0]), 2):
        if np.linalg.norm(data[left] - data[right]) <= 0.5:
            expected.append((left, right))
    expected_pairs = np.asarray(expected, dtype=np.int64).reshape((-1, 2))

    np.testing.assert_array_equal(pairs, expected_pairs)
