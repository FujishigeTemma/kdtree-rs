from __future__ import annotations

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


def test_query_matches_bruteforce_on_random_grid() -> None:
    rng = np.random.default_rng(2026_04_22)
    for dims in (2, 3, 8, 16):
        for n_points in (50, 200, 1_000):
            data = rng.uniform(size=(n_points, dims))
            queries = rng.uniform(size=(32, dims))
            tree = KDTree(data, leafsize=8)
            for p in (1.0, 2.0, np.inf, 3.0):
                for k in (1, 4, 16):
                    distances, indices = tree.query(queries, k=k, p=p)
                    for row in range(queries.shape[0]):
                        expected_d, expected_i = brute_force_query(
                            data, queries[row], k=k, p=p
                        )
                        np.testing.assert_allclose(distances[row], expected_d, atol=1e-12)
                        np.testing.assert_array_equal(indices[row], expected_i)
