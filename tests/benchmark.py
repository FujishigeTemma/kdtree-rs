"""Benchmarks matching the in-scope workloads from SciPy's spatial suite."""

from __future__ import annotations

import sys
from typing import Final, Literal

import numpy as np
import pytest
from kdtree import KDTree
from scipy.spatial import KDTree as SciPyKDTree

Distribution = Literal["bimodal", "uniform", "sorted"]
Implementation = Literal["kdtree", "scipy"]

SIZES = ((3, 10_000, 1_000), (8, 10_000, 1_000), (16, 10_000, 1_000))
IMPLEMENTATIONS: Final = ("kdtree", "scipy")

DATA_SEED: Final = 123_456_789
QUERY_SEED: Final = 987_654_321


def make_data(distribution: Distribution, n_points: int, dims: int) -> np.ndarray:
    rng = np.random.default_rng(DATA_SEED)
    if distribution == "bimodal":
        return np.concatenate(
            (
                rng.standard_normal((n_points // 2, dims)),
                rng.standard_normal((n_points - n_points // 2, dims)) + np.ones(dims),
            )
        )
    if distribution == "uniform":
        return rng.uniform(size=(n_points, dims))
    if distribution == "sorted":
        return np.repeat(np.arange(n_points, 0, -1)[:, None], dims, axis=1) / n_points
    raise ValueError(distribution)


# Builds are serial only — SciPy's KDTree cannot build in parallel, so there
# would be nothing to compare against. The literal `-serial` suffix keeps the
# ids aligned with `benches/grid.rs`.
BUILD_PARAMS = [
    pytest.param(
        dims,
        n_points,
        "bimodal",
        id=f"build-d{dims}-n{n_points}-serial",
        marks=pytest.mark.benchmark(group=f"build-d{dims}-n{n_points}"),
    )
    for dims, n_points, _ in SIZES
] + [
    pytest.param(
        dims,
        n_points,
        distribution,
        id=f"build-unbalanced-d{dims}-n{n_points}-{distribution}-serial",
        marks=pytest.mark.benchmark(group=f"build-unbalanced-d{dims}-n{n_points}-{distribution}"),
    )
    for dims, n_points, _ in SIZES
    for distribution in ("uniform", "sorted")
]

QUERY_PARAMS = [
    pytest.param(
        dims,
        n_points,
        n_queries,
        distribution,
        16,
        2.0,
        parallel,
        id=(
            f"query-unbalanced-d{dims}-n{n_points}-q{n_queries}-{distribution}"
            f"-{'parallel' if parallel else 'serial'}"
        ),
        marks=pytest.mark.benchmark(
            group=f"query-unbalanced-d{dims}-n{n_points}-q{n_queries}-{distribution}"
        ),
    )
    for dims, n_points, n_queries in SIZES
    for distribution in ("uniform", "sorted")
    for parallel in (False, True)
] + [
    pytest.param(
        dims,
        n_points,
        n_queries,
        "uniform",
        leafsize,
        p,
        parallel,
        id=(
            f"query-d{dims}-n{n_points}-q{n_queries}-leaf{leafsize}"
            f"-p{'inf' if np.isinf(p) else f'{p:g}'}"
            f"-{'parallel' if parallel else 'serial'}"
        ),
        marks=pytest.mark.benchmark(
            group=(
                f"query-d{dims}-n{n_points}-q{n_queries}-leaf{leafsize}"
                f"-p{'inf' if np.isinf(p) else f'{p:g}'}"
            )
        ),
    )
    for dims, n_points, n_queries in SIZES
    for p in (1.0, 2.0, float("inf"))
    for leafsize in (8, 128)
    for parallel in (False, True)
]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("dims", "n_points", "distribution"),
    BUILD_PARAMS,
)
def test_build(
    benchmark,
    dims: int,
    n_points: int,
    distribution: Distribution,
    implementation: Implementation,
) -> None:
    data = make_data(distribution, n_points, dims)
    if implementation == "kdtree":
        benchmark(KDTree, data, leafsize=16)
    else:
        benchmark(SciPyKDTree, data, leafsize=16)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("dims", "n_points", "n_queries", "distribution", "leafsize", "p", "parallel"),
    QUERY_PARAMS,
)
def test_query(
    benchmark,
    dims: int,
    n_points: int,
    n_queries: int,
    distribution: Distribution,
    leafsize: int,
    p: float,
    parallel: bool,
    implementation: Implementation,
) -> None:
    data = make_data(distribution, n_points, dims)
    queries = np.random.default_rng(QUERY_SEED).uniform(size=(n_queries, dims))
    if implementation == "kdtree":
        tree = KDTree(data, leafsize=leafsize)
        benchmark(tree.query, queries, k=1, p=p, parallel=parallel)
    else:
        tree = SciPyKDTree(data, leafsize=leafsize)
        benchmark(tree.query, queries, k=1, p=p, workers=-1 if parallel else 1)


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "--benchmark-columns=min,median,mean,stddev,ops", *sys.argv[1:]])
    )
