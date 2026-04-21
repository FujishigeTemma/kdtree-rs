from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from kdtree import KDTree
from scipy.spatial import KDTree as SciPyKDTree
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Case:
    name: str
    dims: int
    n_points: int
    n_queries: int
    leafsize: int
    radius: float
    pair_points: int


def time_call(fn: Callable[[], object], *, repeat: int, warmup: int) -> float:
    samples: list[float] = []
    for _ in range(warmup):
        fn()
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000_000)
    return statistics.median(samples)


def format_ms(value: float) -> str:
    return f"{value:8.3f}"


def run_case(case: Case, *, repeat: int, warmup: int) -> dict[str, float]:
    rng = np.random.default_rng(20260421 + case.dims + case.n_points)
    data = rng.uniform(size=(case.n_points, case.dims))
    queries = rng.uniform(size=(case.n_queries, case.dims))
    pair_data = data[: case.pair_points]

    scipy_tree = SciPyKDTree(data, leafsize=case.leafsize)
    scipy_ckdtree = cKDTree(data, leafsize=case.leafsize)
    rust_tree = KDTree(data, leafsize=case.leafsize)
    scipy_tree_pairs = SciPyKDTree(pair_data, leafsize=case.leafsize)
    scipy_ckdtree_pairs = cKDTree(pair_data, leafsize=case.leafsize)
    rust_tree_pairs = KDTree(pair_data, leafsize=case.leafsize)

    return {
        "build_kdtree": time_call(
            lambda: KDTree(data, leafsize=case.leafsize), repeat=repeat, warmup=warmup
        ),
        "build_scipy_kdtree": time_call(
            lambda: SciPyKDTree(data, leafsize=case.leafsize), repeat=repeat, warmup=warmup
        ),
        "build_scipy_ckdtree": time_call(
            lambda: cKDTree(data, leafsize=case.leafsize), repeat=repeat, warmup=warmup
        ),
        "query_kdtree": time_call(
            lambda: rust_tree.query(queries, k=8, p=2.0, parallel=True),
            repeat=repeat,
            warmup=warmup,
        ),
        "query_scipy_kdtree": time_call(
            lambda: scipy_tree.query(queries, k=8, p=2.0),
            repeat=repeat,
            warmup=warmup,
        ),
        "query_scipy_ckdtree": time_call(
            lambda: scipy_ckdtree.query(queries, k=8, p=2.0),
            repeat=repeat,
            warmup=warmup,
        ),
        "radius_kdtree": time_call(
            lambda: rust_tree.query_radius(queries, case.radius, p=2.0, parallel=True),
            repeat=repeat,
            warmup=warmup,
        ),
        "radius_scipy_kdtree": time_call(
            lambda: scipy_tree.query_ball_point(queries, case.radius, p=2.0),
            repeat=repeat,
            warmup=warmup,
        ),
        "radius_scipy_ckdtree": time_call(
            lambda: scipy_ckdtree.query_ball_point(queries, case.radius, p=2.0),
            repeat=repeat,
            warmup=warmup,
        ),
        "pairs_kdtree": time_call(
            lambda: rust_tree_pairs.query_pairs(case.radius, p=2.0),
            repeat=repeat,
            warmup=warmup,
        ),
        "pairs_scipy_kdtree": time_call(
            lambda: np.asarray(sorted(scipy_tree_pairs.query_pairs(case.radius, p=2.0))),
            repeat=repeat,
            warmup=warmup,
        ),
        "pairs_scipy_ckdtree": time_call(
            lambda: np.asarray(sorted(scipy_ckdtree_pairs.query_pairs(case.radius, p=2.0))),
            repeat=repeat,
            warmup=warmup,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    cases = [
        Case(
            "small-3d",
            dims=3,
            n_points=10_000,
            n_queries=1_000,
            leafsize=32,
            radius=0.050,
            pair_points=5_000,
        ),
        Case(
            "medium-8d",
            dims=8,
            n_points=10_000,
            n_queries=1_000,
            leafsize=32,
            radius=0.300,
            pair_points=4_000,
        ),
        Case(
            "large-16d",
            dims=16,
            n_points=20_000,
            n_queries=2_000,
            leafsize=32,
            radius=0.750,
            pair_points=3_000,
        ),
    ]

    for case in cases:
        metrics = run_case(case, repeat=args.repeat, warmup=args.warmup)
        print(
            f"\n[{case.name}] dims={case.dims} n_points={case.n_points} "
            f"n_queries={case.n_queries} leafsize={case.leafsize} radius={case.radius} "
            f"pair_points={case.pair_points}"
        )
        print("operation             kdtree_ms  scipy_kdtree_ms  scipy_ckdtree_ms")
        print(
            f"build                {format_ms(metrics['build_kdtree'])}"
            f"  {format_ms(metrics['build_scipy_kdtree'])}"
            f"  {format_ms(metrics['build_scipy_ckdtree'])}"
        )
        print(
            f"query                {format_ms(metrics['query_kdtree'])}"
            f"  {format_ms(metrics['query_scipy_kdtree'])}"
            f"  {format_ms(metrics['query_scipy_ckdtree'])}"
        )
        print(
            f"query_radius         {format_ms(metrics['radius_kdtree'])}"
            f"  {format_ms(metrics['radius_scipy_kdtree'])}"
            f"  {format_ms(metrics['radius_scipy_ckdtree'])}"
        )
        print(
            f"query_pairs          {format_ms(metrics['pairs_kdtree'])}"
            f"  {format_ms(metrics['pairs_scipy_kdtree'])}"
            f"  {format_ms(metrics['pairs_scipy_ckdtree'])}"
        )


if __name__ == "__main__":
    main()
