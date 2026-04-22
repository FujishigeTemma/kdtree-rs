"""Benchmark suite mirroring SciPy's `benchmarks/benchmarks/spatial.py`
in-scope KDTree cases (Build, BuildUnbalanced, QueryUnbalanced, Query).

Out-of-scope SciPy benchmarks deliberately omitted, matching this library's
PLAN.md: query_ball_point, query_pairs, count_neighbors,
sparse_distance_matrix, weighted query, periodic boxsize, balanced_tree
toggle. Parallel timings are added on top of SciPy's spec because this
library targets free-threaded CPython.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from kdtree import KDTree
from scipy.spatial import KDTree as SciPyKDTree
from scipy.spatial import cKDTree

# SciPy's spatial.py uses these for Query / Radius parameter sweeps.
LEAF_SIZES = (8, 128)
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
SIZES = ((3, 10_000, 1_000), (8, 10_000, 1_000), (16, 10_000, 1_000))

Distribution = Literal["bimodal", "uniform", "sorted"]


def make_data(distribution: Distribution, n: int, m: int) -> np.ndarray:
    """Reproduce SciPy's three data patterns. Seeds match `spatial.py`."""
    rng = np.random.default_rng(1234)
    if distribution == "bimodal":
        return np.concatenate(
            (
                rng.standard_normal((n // 2, m)),
                rng.standard_normal((n - n // 2, m)) + np.ones(m),
            )
        )
    if distribution == "uniform":
        return rng.uniform(size=(n, m))
    if distribution == "sorted":
        return np.repeat(np.arange(n, 0, -1)[:, None], m, axis=1) / n
    raise ValueError(distribution)


def make_queries(distribution: Distribution, r: int, m: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    if distribution == "bimodal":
        # SciPy advances the same RNG after generating data; we use a fresh
        # RNG per array because we generate them independently. The data /
        # query distribution shape still matches.
        return np.concatenate(
            (
                rng.standard_normal((r // 2, m)),
                rng.standard_normal((r - r // 2, m)) + np.ones(m),
            )
        )
    return rng.uniform(size=(r, m))


def time_call(fn: Callable[[], object], *, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000_000)
    return statistics.median(samples)


def fmt(value: float) -> str:
    return f"{value:9.3f}"


# ---------------------------------------------------------------------------
# Case definitions (mirroring SciPy classes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildCase:
    """SciPy `Build` (bimodal data, sweep dims, KDTree vs cKDTree)."""

    dims: int
    n_points: int
    n_queries: int  # unused in build, kept for label consistency
    distribution: Distribution = "bimodal"
    leafsize: int = 16  # SciPy default for KDTree/cKDTree

    def label(self) -> str:
        return (
            f"build dims={self.dims} n={self.n_points} "
            f"dist={self.distribution} leafsize={self.leafsize}"
        )


@dataclass(frozen=True)
class UnbalancedBuildCase:
    """SciPy `BuildUnbalanced` (sweep order ∈ {random, sorted})."""

    dims: int
    n_points: int
    n_queries: int
    distribution: Distribution
    leafsize: int = 16

    def label(self) -> str:
        return (
            f"build-unbal dims={self.dims} n={self.n_points} "
            f"order={self.distribution} leafsize={self.leafsize}"
        )


@dataclass(frozen=True)
class UnbalancedQueryCase:
    """SciPy `QueryUnbalanced` (build per order, time default k=1 query)."""

    dims: int
    n_points: int
    n_queries: int
    distribution: Distribution
    leafsize: int = 16
    k: int = 1
    p: float = 2.0

    def label(self) -> str:
        return (
            f"query-unbal dims={self.dims} n={self.n_points} "
            f"r={self.n_queries} order={self.distribution} "
            f"leafsize={self.leafsize}"
        )


@dataclass(frozen=True)
class QueryCase:
    """SciPy `Query` minus the (excluded) `boxsize` axis."""

    dims: int
    n_points: int
    n_queries: int
    p: float
    leafsize: int
    k: int = 1
    distribution: Distribution = "uniform"

    def label(self) -> str:
        p_label = "inf" if self.p == float("inf") else f"{self.p:g}"
        return (
            f"query dims={self.dims} n={self.n_points} r={self.n_queries} "
            f"p={p_label} leafsize={self.leafsize} k={self.k}"
        )


# ---------------------------------------------------------------------------
# Timing helpers — three implementations, two execution modes
# ---------------------------------------------------------------------------


def time_build(
    data: np.ndarray, leafsize: int, *, repeat: int, warmup: int
) -> tuple[float, float, float]:
    return (
        time_call(
            lambda: KDTree(data, leafsize=leafsize), repeat=repeat, warmup=warmup
        ),
        time_call(
            lambda: SciPyKDTree(data, leafsize=leafsize),
            repeat=repeat,
            warmup=warmup,
        ),
        time_call(
            lambda: cKDTree(data, leafsize=leafsize), repeat=repeat, warmup=warmup
        ),
    )


def time_query_pair(
    rust_tree: KDTree,
    scipy_tree: SciPyKDTree,
    scipy_ckd: cKDTree,
    queries: np.ndarray,
    *,
    k: int,
    p: float,
    repeat: int,
    warmup: int,
) -> dict[str, float]:
    return {
        "kd_serial": time_call(
            lambda: rust_tree.query(queries, k=k, p=p, parallel=False),
            repeat=repeat,
            warmup=warmup,
        ),
        "scipy_kd_serial": time_call(
            lambda: scipy_tree.query(queries, k=k, p=p, workers=1),
            repeat=repeat,
            warmup=warmup,
        ),
        "scipy_ck_serial": time_call(
            lambda: scipy_ckd.query(queries, k=k, p=p, workers=1),
            repeat=repeat,
            warmup=warmup,
        ),
        "kd_parallel": time_call(
            lambda: rust_tree.query(queries, k=k, p=p, parallel=True),
            repeat=repeat,
            warmup=warmup,
        ),
        "scipy_kd_parallel": time_call(
            lambda: scipy_tree.query(queries, k=k, p=p, workers=-1),
            repeat=repeat,
            warmup=warmup,
        ),
        "scipy_ck_parallel": time_call(
            lambda: scipy_ckd.query(queries, k=k, p=p, workers=-1),
            repeat=repeat,
            warmup=warmup,
        ),
    }


# ---------------------------------------------------------------------------
# Per-section runners
# ---------------------------------------------------------------------------


def print_build_row(label: str, kd: float, scipy_kd: float, scipy_ck: float) -> None:
    print(f"  build       {fmt(kd)}  {fmt(scipy_kd)}  {fmt(scipy_ck)}    [{label}]")


def print_query_rows(label: str, m: dict[str, float]) -> None:
    print(
        f"  serial      {fmt(m['kd_serial'])}  {fmt(m['scipy_kd_serial'])}"
        f"  {fmt(m['scipy_ck_serial'])}    [{label}]"
    )
    print(
        f"  parallel    {fmt(m['kd_parallel'])}  {fmt(m['scipy_kd_parallel'])}"
        f"  {fmt(m['scipy_ck_parallel'])}    [{label}]"
    )


def header() -> None:
    print(f"  {'operation':<10}  {'kdtree_ms':>9}  {'scipy_kd_ms':>9}  {'scipy_ck_ms':>9}")


def run_build_section(*, repeat: int, warmup: int) -> list[tuple[str, float, float]]:
    """SciPy `Build` -- bimodal data, dims in {3,8,16}, n=10000."""
    print("\n=== Build (mirrors SciPy `Build`) ===")
    header()
    summary: list[tuple[str, float, float]] = []
    for dims, n, r in SIZES:
        case = BuildCase(dims=dims, n_points=n, n_queries=r)
        data = make_data(case.distribution, n, dims)
        kd, sk, sc = time_build(data, case.leafsize, repeat=repeat, warmup=warmup)
        print_build_row(case.label(), kd, sk, sc)
        summary.append((case.label(), kd, sc))
    return summary


def run_build_unbalanced_section(
    *, repeat: int, warmup: int
) -> list[tuple[str, float, float]]:
    """SciPy `BuildUnbalanced` -- sweep order in {random, sorted}."""
    print("\n=== BuildUnbalanced (mirrors SciPy `BuildUnbalanced`) ===")
    header()
    summary: list[tuple[str, float, float]] = []
    for dims, n, r in SIZES:
        for order in ("uniform", "sorted"):
            case = UnbalancedBuildCase(
                dims=dims,
                n_points=n,
                n_queries=r,
                distribution=order,  # type: ignore[arg-type]
            )
            data = make_data(case.distribution, n, dims)
            kd, sk, sc = time_build(data, case.leafsize, repeat=repeat, warmup=warmup)
            print_build_row(case.label(), kd, sk, sc)
            summary.append((case.label(), kd, sc))
    return summary


def run_query_unbalanced_section(
    *, repeat: int, warmup: int
) -> list[tuple[str, float, float, float, float]]:
    """SciPy `QueryUnbalanced` -- k=1 query against random/sorted-built tree."""
    print("\n=== QueryUnbalanced (mirrors SciPy `QueryUnbalanced`) ===")
    header()
    summary: list[tuple[str, float, float, float, float]] = []
    for dims, n, r in SIZES:
        for order in ("uniform", "sorted"):
            case = UnbalancedQueryCase(
                dims=dims,
                n_points=n,
                n_queries=r,
                distribution=order,  # type: ignore[arg-type]
            )
            data = make_data(case.distribution, n, dims)
            queries = make_queries("uniform", r, dims)
            rust_tree = KDTree(data, leafsize=case.leafsize)
            scipy_tree = SciPyKDTree(data, leafsize=case.leafsize)
            scipy_ckd = cKDTree(data, leafsize=case.leafsize)
            m = time_query_pair(
                rust_tree,
                scipy_tree,
                scipy_ckd,
                queries,
                k=case.k,
                p=case.p,
                repeat=repeat,
                warmup=warmup,
            )
            print_query_rows(case.label(), m)
            summary.append(
                (
                    case.label(),
                    m["kd_serial"],
                    m["scipy_ck_serial"],
                    m["kd_parallel"],
                    m["scipy_ck_parallel"],
                )
            )
    return summary


def run_query_section(
    *, repeat: int, warmup: int
) -> list[tuple[str, float, float, float, float]]:
    """SciPy `Query` -- sweep dims by p by leafsize, k=1, uniform data."""
    print("\n=== Query (mirrors SciPy `Query`, boxsize axis omitted) ===")
    header()
    summary: list[tuple[str, float, float, float, float]] = []
    for dims, n, r in SIZES:
        for p in P_VALUES:
            for leafsize in LEAF_SIZES:
                case = QueryCase(
                    dims=dims,
                    n_points=n,
                    n_queries=r,
                    p=p,
                    leafsize=leafsize,
                )
                data = make_data(case.distribution, n, dims)
                queries = make_queries(case.distribution, r, dims)
                rust_tree = KDTree(data, leafsize=case.leafsize)
                scipy_tree = SciPyKDTree(data, leafsize=case.leafsize)
                scipy_ckd = cKDTree(data, leafsize=case.leafsize)
                m = time_query_pair(
                    rust_tree,
                    scipy_tree,
                    scipy_ckd,
                    queries,
                    k=case.k,
                    p=case.p,
                    repeat=repeat,
                    warmup=warmup,
                )
                print_query_rows(case.label(), m)
                summary.append(
                    (
                        case.label(),
                        m["kd_serial"],
                        m["scipy_ck_serial"],
                        m["kd_parallel"],
                        m["scipy_ck_parallel"],
                    )
                )
    return summary


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    build_results: list[tuple[str, float, float]],
    query_results: list[tuple[str, float, float, float, float]],
) -> None:
    print("\n=== Summary (vs scipy.cKDTree, ratio < 1.0 means we win) ===")
    print(f"  {'case':<60}  {'serial':>8}  {'parallel':>8}")
    for label, kd, sc in build_results:
        ratio_serial = kd / sc if sc > 0 else float("inf")
        print(f"  {label:<60}  {ratio_serial:>8.3f}        --")
    for label, kd_s, sc_s, kd_p, sc_p in query_results:
        rs = kd_s / sc_s if sc_s > 0 else float("inf")
        rp = kd_p / sc_p if sc_p > 0 else float("inf")
        print(f"  {label:<60}  {rs:>8.3f}  {rp:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--sections",
        nargs="*",
        default=["build", "build-unbal", "query-unbal", "query"],
        help="subset of sections to run",
    )
    args = parser.parse_args()

    sections = set(args.sections)
    build_results: list[tuple[str, float, float]] = []
    query_results: list[tuple[str, float, float, float, float]] = []

    if "build" in sections:
        build_results.extend(run_build_section(repeat=args.repeat, warmup=args.warmup))
    if "build-unbal" in sections:
        build_results.extend(
            run_build_unbalanced_section(repeat=args.repeat, warmup=args.warmup)
        )
    if "query-unbal" in sections:
        query_results.extend(
            run_query_unbalanced_section(repeat=args.repeat, warmup=args.warmup)
        )
    if "query" in sections:
        query_results.extend(
            run_query_section(repeat=args.repeat, warmup=args.warmup)
        )

    print_summary(build_results, query_results)


if __name__ == "__main__":
    main()
