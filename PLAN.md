# KDTree Plan

Date: 2026-08-12

## Goal

Rust で KDTree を実装し、Python ライブラリとして提供する。

この PLAN は最終形の設計書である。試作版、初版、後で広げる前提では書かない。

このプロジェクトの優先順位は次の通り:

1. ミニマルで読みやすい設計
2. 高い実行性能
3. free-threaded CPython 向けの最新構成
4. Python から自然に使える API

SciPy 完全互換は主目的ではない。SciPy は参照実装と benchmark 対象として使うが、API と内部設計はこのライブラリに最適化する。

## 要件の確定

- `CPython >= 3.13`
- サポート対象は free-threaded CPython のみ:
  - `cp313t`
  - `cp314t`
- GIL ありの `cp313` / `cp314` は対象外にする。
- この実装をそのまま完成形として扱う。
- この完成形には、主要機能、配布、テスト、benchmark、型定義、ドキュメントを含める。

## Packaging 方針

### 採用スタック

- `uv` でプロジェクト管理する。
- `maturin` を `pyproject.toml` の build backend にする。
- `PyO3` を Python binding 層に使う。
- `rust-numpy` を ndarray 受け渡しの標準経路にする。

### Python パッケージ構成

- mixed Rust/Python layout を採用する。
- Rust code は `src/`
- Python package は `python/kdtree/`
- top-level import は `from kdtree import KDTree`

### free-threaded 専用配布

- wheel は `cp313t` / `cp314t` のみ作る。
- `abi3` は使わない。
  - 理由: free-threaded build は現時点で Limited API / Stable ABI をサポートしない。
- `abi3t` も現時点では採用しない。
  - 理由: `abi3t` は Python `3.15` 系で導入される仕様であり、今回の対象 `3.13t` / `3.14t` には適用できない。
- module は `#[pymodule(gil_used = false)]` を前提に設計する。
- Rust 側の公開 state は immutable かつ `Send + Sync` を満たす設計にする。

### sdist 方針

- リリース成果物は wheel を優先する。
- sdist を公開すると、非対象の `cp313` / `cp314` で source build が試行される導線が増える。
- 公開成果物は wheel-only とする。

## ライブラリとしての完成条件

このプロジェクトで「完成」とみなす条件は以下:

- KDTree 構築ができる
- 1 点 / 複数点の k-nearest query ができる
- Python API が型付きで公開される
- free-threaded wheel が `cp313t` / `cp314t` で配布できる
- test と benchmark が揃っている
- README と使用例がある

## Public API

SciPy の写経はしない。薄く、読みやすく、かつ用途として過不足がない API にする。

### クラス

- `class KDTree`

### constructor

- `KDTree(x: numpy.ndarray, *, leafsize=32)`

### methods

- `query(x: numpy.ndarray, *, k=1, p=2.0, max_distance=None, eps=0.0, parallel=False)`
- `__len__()`
- `__repr__()`

### properties

- `data`
- `ndim`
- `n_points`
- `leafsize`

### API の意味

- `query`
  - 1 点と batch の両方を受ける
  - `(distances, indices)` を返す
  - `k == 1` でも返り値 shape は予測可能な規則にそろえる
  - SciPy の squeeze 挙動そのものには合わせない

### あえて入れないもの

- 半径検索 (`query_radius`)
- 点対列挙 (`query_pairs`)
- SciPy 互換の細かい shape 仕様
- `workers`
- `boxsize`
- weighted query
- sparse distance matrix
- pickle 互換
- 可変な tree update

このライブラリは immutable KDTree に絞り、k-nearest query 専用とする。

## アーキテクチャ

### 基本方針

- Rust core と Python binding を分離する。
- KDTree core は pure Rust で完結し、Python なしで unit test できるようにする。
- Python 側は input normalization と output boxing だけを担う。

### ファイル構成

- `Cargo.toml`
- `pyproject.toml`
- `README.md`
- `src/*.rs`
- `python/kdtree/__init__.py`
- `python/kdtree/_core.pyi`
- `python/kdtree/py.typed`
- `tests/test_kdtree.py`
- `tests/benchmark.py`

### dtype

- 内部実装は `f64` 固定
- `float32` / `int` 入力は受けるが `f64` に昇格させる

理由:

- 汎用性よりシンプルさを優先
- metric 実装と境界条件が明確になる

## Sources

- PyO3 features: https://pyo3.rs/v0.28.3/features
- PyO3 building/distribution: https://pyo3.rs/main/building-and-distribution.html
- PyO3 free-threading: https://pyo3.rs/main/free-threading.html
- maturin guide: https://www.maturin.rs/
- maturin config: https://www.maturin.rs/config.html
- uv docs: https://docs.astral.sh/uv/
- uv build: https://docs.astral.sh/uv/concepts/projects/build/
- uv dependency management: https://docs.astral.sh/uv/concepts/projects/dependencies/
- Python stable ABI: https://docs.python.org/3.13/c-api/stable.html
- Python free-threading HOWTO for extensions: https://docs.python.org/3.13/howto/free-threading-extensions.html
- SciPy KDTree source: https://github.com/scipy/scipy/blob/main/scipy/spatial/_kdtree.py
- SciPy spatial benchmarks: https://github.com/scipy/scipy/blob/main/benchmarks/benchmarks/spatial.py

