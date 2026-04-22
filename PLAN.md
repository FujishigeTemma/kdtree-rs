# KDTree Plan

Date: 2026-04-21

## Goal

Rust で KDTree を実装し、Python ライブラリとして提供する。

この PLAN は最終形の設計書である。試作版、初版、後で広げる前提では書かない。

このプロジェクトの優先順位は次の通り:

1. ミニマルで読みやすい設計
2. 高い実行性能
3. free-threaded CPython 向けの最新構成
4. Python から自然に使える API

SciPy 互換は主目的ではない。SciPy は参照実装と benchmark 対象として使うが、API と内部設計はこのライブラリに最適化する。

## 要件の確定

- `Python>=1.13` は存在しない表記なので、`CPython >= 3.13` の意味だと解釈する。
- サポート対象は free-threaded CPython のみ:
  - `cp313t`
  - `cp314t`
- GIL ありの `cp313` / `cp314` は対象外にする。
- この実装をそのまま完成形として扱う。
- この完成形には、主要機能、配布、テスト、benchmark、型定義、ドキュメントを含める。

## 2026-04-21 時点で確認した最新版

| Component | Confirmed latest | Role |
| --- | --- | --- |
| PyO3 | `0.28.3` | Rust/Python bindings |
| maturin | `1.13.1` | PEP 517 backend / wheel builder |
| uv | `0.11.7` | env, lock, sync, build frontend, publish frontend |
| rust-numpy (`numpy` crate) | `0.28.0` | NumPy interop |
| NumPy | `2.4.4` | Python-side ndarray runtime |
| SciPy | `1.17.1` | reference / benchmark only |
| pytest | `9.0.3` | test runner |
| Ruff | `0.15.11` | lint / format |

## Packaging 方針

### 採用スタック

- `uv` をプロジェクト管理の表口にする。
- `maturin` を `pyproject.toml` の build backend にする。
- `PyO3` を Python binding 層に使う。
- `rust-numpy` を ndarray 受け渡しの標準経路にする。

### Python パッケージ構成

- mixed Rust/Python layout を採用する。
- Rust code は `src/`
- Python package は `python/kdtree/`
- native extension module は `kdtree._core`
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
- これは project-local な判断:
  - 現状の Python packaging metadata では「free-threaded Python 必須」を十分きれいに表現しづらい。
  - sdist を公開すると、非対象の `cp313` / `cp314` で source build が試行される導線が増える。
- 公開成果物は wheel-only とする。

## ライブラリとしての完成条件

このプロジェクトで「完成」とみなす条件は以下:

- KDTree 構築ができる
- 1 点 / 複数点の k-nearest query ができる
- Python API が型付きで公開される
- free-threaded wheel が `cp313t` / `cp314t` で配布できる
- correctness test と benchmark が揃っている
- README と使用例がある

## Public API

SciPy の写経はしない。薄く、読みやすく、かつ用途として不足がない API にする。

### クラス

- `class KDTree`

### constructor

- `KDTree(data, *, leafsize=32, copy_data=False)`

設計意図:

- `data` は shape `(n_points, ndim)` の 2-D array-like
- 内部表現は `float64` contiguous row-major
- `leafsize` の default は `32`
  - これは project-local な判断
  - `10` は SciPy 由来の値だが、現代 CPU では leaf を少し大きめにした方が実装と性能のバランスを取りやすい

### properties

- `ndim: int`
- `n_points: int`
- `leafsize: int`
- `data: numpy.ndarray`

### methods

- `query(x, *, k=1, p=2.0, max_distance=None, eps=0.0, parallel=None)`
- `__len__()`
- `__repr__()`

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
- `src/lib.rs`
- `src/error.rs`
- `src/metric.rs`
- `src/node.rs`
- `src/tree.rs`
- `src/query.rs`
- `python/kdtree/__init__.py`
- `python/kdtree/_core.pyi`
- `python/kdtree/py.typed`
- `tests/test_api.py`
- `tests/test_correctness.py`
- `scripts/benchmark.py`

## Rust core 設計

### データ表現

- 点群は `Vec<f64>` に row-major で保持する
- 元の論理 index は `Vec<usize>` で持つ
- 点データの重複コピーは避ける
- tree は pointer-based ではなく flat node array で保持する

理由:

- flat array の方が読みやすく、ownership が単純
- cache locality が良い
- Python からは immutable object として扱いやすい

### node 表現

- node は二択の `enum`:
  - `Leaf { start, end }` — leaf 内に並ぶ点の範囲
  - `Inner { left, right, split_dim, split_value }` — 分割軸と分割値、子 node id
- 全 node に bounding box を持たせない
- データ全体の bounding box は tree 直下に 1 つだけ持ち、query 開始時の per-axis side-distance 初期化に使う
- 子 cell の bounding box は親 cell から `split_dim` 上の片側を `split_value` で詰めただけなので暗黙に決まる

### build アルゴリズム

- split axis は widest spread dimension
- split point は median split
- Rust の `select_nth_unstable_by` を使って in-place partition
- `leafsize` 以下で leaf 化

設計判断:

- sliding midpoint より median split を優先する
- 理由は可読性と実装の局所性
- このライブラリでは median split を最終方針とする

### query アルゴリズム

- branch-and-bound
- 単一点 / batch を区別せず、同じコードパスで処理する
- 下界 (current cell までの L^p 距離) は per-axis に保持し、descent 中に O(1) で更新する
  - parent から child に降りるとき、cell が変化するのは `split_dim` 1 軸のみ
  - 「near 側」(query が同じ側に居る方) は `side_dist[split_dim]` 不変
  - 「far 側」は `side_dist[split_dim]` を `|q[d] - split_value|` の axis_accum 値に置き換え、合計を sum / max に応じて差し替えで更新
- traversal は recursive DFS。near 側を先に降りて upper bound を絞り、far 側は `min_dist * (1+eps)^p > upper` で prune
  - tree 深さは log_2(n_points / leafsize) なので stack overflow にはならない
- pop 時点で current upper bound を再チェックして prune する
- leaf 内の per-point 距離計算は chunk (8 軸ずつ) ごとに upper bound を再チェックして早期打ち切り
  - 1 軸ごとに分岐すると LLVM auto-vectorize が無効になるため、SIMD 可能な chunk 単位で打ち切る
- k 個の neighbor は小さな sorted Vec で保持し、insertion sort で更新する

## 並列化方針

free-threaded Python のみを対象にするので、batch query は Rust 内で積極的に並列化する。

### 採用案

- `rayon` を採用する
- 並列対象: batch `query`
- `parallel=None` のとき:
  - query 数が小さければ single-thread
  - query 数が閾値 (256) を超えたら自動並列
- Python から呼ばれている間は `Python::detach` を使い、CPython runtime に不要な時間は attachment を外す

### thread-safety

- `KDTree` は immutable
- 内部バッファは query ごとにローカルに確保
- global mutable cache は作らない

## NumPy interop 方針

### input

- `numpy.ndarray[float64]` を最速経路にする
- array-like も受けるが、最終的には contiguous `float64` に正規化する

### output

- nearest query の `distances` と `indices` は NumPy array で返す
- `data` property は NumPy array view または copy-safe な返却戦略にする

### dtype

- 内部実装は `f64` 固定
- `float32` / `int` 入力は受けるが `f64` に昇格させる

理由:

- 汎用性よりシンプルさを優先
- metric 実装と境界条件が明確になる

## Python 側の構成

### `pyproject.toml`

- `[build-system]`
  - `maturin>=1.13,<2`
- `[project]`
  - `requires-python = ">=3.13"`
  - runtime dependency は `numpy>=2.4`
- `[dependency-groups]`
  - `dev`: `pytest`, `ruff`, `maturin`
  - `bench`: `scipy`

### `tool.maturin`

- `module-name = "kdtree._core"`
- `python-source = "python"`
- `python-packages = ["kdtree"]`

### `Cargo.toml`

- `pyo3 = "0.28.3"`
- `numpy = "0.28.0"`
- `rayon` を追加

## テスト方針

SciPy との厳密互換ではなく、brute-force reference との一致を正とする。

### Rust unit tests

- build invariant
- median partition invariant
- bounding box lower bound
- nearest query correctness
- duplicate points
- degenerate data

### Python tests

- constructor validation
- dtype coercion
- 1 点 / batch query
- `k = 1`, `k > 1`
- `p = 1`, `2`, `inf`
- `eps`
- `max_distance`
- multi-thread からの query safety

### SciPy の位置付け

- correctness の主基準ではない
- benchmark と参考比較に使う

## Benchmark 方針

ベンチマークは「SciPy より同じ API をまねる」ためではなく、「どの設計が速いか」を測るために置く。

### 比較対象

- 本ライブラリ
- `scipy.spatial.KDTree`
- `scipy.spatial.cKDTree`
- brute-force NumPy baseline

### 測る項目

- build time
- query one
- query batch
- query batch with automatic parallelism on/off

### パラメータ

- dimensions: `2`, `3`, `8`, `16`, `32`
- points: `1_000`, `10_000`, `100_000`
- queries: `1`, `100`, `10_000`
- leafsize: `8`, `16`, `32`, `64`
- metrics: `1`, `2`, `inf`

## 実装内容

### Packaging

- `pyproject.toml` を free-threaded 配布方針で固定する
- Python package を `python/kdtree/` に置く
- `Cargo.toml` を現行 best practice に合わせる
- `README.md` に使用例を入れる

### Core Data Structures

- point storage
- node storage
- bounding box
- error types
- metric helpers

### Tree Build

- median split build
- leaf formation
- flat node array
- build tests

### Query Engine

- 単一 / batch 共通の `query` を一本化
- k-nearest
- `eps`
- `max_distance`

### Python Binding

- `KDTree` class
- ndarray conversion
- typed stub
- repr / properties

### Parallel Execution

- `rayon` integration
- `Python::detach`
- auto parallel threshold
- thread safety tests

### Quality

- Python tests
- benchmark script
- Ruff / format
- README examples

### Release

- `cp313t` wheel
- `cp314t` wheel
- install smoke test
- import smoke test
- benchmark snapshot 記録

## リスク

- free-threaded 専用に振るため、一般的な CPython wheel より利用対象は狭い
- split-cell bbox は data-tight bbox より緩いので、特定分布では余計な leaf を訪問する可能性がある (per-descent O(1) の利得の方が勝つと判断して採用)
- median split は実装が素直だが、特定分布では最速ではない可能性がある
- 内部並列化は小規模 query では逆効果になり得るので閾値調整が必要
- 現在この環境のグローバル `uv` は `0.8.13` で最新ではないため、プロジェクト内の設定は `0.11.7` の公式仕様に寄せ、ローカル global install には依存しない

## Sources

- PyO3 features: https://pyo3.rs/v0.28.3/features
- PyO3 building/distribution: https://pyo3.rs/main/building-and-distribution.html
- PyO3 free-threading: https://pyo3.rs/main/free-threading.html
- maturin guide: https://www.maturin.rs/
- maturin config: https://www.maturin.rs/config.html
- uv docs: https://docs.astral.sh/uv/
- uv project init: https://docs.astral.sh/uv/concepts/projects/init/
- uv build: https://docs.astral.sh/uv/concepts/projects/build/
- uv dependency management: https://docs.astral.sh/uv/concepts/projects/dependencies/
- Python stable ABI: https://docs.python.org/3.13/c-api/stable.html
- Python free-threading HOWTO for extensions: https://docs.python.org/3.13/howto/free-threading-extensions.html
- SciPy KDTree source: https://github.com/scipy/scipy/blob/main/scipy/spatial/_kdtree.py
- SciPy spatial benchmarks: https://github.com/scipy/scipy/blob/main/benchmarks/benchmarks/spatial.py
- PyPI: maturin https://pypi.org/project/maturin/
- PyPI: uv https://pypi.org/project/uv/
- Docs.rs: PyO3 https://docs.rs/crate/pyo3/0.28.3
- Docs.rs: rust-numpy https://docs.rs/crate/numpy/latest
- PyPI: NumPy https://pypi.org/project/numpy/
- PyPI: SciPy https://pypi.org/project/scipy/
- PyPI: pytest https://pypi.org/project/pytest/
- PyPI: Ruff https://pypi.org/project/ruff/
