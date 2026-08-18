# 計測とチューニング

Date: 2026-08-17

この文書は**Rust 側のカウンタで測るときの手順と判断**を
書いたものである。

valgrind と perf は macOS で動かないので、計測環境は Linux に限る。手元の
Apple Silicon は「壁時計での確認用」であって、プロファイルは取れない。

## 結論を先に

1. **命令数（Ir）で順位付けし、cycles で採否を決める。** このクレートでは両者が
   一致しない。命令のごく一部しか占めない `partition_rows` が分岐ミスで律速する
   一方（内訳は `partition_rows` の doc comment 側）、Ir の過半を占める
   `select_nth_unstable_by` は命令を削っても cycles が動かない。
2. **壁時計では判定しない。** 計測ホストは共有でクロックが 0.4-3.7 GHz 振れる。
   cycles は同じ仕事量に対して動かないので、再現性が <1% になる。
3. **A/B は2つのバイナリを作って交互に走らせる。** セッションを跨いだ数字を比較
   しない。
4. `src/kernel.rs` を触ったら **`nm | grep point_dist`** を必ず見る。

## 最小セット

対象ホストの要件は `valgrind` / `callgrind_annotate` / `perf` / `taskset` / `gcc`
と、`rust-toolchain.toml` の nightly。OIST では `ssh h100`（`bndds01`）が満たす。

### 1. 同期

`.cargo/config.toml` が `PYO3_PYTHON` を `.venv/bin/python3` と**相対**指定して
いるので、リモート側にも venv が要る。`target/` と `.venv/` は同期しない。

```bash
rsync -az --delete \
  --exclude target --exclude .venv --exclude .benchmarks --exclude dist \
  --exclude .hypothesis --exclude __pycache__ --exclude '.*_cache' \
  ./ h100:kdtree-rs/
```

### 2. 一度だけのセットアップ

```bash
ssh h100
cd kdtree-rs
uv venv --python 3.13t .venv          # PYO3_PYTHON の相対パスを満たす
./scripts/binpath.sh                  # bench をビルドして場所を出す
```

`binpath.sh` は `cargo bench --no-run` を回して成果物のパスを1行で返す。nightly は
`rust-toolchain.toml` から来るので指定は要らない。

`pyo3` は `extension-module` なしでビルドされるので、bench バイナリは
libpython を動的リンクする。上の venv がその実体になる。

### 3. 計測

```bash
./scripts/perfstat.sh  build-d8-n10000-serial      # cycles / IPC / 分岐・キャッシュミス
./scripts/callgrind.sh build-d8-n10000-serial      # 命令数の内訳
./scripts/callgrind.sh build-d8-n10000-serial --branch-sim=yes
./scripts/memcheck.sh                              # リーク検査（bench ではなくテスト）
```

**id は1つに解決してから測る。** 一覧は `cargo bench --bench grid -- --list`、部分
文字列から完全な id を引くのは `./scripts/benchid.sh build-d8`。bench は id を部分
一致で選ぶ一方、perf も callgrind も**プロセス全体**を数えるので、2つ以上に当たる
id は合算されて1つの数字になってしまう。`perfstat.sh` と `callgrind.sh` は
`benchid.sh` を通し、解決できなければ測らずに止まる。

測るバイナリはどのスクリプトでも `BIN` で差し替える。既定は `binpath.sh` が cargo に
聞いたもので、`nm` を当てるときも同じ出所を使う（後述の「インライン化」）。

`perfstat.sh` の第2引数が反復回数（既定 200、d16 クエリなら 20 程度）で、`CPU` が
固定するコア番号（既定 4）。出力は `key=value` の並びで、`ab.sh` がここから cycles を
読む。`callgrind.sh` の第2引数以降は valgrind にそのまま渡る（反復は `--iters 1`
固定、出力は `target/callgrind/<id>.<バイナリ名>.out`。既定のバイナリ名は
`grid-<hash>` なのでビルドごとに増える。あとから `callgrind_annotate` を自分で回す
ときは glob ではなくスクリプトが出したパスを使う）。

`perfstat.sh` は `-parallel` の id を弾く。1コアに固定するので、rayon が同じコアを
取り合う様子を測ることになるからで、並列は別途 `cargo bench` で見る。

`memcheck.sh` が走らせるのは bench ではなく `--lib` のテストバイナリで、引数はテスト名
の絞り込みになる。definite leak だけをエラーにしているのは、rayon のプールとハーネス
のスレッドが終了時点で生きていて、その TLS を誰も解放できないからである
（`RAYON_NUM_THREADS=1` はそのノイズを1ブロックに抑えるためにある）。

### 4. A/B の手順

同じソースから2つのバイナリを作り、交互に走らせる。片方ずつ順に流すと、その間の
クロック変動が差として出てしまう。

```bash
./scripts/ab.sh stash a                  # 変更前
# ... src/ を編集する ...
./scripts/ab.sh stash b                  # 変更後

./scripts/ab.sh build-d8-n10000-serial   # 交互3ラウンド、各辺の min と b/a
```

id 以降の引数は `perfstat.sh` にそのまま渡る（つまり第2引数が反復回数）。ラウンド数は
`ROUNDS`（既定 3）、バイナリの置き場は `AB_DIR`（既定 `~/ab`）。ラウンド内でクロックは
動くが、A と B が同じ条件を見るので相対比較は保たれる。各ラウンドの生の行は stderr に
出るので、stdout には判定行だけが残る。

**min を採る。平均ではない。** 干渉は cycles を増やす方向にしか働かないので、各辺の
最速の観測がいちばん汚れていない。

**`stash b` を忘れると `b/a=1.000` という完全にもっともらしい判定が出る。** `~/ab` は
それを置いたセッションより長く生きるので、`ab.sh` は測る前に両者を `cmp` して同一なら
警告し、それぞれの mtime を出す。逆に、**同じバイナリを両側に置くのはこのホストの
ノイズ幅を測る較正として正しい使い方**である（下の ±3% はそうやって出す）。

判定は cycles で行う。命令数が減っても cycles が動かないことがよくある（IPC が
比例して落ちる = 削った命令は空きスロットを埋めていただけ）。命令内訳を A/B で並べたい
ときは `BIN=~/ab/a ./scripts/callgrind.sh <id>` を両辺で流す。出力名にバイナリ名が
入るので上書きされない。

## つまりどころ

### ssh: `ProxyCommand` に `ControlMaster` を書くと別ホストに着く

外側の ssh は `ProxyCommand` 文字列**全体**の `%h` を最終ターゲットに展開する。
`ControlPath` に `%h` が入っていると、踏み台への接続がターゲット名のソケットを
作り、外側の ssh がそこに多重化して、**踏み台（ログインVM）に着地する**。

多重化するなら外側だけに書く:

```
ssh -o ControlMaster=auto -o ControlPath=/tmp/kdt-cm-%h -o ControlPersist=8h \
    -o ProxyCommand="ssh -W %h:%p oist" h100
```

### ログインVMと計算ノードは別物

これを1時間分の測定ごと捨てたことがある。**必ず着地先を確認する**:

```bash
ssh h100 'hostname; nproc'
```

| | login VM (`loginc01`) | 計算ノード (`bndds01`) |
| --- | --- | --- |
| CPU | 4 core Broadwell (VM) | EPYC 9654 192 core |
| valgrind / perf / gcc | **無い** | ある |
| glibc | — | 2.28 |

`nproc` が 4 なら間違ったホストにいる。ログインVM上の数字は計算ノードの
3-8倍遅く、しかも machine-wide のクロック変動を拾うので一貫して見えてしまう。

そこで `perfstat.sh` を叩くと、コア 0-3 しかないので既定の `CPU=4` に固定できず
`taskset` が先に落ちる。perf のカウンタ不在（`<not counted>` を拾って
「perf counters unavailable」と言う）まで進まない。

### `maturin develop` は debug ビルド

`--release` を付けないと拡張モジュールが約40倍遅くなり、`tests/benchmark.py` が
「SciPy より 12-32倍遅い」と報告する。Python 側のベンチが破滅的な数字を出したら
まずこれを疑う。回帰ではない。

```bash
uv run maturin develop --uv --release
```

### callgrind のシミュレーション値は「場所」だけ信じる

`--branch-sim=yes` / `--cache-sim=yes` の絶対値は使えない。同一ホスト・同一
ワークロード（`build-d8-n10000-serial`）で:

| | callgrind | perf |
| --- | --- | --- |
| 分岐ミス | 32,323 | 3,625 |

callgrind の予測器は単純な2レベルなので、実機が完璧に予測できる分岐を外し、
**分岐の少ないコードほど過大評価する**（ここでは9倍）。分岐が多いコードでは
オーダーが合うので、「どの行で外しているか」の順位付けには使える。大きさは
`perf` で採る。キャッシュも同様で、LL を 256KB と仮定しているため実機と乖離する。

**分岐ミス1回のコストも実機依存**である。Apple Silicon はモデルよりずっと吸収
するので、分岐ミスを消す変更は x86_64 で -40%、aarch64 で -20% と効き方が倍
違う。両方で測る。

### `perf annotate` のホットな行は「原因」ではなく「依存チェーンの末端」

依存チェーンを待っている命令に cycle が課される。d8 クエリで
`Mask::<i64,8>::any()` の展開がサンプルの26%を占めていたが、等価で安い式に
置き換えても両プラットフォームで ±1.5% しか動かなかった。

**先に IPC を見る。** `Packed` の d3/d8 スキャンは既に IPC 4.1-4.4 で発行幅に
近く、命令を削っても効かない。そこを速くするには走査する点数を減らすしかない。

### `src/kernel.rs` はインライン化に極端に敏感

`codegen-units = 1` なので、クレートのどこかを増やすだけで `point_dist` が黙って
実関数呼び出しに落ちる。機構と実測値は `point_dist` と `Folded` の doc comment 側に
ある。ここで要るのは手順だけ:

```bash
nm -C "$(./scripts/binpath.sh)" | grep point_dist    # シンボルが出たら落ちている
```

**葉スキャンの変種は `Strategy` 型として足し、`point_dist` の `match m` は触らない。**
アームを1つ足すだけで、触っていない `L2` の d16 走査が遅くなる（実測値は `Folded`
の doc comment）。

### 交絡した A/B に注意

「`L2` の早期脱出は +47% の価値がある」という結果は、実は変種側でインライン化が
落ちていただけだった。独立した `Strategy` として切り出して測り直すと、d16 で
早期脱出を外す方が 12% **速い**。驚く結果が出たら `nm` を見て、変更を独自の
monomorphization に隔離して測り直す。

### コードレイアウトのノイズは約3%

`perf` の cycles で測っても、無関係なコードを足すだけで同一ワークロードが ±3%
動く。それ以下の差は判定に使わない。`ab.sh` はこの帯（awk の `band`）に入る比を
`(within layout noise)` と印字する。フルグリッドを1発ずつ流すと ±15% の見かけの
回帰が出るので、疑わしいものは必ず交互 A/B で確認する。

（`DESIGN.md` の ±10% は別の話で、あちらは Python 側の壁時計・50-200us 帯の話。）

### その他

- **NUMA**: 計算ノードは 8 ノード構成で `numactl` が無いが、`CPU` で固定すれば
  first-touch でローカルに載るので足りる。
- **`perf_event_paranoid = 2`**: ユーザ空間の `perf stat` は動く。
- **home は共有**: `~/.cargo` はログインノードと計算ノードで同じものが見える。
- **codegen フラグは既定のまま**にする。`target-cpu=native` を付けると
  AVX-512 が有効になり、CI が配る manylinux wheel（baseline x86-64）と別物を
  チューニングすることになる。計算ノードは AVX-512 を持っているので特に注意。
- **callgrind の測定区間**は `benches/grid.rs` の `kdtree_bench_target`。なぜ
  `#[unsafe(no_mangle)]` と `#[inline(never)]` が付いているかはそこに書いてある。
