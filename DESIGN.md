# KDTree core design

Date: 2026-08-14

`PLAN.md` がパッケージング・公開 API の設計書であるのに対し、この文書は Rust core
の**アルゴリズムと抽象**の設計書である。

この文書は 2 部構成:

- 第 1 部: 現行実装が持っている高速化の工夫の全列挙（何を失ってはいけないか）
- 第 2 部: それらを綺麗に表現するためのゼロベース設計

---

# 第 1 部: 現行実装の工夫（全列挙）

## A. Python 境界

1. **f64 ndarray の直接 extract**。`PyReadonlyArrayDyn<f64>` の extract をまず試し、
   成功したら Python 側の呼び出しは 0 回。非 f64 のときだけ `astype` を 1 回呼ぶ。
2. **contiguous なら memcpy**。`as_slice()` が取れれば `to_vec()`、取れないときだけ
   要素ごとのコピーに落ちる。
3. **GIL 解放**。build と query の両方を `py.detach` で囲む。データを値で受け取るので
   Python 側の借用が detach より前に終わっている。
4. **free-threaded 前提**。`#[pymodule(gil_used = false)]` + `#[pyclass(frozen)]`。
5. **出力の再割り当てを作らない**。index は最初から `Vec<i64>` で作る。
6. **元順序はオンデマンド**。内部は tree 順序のまま持ち、`data` getter のときだけ
   逆置換して materialize する。

## B. 距離の代数（reduced distance）

7. **reduced distance 領域**。L2 なら二乗、L^p なら p 乗、L1/L^inf は素の値。比較・
   枝刈り・累積を全部この単調像で行い、平方根は「出力 1 件につき 1 回」だけ取る。
8. **L1 / L2 / L^inf を LP(p) から分離**。`powf` には SIMD 命令が無いので、
   汎用 L^p はベクトル経路に入れない。
9. **`eps_factor` を reduced 領域に持ち込む**。近似探索は「下界 × (1+eps)^p」を
   比較する。best 側を割らないので除算も再ルートも出ない。
10. **`replace_axis` による O(1) 下界更新**。親セルの下界から split 軸の寄与だけを
    差し替えて far child の下界を得る。L^inf は max、それ以外は減算+加算。
11. **`bound()` を cutoff で cap**。`max_distance` が探索の最初から枝刈りに効く。

## C. 記憶レイアウト

12. **preorder フラット node 配列**。部分木は node 配列上の連続窓になる。
13. **点の物理的な並べ替え**。build 中に行そのものを動かすので、どの部分木（特に葉）も
    row-major な連続ブロックを占める。葉走査が純粋なストリーミングになる。
14. **`indices[pos] -> 元の行番号`**、`u32`。
15. **node ごとの tight bounding box**。`2 * ndim` 値 / node のフラット配列。
    分割平面より強い下界を与える。
16. **`Node` を 24 バイト以内に収める**。id は `u32`、`order_by_box: bool` は
    `f64` のアラインメントが要求する既存 padding に収まるので実質無料。

## D. build

17. **分割位置を `len / 2` に固定**。これでレイアウト全体が `(n_points, leafsize)` の
    純関数になる。`count_nodes` で各部分木の node 数が事前に分かるので、子に
    **disjoint な `&mut` 窓**を切り出せる。結果として直列でも並列でも**同一の木**が
    できる（並列化のためのロックも後段のマージも不要）。
18. **割り当ては n によらず定数個**。出力 4 本 + `keys` scratch 1 本だけ。
    （テストで `< 20 allocations` を固定している）
19. **median 選択を連続コピーに対して行う**。分割軸の座標だけを `keys` に抜き出して
    `select_nth_unstable_by` する。index 経由の比較ごとの gather を消す。
20. **行を 1 回だけ動かす分割**。Hoare 2 ポインタで `< pivot` を寄せ、その後
    pivot 同値行を隙間に引き込む fix-up。Dutch flag ループの約 1/3 の移動量。
21. **`swap_rows` を行幅で monomorphize**。固定幅ならアドレス計算もコピーも展開され、
    動的幅は `swap_nonoverlapping` 1 回。
22. **再帰全体を行幅で monomorphize**（`ndim ∈ {1,2,3,4,8,16}` + 動的 fallback）。
23. **root の bbox 計算と有限性検査を 1 パスに融合**。入力を 1 回しか読まない。
24. **bbox kernel をフラットベクトルで流す**。行ごとに chunk するのは行が
    ベクトル幅より短いと無意味なので、ブロック全体を `LANES` 幅で流す。
    フラットベクトル `i` の lane `j` は次元 `(i*LANES + j) % ndim` を持ち、この
    パターンは `ndim / gcd(ndim, LANES)` ベクトルごとに巡回する。その本数だけ
    phase accumulator を持てば全次元をカバーでき、最後に scalar で
    lane → 次元へ散らす。周期が長すぎる行幅は scalar fallback。
25. **`v * 0 != 0` で非有限を検出**。inf と NaN の両方を 1 乗算 + 1 比較で拾う。
26. **`rayon::join` による子の並列 build**。既定は off（Python 側が既に N スレッドで
    並列に木を建てている場合の oversubscription を避ける）。
27. **`order_by_box` の bottom-up 判定**。両方の子が split 以外の全次元で親の 0.6 未満に
    縮んでいて、かつ部分木が 64 点以上ならフラグを立てる。

## E. query 降下

28. **コスト順の 2 段枝刈り**。まず O(1) の incremental 分割平面下界。これで刈れない
    ときだけ、その部分木が実際に含む点の tight box 距離（O(ndim)）を計算する。
    well-separated なデータでは box を一切触らない。
29. **`cell`（軸ごとの寄与）の save/restore プロトコル**。far child に降りる直前に
    1 スロットだけ書き換え、戻ったら復元する。
30. **クエリ間の全ゼロ不変**。クエリが root box の内側なら `cell` はゼロのままで
    正しいので、`box_dist == 0.0` を fast path にして seed 自体を飛ばす。
    外側のクエリだけが埋めて、降下後にゼロへ戻す。
31. **`Best1` / `BestK` の monomorphize**。`k == 1` はヒープを持たず 2 レジスタ。
32. **`order_by_box` node は box 距離で子を順序付ける**。両方の子を box で gate し、
    片方を訪問した後の締まった bound で再テストする。`#[inline(never)]` で
    hot path から追い出してあるので、フラグが立たないデータはフラグ判定しか払わない。
33. **葉走査の bound を 1 回だけ取る**。以降は hit したときに `on_hit` の戻り値で
    更新する（点ごとに k-best を再問い合わせしない）。
34. **`Scratch` と `Descent` の分離**。全部を 1 本の `&mut` で渡すと、再帰呼び出しの
    たびに読み取り専用コンテキストの再ロードが強制される。
35. **metric は runtime enum dispatch のまま**。descend を metric で monomorphize すると
    kernel が inline 展開されて descend が肥大し、遅くなる（計測済み）。
36. **バッチ query の並列化**。`par_chunks_mut(k)` + `with_min_len(16)` +
    `for_each_init` でスレッドごとに scratch を持つ。1 クエリ 1 タスクだと
    タスク生成コストが仕事を上回る。

## F. SIMD kernel

37. **`LANES = 8` を論理幅として固定**。`std::simd` が SSE2 / NEON / AVX-512 へ
    それぞれ降ろす。アーキ固有コードは書かない。
38. **`vmax` / `vmin` を compare + select で書く**。全データが構築時に有限と検証済みなので
    IEEE の maxNum 意味論は不要で、compare 形の方が packed max 1 命令に落ちる。
39. **`hsum` / `hmax` を pairwise tree で書く**。`reduce_sum` は順序付き（＝直列）
    リダクションに落ちる。
40. **bound チェックの粒度は「SIMD chunk ごと」**。軸ごとにすると horizontal reduction が
    毎軸入って直列化し、行ごとにすると高次元での early exit を失う。
41. **行幅ごとの葉走査カーネル**:
    - d1 / d5-7: 軸ループを完全展開した scalar。early exit を捨てる代わりに
      LLVM が**点方向に**自動ベクトル化する。
    - d2 / d3 / d4: 1 レジスタに複数点を詰め、swizzle で軸ごとに de-interleave する。
    - d8: 1 点 = 1 レジスタ。8 点を 3 段の butterfly で同時に畳む
      （点ごとに `hsum` すると shuffle が倍で、しかも 1 依存鎖に直列化する）。
    - それ以外 / LP: 点ごとに early exit 付きで走査。
42. **`scan_chunks` のベクトル gate**。P 点分の reduced distance をまとめて bound と
    比較し、in-bound な lane が 1 つでもある**稀な** chunk でだけ scalar に降りる。
43. **`on_hit` が新しい bound を返す**。コールバックが bound を所有する。
44. **LP は全ベクトル経路を迂回**。各入口で最初に振り分けるので、ベクトル
    プリミティブの `LP` アームは到達不能（`unreachable!`）。
45. **`codegen-units = 1`**。descend と距離カーネルの cross-module inline を確実にする。

## G. 「やって遅くなった」記録（設計で壊してはいけない負の計測結果）

これらは全部「綺麗にしようとすると踏む地雷」なので、明示的に残す。

| 変更 | 結果 |
| --- | --- |
| ベクトルプリミティブの引数を 3 variant enum に絞る | d8/d16 の葉走査で **10-25% 遅い** |
| `descend` を metric で monomorphize | 遅い（kernel が inline 展開されて descend が肥大） |
| `Descent` と `Scratch` を 1 本の `&mut` に統合 | 再帰後にコンテキスト再ロード |
| `codegen-units` を既定の 16 に戻す | descend 律速のワークロードで最大 **10% 遅い** |
| `simd_max` / `reduce_sum` を使う | maxNum の fixup / 直列リダクション |
| bound チェックを軸ごと or 行ごとにする | どちらも遅い |
| `descend_by_box` を inline させる | フラットデータの hot path が太る |
| `point_dist` を `#[inline]` のままにする | d16 の葉走査で **12% 遅い**（下記） |

### 抽象が性能を落とす 3 つの経路（実測で全部踏んだ）

型で語彙を整えると、**同じ命令列にならない**ことがある。踏んだのは 3 パターンで、
どれも「値がレジスタから追い出される / inline されなくなる」という同じ根:

1. **`(data, ndim)` を `&mut RowsMut` にすると、行幅が毎回メモリロードになる。**
   `partition_rows` の Hoare ループは `rows.coord(i, dim)` が本体なので、
   `self.width` のロードが毎反復入る。**`RowsMut` を値渡しにする**と SROA が効いて
   幅がレジスタに載る（build d5 で 12.5% 差）。
2. **走査の bound を構造体フィールドに持つと、点ごとのロードになる。**
   `Scan` に `bound: Dist` を置くと `&mut self` 越しの読み出しが毎点入る。
   **bound は「単調に締まっていく値」として引数と戻り値で流す**のが正しい
   （型としても正直で、レジスタにも載る）。
3. **`#[inline(always)]` を付けないと、hot path が関数呼び出しに戻る。**（下記）

**教訓**: 「ゼロコスト抽象」はゼロコストに*できる*というだけで、自動ではない。
リファクタ後は必ず `nm` でシンボル差分を見て、A/B を取る。

### `codegen-units = 1` と inline 判断の結合（実測で踏んだ）

`codegen-units = 1` は crate 全体を 1 つの翻訳単位にするので、LLVM の inline
ヒューリスティックは**クレート全体のコードサイズに依存する**。リファクタで型と
メソッドを増やしただけで、それまで inline されていた `kernel::point_dist` が
アウトオブライン関数として残るようになり、d16（wide-row = `streamed` 経路）の
葉走査が 12% 遅くなった。d8 以下（packed 経路）は無関係なので、
「d16 だけ遅い」という分かりにくい形で出る。

診断は `nm` でシンボルが残っているかを見るのが速い:

```bash
nm target/release/libkdtree.dylib | grep point_dist   # 出たら inline されていない
```

**教訓**: `codegen-units = 1` を使う以上、hot path の inline は
ヒューリスティックに任せず `#[inline(always)]` で固定する。
リファクタ後は「シンボルが増えていないか」を確認する。

---

# 第 2 部: ゼロベース設計

## 0. 現状の何が汚いのか

関数の粒度やファイル分割は問題ではない。問題は**ドメインの概念が型になっていない**こと:

- 「reduced distance かどうか」がコメントで管理されている（`query.rs` の冒頭が
  「このモジュールの距離値の `f64` は全部 reduced distance」と宣言している）。
- 「フラット slice + 行幅」という 1 個の概念が、`(data, ndim)` の 2 引数として
  全モジュール 8 箇所以上に散らばっている。
- 「bounding box」が `&[f64]` + `split_box(bounds, ndim)` という手作業の切り出しで、
  `2 * ndim * id` のアドレス計算が 5 箇所に露出している。
- 「`cell` は クエリ間で全ゼロ」という不変条件が 4 つのメソッドに散らばって
  手で維持されている。
- 「行幅による特殊化」が `const D: usize` として 8 個の関数シグネチャを貫通している。
- 「葉走査カーネル」が 9 アームの match と 4 つの専用関数で、共通部分
  （bound gate、hit のばら撒き、端数処理）が `scan_chunks` に半分だけ括り出されている。

つまり **抽象の穴が 6 個**ある。設計はこの 6 個を埋めるだけでよく、それ以上は要らない。

## 1. 型: `Dist` — reduced-distance 表現を持つ距離

```rust
/// 単調像としての距離。生成と復元は `Metric` だけが行う。
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Dist(f64);
```

- 構築子は `Metric::reduce` / kernel の内部だけ。復元は `Metric::restore` だけ。
- `Dist` 同士の `min` / 比較 / `eps` 倍は許す。true distance との混同はコンパイルエラー。
- **削除されるもの**: `metric.rs` / `query.rs` / `kernel.rs` 冒頭の「この値は reduced だ」
  という宣言コメント群と、それを人間が追う必要そのもの。
- ランタイムコストはゼロ（`repr(transparent)`）。

## 2. `Metric` の分解は検討して却下した

現行の `L1 | L2 | LInf | LP(p)` は、一見 2 つの独立な軸の直積に見える:

| | 軸写像 | 畳み込み |
| --- | --- | --- |
| L1 | `abs` | sum |
| L2 | `square` | sum |
| L^inf | `abs` | **max** |
| L^p | `powf(p)` | sum |

`struct Metric { axis: AxisMap, fold: Fold }` にすると `reduce` / `restore` /
`combine` は確かに機械的になり、`combine` からは `unreachable!` が消える。

**しかし `fold_scalar` と `axes_lanes` が悪化する。** 現在この 2 つは
「match をループの外に出して 3 本のタイトなループ本体を持つ」形で、これが
工夫 41 の要になっている。直積に分解すると match は `(axis, fold)` の組に対して
行うことになり、**3 本 → 4 本 + 到達不能アーム 1 本**（`Square × Max` はどの
metric も作らないが型の上では存在する）になる。負の計測結果 G-1 が起きたのと
同じ領域なので、リスクに見合わない。

したがって `Metric` は 4 variant の enum のまま残す。実際に入れた変更は:

- `Dist` newtype の導入（第 1 の穴）
- `axis_rd` の削除（`reduce` と同一実装の別名だった）

分解を再検討するなら、ランタイム enum ではなく**型レベル化**（ZST の型引数）で
行うこと。ただし `scan_leaf` の幅テーブルが 3 倍実体化されるので、A/B 必須。

## 3. 型: `Rows<'a, W>` / `RowsMut<'a, W>` — 行幅つき連続点ブロック

```rust
trait Width: Copy { fn ndim(self) -> usize; }
struct Dyn(usize);
struct Fixed<const N: usize>;

struct Rows<'a, W: Width = Dyn> { flat: &'a [f64], width: W }
```

操作: `len()` / `row(i)` / `iter()` / `split_at(mid)` / `bbox_into(BBoxMut)` /
`bbox_checked_into(..) -> bool`、`RowsMut` に `swap(a, b)` / `partition(dim, pivot, mid)`。

- **`(data, ndim)` の 2 引数ペアが全部消える**。`kernel::bbox(data, ndim, lo, hi)`、
  `partition_rows(data, indices, ndim, dim, pivot, mid)`、`swap_rows(data, ndim, a, b)`、
  `tree.rows(start, end)` + 暗黙の ndim、`block.chunks_exact(ndim)` が全部
  `Rows` のメソッドになる。
- **`const D: usize` が 8 個の関数シグネチャから消えて、1 個の型引数になる**。
  monomorphize は `Rows<'_, Fixed<8>>` を作るだけで自動的に効く。
- 行幅の特殊化は 1 個のディスパッチ機構に集約する:

  ```rust
  fn with_width<R>(ndim: usize, f: impl WidthFn<R>) -> R
  ```

  build は `{1,2,3,4,8,16}` を、葉走査は `{1,5,6,7}` を実体化する
  （**どの幅を実体化するかは呼び出し側のチューニング表**であり、機構は共通）。

## 4. 型: `BBox<'a>` / `Boxes` — 境界箱

```rust
struct BBox<'a> { lo: &'a [f64], hi: &'a [f64] }
struct BBoxMut<'a> { lo: &'a mut [f64], hi: &'a mut [f64] }
struct Boxes { values: Vec<f64>, ndim: usize }   // Boxes::of(id) -> BBox<'_>
```

- **削除されるもの**: `split_box` / `split_box_mut` / `Tree::box_of` / `Tree::root_box` と、
  `2 * ndim * id` のアドレス計算 5 箇所、`(lo, hi)` タプルの引き回し。
- `Metric::box_dist(q, bbox) -> Dist`、`Rows::bbox_into(BBoxMut)`、
  `shrank_off_axis(parent: BBox, child: BBox, ..)` が全部素直に書ける。

## 5. 抽象: `Sink` — 葉走査の受け側

現在は `impl FnMut(usize, f64) -> f64`（「新しい bound を返すコールバック」）。
これを名前のある trait にする:

```rust
trait Sink {
    fn bound(&self) -> Dist;                 // 現在の枝刈り境界
    fn offer(&mut self, offset: usize, dist: Dist);
}
```

- 「戻り値が新しい bound」という暗黙の規約が、`bound()` という明示的な問い合わせになる。
- 実装は 1 個（`Best` + leaf の開始位置 + `indices` マップ）。
- **工夫 33 を壊さないこと**: `bound()` は「走査開始時に 1 回 + hit 1 件につき
  1 回」だけ呼ぶ。点ごとに呼ぶと、以前の実装が直した「点ごとに k-best を
  再問い合わせする」形に戻る。`Scan` が bound をキャッシュし、`Scan::hit` の中
  でだけ更新することで規律を型に閉じる。機械的な検査は
  `grep -c 'sink.bound()' src/kernel.rs` が 2 であること。

## 6. 抽象: `Strategy` / `Scan` — 葉走査

現行 `kernel.rs` の 9 アーム match の正体はこれ。可変なのは
**「1 ステップに何点畳むか」と「行の途中で抜けられるか」**の 2 点だけ。

| 戦略 | 適用 | 分岐 |
| --- | --- | --- |
| `Packed` / `Scan::packed` | d2, d3, d4, d8 | 分岐なし、chunk ごとのベクトル gate |
| `Packed` / `Scan::unrolled` | d1, d5, d6, d7 | 分岐なし、LLVM が点方向に自動ベクトル化 |
| `Streamed` | d >= 9、および全 `L^p` | 行の途中で early exit |

### 戦略は葉ごとではなく**クエリ呼び出しごと**に決める

これは性能チューニングに見えて、実はドメインの事実である:
**どの戦略を使うかは `(metric, 行幅)` だけで決まり、両方 1 回の `query` 呼び出しの
間ずっと不変**。だから葉ごとに match するのは、不変な決定を毎回やり直している。

`Descent<K: Strategy>` として呼び出しごとに解決すると、葉ごとの分岐が消えるだけ
でなく、**各 descent の機械語がその 1 戦略ぶんだけになる**。9 戦略を全部 1 個の
`descend` に inline すると、d3 のクエリは絶対に通らない `Streamed`（`point_dist` の
SIMD 3 アームを含み、最大）のぶんまで I-cache を負担する。逆に `Streamed` だけを
関数外に出すと今度は d16 が損をする。どちらか一方を選ぶ問題ではなく、
**分割の粒度が間違っている**というのが正解だった。

- ドライバ（chunk ループ、ベクトル gate、hit のばら撒き、端数の tail）は
  `packed` に 1 つだけ。`d2` / `d3` / `d4` / `d8` は**制御フローを一切持たない
  swizzle レシピ**になる。
- `(m, q, flat, bound, sink)` を `Scan` に束ねることで、7 個のシグネチャを
  貫通していた 5 引数の引き回しが消える。
- 幅 → 戦略の対応は `scan_leaf` の**1 個の表**。
- **「早期脱出を持つ経路と持たない経路がある」ことを抽象に残す**。これを
  「1 つの一様な kernel」に潰すと 40 番・41 番の工夫が消える。

`Packing` を trait にして `CHUNK` / `POINTS` を関連定数にする案は採らなかった。
`&[f64; Self::CHUNK]` や `Simd<f64, Self::POINTS>` を書くには
`generic_const_exprs`（未完成の unstable feature）が要る。const を trait の
ジェネリック引数にすれば書けるが、それは結局呼び出し側で `::<CHUNK, P, D>` を
書くことになり、今のクロージャ版と情報量が変わらない。

## 7. 型: `CellBound` — 分割平面下界のスタック規律

```rust
struct CellBound { axes: Vec<Dist>, seeded: bool }
impl CellBound {
    fn start(&mut self, m: Metric, q: &[f64], root: BBox) -> Dist;  // 内側なら 0 で即返す
    fn finish(&mut self);                                          // seeded のときだけゼロ戻し
    fn axis(&self, dim: usize) -> Dist;
    #[must_use] fn swap_axis(&mut self, dim: usize, axis: Dist) -> Dist;
}
```

- 「クエリ間で全ゼロ」という不変条件が 1 つの型に閉じる（今は
  `Scratch::cell` のコメント、`Descent::run`、`seed_cell`、`enter_far` の 4 箇所）。
- fast path（root box の内側）は `start` の内部実装になり、`run` から分岐が消える。
- **畳み込み後の合計 `cell_dist` は再帰引数のまま残す**。これは
  「このフレームのノードの下界」というフレームローカルな値で、`Scratch` の
  メモリに置くと再帰呼び出し後にリロードが出る（工夫 34 と同じ理由）。

### 破ってはいけない不変条件

> **box 距離は gate にしか使わない。分割平面下界の代数（`cell` と `cell_dist`）には
> 絶対に混ぜない。**

`descend_by_box` で近い方の子に降りるとき、渡すのは親の `cell_dist` であって
`near_box` ではない。ここを「2 つの下界を 1 個の引数に統一する」と綺麗に見えるが、
枝刈りが壊れる（`cell` との整合が崩れて以降の `replace_axis` が嘘になる）。

## 8. `Node` / `Tree`

```rust
enum Node {
    Leaf  { start: u32, end: u32 },
    Inner { right: u32, split_dim: u32, order_by_box: bool, split_value: f64 },
}
struct Tree { rows: Vec<f64>, ndim: usize, indices: Vec<u32>, nodes: Vec<Node>, boxes: Boxes, leafsize: usize }
```

- preorder なので `left == id + 1`。フィールドを持たせる必要はない。
  ただし **`f64` のアラインメント上サイズは 24 バイトのまま**なので性能利得はゼロ
  （`u32 * 3 + bool + discriminant` は 16 バイト枠に収まり、`left` を消しても
  枠が減らない）。「preorder レイアウトである」ことが構造的に保証される、という
  可読性だけの変更なので、今回は入れていない。
- `right` の方は導出できない。node は自分の点数を持たないので
  `count_nodes` を再評価できず、`right = id + 1 + left_nodes` を保存する必要がある。

## 9. モジュール構成

抽象が正しく取れていればファイル分割は自明に決まる（そして、それ自体は本質ではない）。

```
metric.rs   Dist, Metric                            距離の代数
layout.rs   Width, Rows, RowsMut, BBox, Boxes     レイアウトの語彙
simd.rs     LANES, F64s, vmin/vmax/hsum/hmax      SIMD プリミティブ
kernel.rs   bbox, point_dist, box_dist, Sink, Scan    バルクループ
tree.rs     Node, Tree                            記憶レイアウト
build.rs    Subtree の切り出しと再帰              唯一の writer
query.rs    CellBound, Descent, Best              分枝限定
lib.rs      PyO3 境界
```

## 10. 実装順序（各段でベンチを取る）

一括書き換えはしない。負の計測結果が多すぎるので、一度に変えると
「どれが遅くした変更か」の信号が消える。

**Stage A** — `Dist` + `Width` / `Rows` / `BBox` / `Boxes`。型と語彙だけ。
`const D` と `(data, ndim)` を一掃する。

**Stage B** — `Sink` + `Scan` の 3 戦略。kernel の再構成。

**Stage C** — `CellBound`。descent の不変条件を型に閉じる。

**Stage D（任意）** — `Node.left` の削除、ベクトル metric の型レベル化。

### 計測の注意

- **`tests/benchmark.py` は d3 / d8 / d16 しか回さない。** つまり葉走査 9 経路のうち
  d1 / d2 / d4 / d5 / d6 / d7 の 6 つは**性能計測がゼロ**（正しさは
  `row-width-NN` ケースが見ている）。kernel を触る Stage B 以降は、
  ndim 1..18 を掃く別スクリプトで測ること。
- 最終的な合否は、HEAD と最終形を**連続して**測って出すこと。
  ビルド負荷や thermal で median は数 % 動く。

### Stage D について

工夫 44 の `unreachable!` を消す綺麗な方法は「ベクトル metric を ZST の型引数にする」
だが、これは `scan_leaf` の幅テーブルを 3 倍実体化するので I-cache を食う。
負の計測結果 G-1（3 variant enum で 10-25% 遅い）は**ランタイム enum**の話であって
型レベル化とは別物だが、リスクは同種。**必ず A/B すること。**
回帰したら `Metric` を引数に取る現行形に戻し、`unreachable!` は
「LP は入口で振り分け済み」というコメント付きで残す。

## 11. 受け入れ基準

- `uv run pytest tests/test_kdtree.py` が pass すること。
  `test_compatibility` は seed 固定なしの hypothesis 100 例なので、
  **複数回**回すこと。特に index の同着解決（小さい元 index を優先）は
  `assert_array_equal` で見られている硬い契約。
- `cargo test` も回すこと。pytest では走らない 2 つの契約がある:
  - `build_allocation_count_does_not_scale_with_n`（< 20 allocations）
  - `node_stays_within_one_cache_slot`（`size_of::<Node>() <= 24`）
- `uv run pytest tests/benchmark.py` が現行以上であること。

### 2026-08-14 時点の実測（median、head と最終形を交互 2 ラウンド、各辺の min）

vs SciPy（同一セッションで測った SciPy を基準）:

| | head | 最終形 |
| --- | --- | --- |
| build | 0.94x - 2.62x | 0.92x - 2.64x |
| query | 1.60x - 23.4x | 1.67x - 22.8x |
| 3x 未満のグループ | 21 / 57 | 23 / 57 |

**「build も query も SciPy 比 3x」は元から未達**であり、リファクタで達成できる
性質のものでもない（build は最良でも 2.6x、`d16-sorted` では SciPy より遅い）。
リファクタの受け入れ基準は「同等以上」であり、3x は別の最適化課題。

head 対 最終形（57 グループ）: geomean **1.012**、best 0.906、worst 1.129。
5% 超の改善 4 グループ / 5% 超の悪化 7 グループ。悪化はすべて 137-2300us の
小次元・直列クエリに集中している。

### 計測ノイズについて（重要）

このリポジトリのベンチには **50-200us のグループが多数ある**。同じバイナリを
交互に測っても、その帯域では ±10% 程度ぶれる。実際、query しか触っていない
変更で `build-d12` が 14% 動いた記録がある。したがって:

- **単発の実行で 10% 以下の差を判断してはいけない。**
- head/new を**交互に複数ラウンド**回し、各辺の min を取る。それでも
  1 桁 % の差は「ノイズ帯」として扱う。
- 因果的にありえない場所（触っていないコード）が動いていたら、それは
  マシンの状態が変わった証拠なので、その回の結果を全部捨てる。
