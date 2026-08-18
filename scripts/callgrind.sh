#!/usr/bin/env bash
#
#   ./scripts/callgrind.sh build-d8-n10000-serial
#   ./scripts/callgrind.sh build-d8-n10000-serial --branch-sim=yes
#   BIN=~/ab/a ./scripts/callgrind.sh build-d8-n10000-serial
#
# See BENCHMARK.md. Linux only.
set -euo pipefail

here=$(dirname "$0")

id=${1:?callgrind needs a workload id, e.g. build-d8-n10000-serial}
shift

bin=${BIN:-$("$here/binpath.sh")}
id=$(BIN=$bin "$here/benchid.sh" "$id")

out=$(cd "$here/.." && pwd)/target/callgrind/$id.$(basename "$bin").out
mkdir -p "$(dirname "$out")"

valgrind --tool=callgrind \
  --collect-atstart=no \
  --toggle-collect=kdtree_bench_target \
  --callgrind-out-file="$out" \
  "$@" \
  "$bin" --iters 1 "$id"

callgrind_annotate --auto=no "$out"
