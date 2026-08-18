#!/usr/bin/env bash
#
#   ./scripts/perfstat.sh build-d8-n10000-serial
#   ./scripts/perfstat.sh query-d16-n10000-q1000-leaf128-p2-serial 20
#   BIN=~/ab/a CPU=8 ./scripts/perfstat.sh build-d8-n10000-serial
#
# See BENCHMARK.md. Linux only.
set -euo pipefail

here=$(dirname "$0")

id=${1:?perfstat needs a workload id, e.g. build-d8-n10000-serial}
iters=${2:-200}
cpu=${CPU:-4}

# `ab.sh` forwards its own arguments here, so a stray third one is a round count
# aimed at the version that took one positionally. Silently ignoring it would
# measure a different number of iterations than the caller asked for.
[ $# -le 2 ] || { echo "perfstat: unexpected argument '$3' (ab.sh rounds are ROUNDS=)" >&2; exit 2; }

bin=${BIN:-$("$here/binpath.sh")}
# `BIN=`: resolve the id against the binary being measured, not a fresh build.
id=$(BIN=$bin "$here/benchid.sh" "$id")

case $id in
  *-parallel) echo "$id: not measurable pinned to one core" >&2; exit 2 ;;
esac

# `2>&1 >/dev/null` in that order: perf writes its counters to stderr, which has
# to reach the pipe, while the benchmark's own stdout is discarded.
taskset -c "$cpu" perf stat -x, \
  -e cycles:u,instructions:u,branches:u,branch-misses:u,cache-misses:u \
  "$bin" --iters "$iters" "$id" 2>&1 >/dev/null \
  | awk -F, -v n="$iters" -v id="$id" -v cpu="$cpu" '
      $3 ~ /cycles/         { c = $1 }
      $3 ~ /instructions/   { i = $1 }
      $3 == "branches:u"    { b = $1 }
      $3 ~ /branch-misses/  { bm = $1 }
      $3 ~ /cache-misses/   { cm = $1 }
      END {
        # `<not counted>` coerces to 0 and every ratio would print as nan.
        if (c == 0 || b == 0) { print "perf counters unavailable on this host" > "/dev/stderr"; exit 1 }
        # One `key=value` per whitespace-separated field, padded on the right so
        # that stacked rounds still line up: `ab.sh` reads cycles out of this.
        printf "%-46s %-16s %-16s %-9s %-22s %-16s n=%s cpu=%s\n", id,
               sprintf("cyc=%.0f", c / n), sprintf("ins=%.0f", i / n),
               sprintf("ipc=%.2f", i / c),
               sprintf("bmiss=%.0f(%.2f%%)", bm / n, 100 * bm / b),
               sprintf("cmiss=%.0f", cm / n), n, cpu
      }'
