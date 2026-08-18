#!/usr/bin/env bash
#
#   ./scripts/ab.sh stash a                       # build the current source as A
#   ./scripts/ab.sh stash b                       # ... then edit src/ and stash B
#   ./scripts/ab.sh build-d8-n10000-serial        # alternate 3 rounds, min per side
#   ROUNDS=5 ./scripts/ab.sh query-d16-n10000-q1000-leaf128-p2-serial 20
#
# See BENCHMARK.md. Linux only.
set -euo pipefail

here=$(dirname "$0")
dir=${AB_DIR:-$HOME/ab}

if [ "${1:-}" = stash ]; then
  name=${2:?stash needs a name, e.g. "stash a"}
  mkdir -p "$dir"
  # Remove first: a failed build would otherwise leave the previous binary in
  # place, and the next comparison would silently pit two different sources.
  rm -f "$dir/$name"
  cp "$("$here/binpath.sh")" "$dir/$name"
  exit
fi

[ $# -gt 0 ] || { echo "usage: $0 <workload id> [iters]  |  $0 stash <a|b>" >&2; exit 2; }

for v in a b; do
  [ -x "$dir/$v" ] || { echo "$dir/$v: run '$0 stash $v' first" >&2; exit 2; }
  printf '%s  %s\n' "$dir/$v" "$(date -r "$dir/$v" '+%F %T')" >&2
done
cmp -s "$dir/a" "$dir/b" && echo "warning: a and b are the same binary" >&2

runs=
for _ in $(seq "${ROUNDS:-3}"); do
  for v in a b; do
    line=$(BIN=$dir/$v "$here/perfstat.sh" "$@")
    printf '%s %s\n' "$v" "$line" >&2
    runs+="$v $line"$'\n'
  done
done

printf '%s' "$runs" | awk -v band=0.03 '
    {
      c = ""
      for (f = 3; f <= NF; f++) if ($f ~ /^cyc=/) { split($f, kv, "="); c = kv[2] }
      if (c == "") next
      id = $2
      if (!($1 in min) || c + 0 < min[$1]) min[$1] = c + 0
    }
    END {
      if (!("a" in min) || !("b" in min)) { print "no cycles parsed" > "/dev/stderr"; exit 1 }
      r = min["b"] / min["a"]
      printf "%s  min cyc  a=%.0f  b=%.0f  b/a=%.3f%s\n", id, min["a"], min["b"], r,
             (r > 1 - band && r < 1 + band) ? "  (within layout noise)" : ""
    }'
