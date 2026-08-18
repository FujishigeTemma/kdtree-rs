#!/usr/bin/env bash
#
#   ./scripts/benchid.sh build-d8          # a substring -> the one full id
#   BIN=~/ab/a ./scripts/benchid.sh build-d8
#
# See BENCHMARK.md. Linux only.
set -euo pipefail

here=$(dirname "$0")

id=${1:?benchid needs a workload id or a substring of one}
bin=${BIN:-$("$here/binpath.sh")}

matched=$("$bin" --list "$id") ||
  { echo "$id: no workload matched, or the bench failed to run" >&2; exit 2; }
if [ "$(printf '%s\n' "$matched" | grep -c .)" -ne 1 ]; then
  echo "$id: matches more than one workload" >&2
  exit 2
fi
printf '%s\n' "$matched"
