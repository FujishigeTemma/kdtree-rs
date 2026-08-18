#!/usr/bin/env bash
#
#   ./scripts/paritycheck.sh   # benches/grid.rs ids == tests/benchmark.py ids
#
# The grid mirrors benchmark.py's ids character for character; this is the
# check that the two sets actually correspond. Parallel builds are the one
# documented exception (grid.rs is a superset there).
set -euo pipefail

here=$(dirname "$0")

bin=${BIN:-$("$here/binpath.sh")}

rust_ids=$("$bin" --bench --list | grep -v '^build.*-parallel$' | sort)
py_ids=$(cd "$here/.." && uv run pytest tests/benchmark.py --collect-only -q \
  | sed -n 's/.*\[\(.*\)-kdtree\]$/\1/p' | sort -u)

diff <(printf '%s\n' "$py_ids") <(printf '%s\n' "$rust_ids") \
  && echo "paritycheck: ids match"
