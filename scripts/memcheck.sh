#!/usr/bin/env bash
#
#   ./scripts/memcheck.sh
#   ./scripts/memcheck.sh build::tests::parallel_build_matches_serial_build
#
# See BENCHMARK.md. Linux only.
set -euo pipefail

here=$(dirname "$0")

bin=$("$here/binpath.sh" --test)

RAYON_NUM_THREADS=1 valgrind --tool=memcheck \
  --leak-check=full \
  --errors-for-leak-kinds=definite \
  --error-exitcode=1 \
  "$bin" --test-threads=1 "$@"
