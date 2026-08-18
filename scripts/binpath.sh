#!/usr/bin/env bash
#
#   ./scripts/binpath.sh          # benches/grid.rs
#   ./scripts/binpath.sh --test   # the --lib test binary, for memcheck.sh
#
# See BENCHMARK.md. Linux only.
set -euo pipefail

if [ "${1:-}" = --test ]; then
  build=(test --lib)
  name=kdtree-
else
  build=(bench --bench grid)
  name=grid-
fi

# cargo emits one JSON line per artifact; the last `executable` naming the crate
# is the binary it just built. `json-render-diagnostics` keeps that machine
# stream on stdout while a failed build still reports itself on stderr — plain
# `--message-format=json` would put the errors in the stream the grep eats.
bin=$(cargo "${build[@]}" --no-run --message-format=json-render-diagnostics \
  | grep -o "\"executable\":\"[^\"]*$name[^\"]*\"" | tail -1 | cut -d'"' -f4) || true

if [ ! -x "$bin" ]; then
  echo "binpath: cargo ${build[*]} produced no $name executable" >&2
  exit 1
fi
printf '%s\n' "$bin"
