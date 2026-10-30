#!/bin/zsh -l
# Run both benchmark backends on this machine and log the output.
# Results land in benchmarks/results/ as JSONL; the full console log goes to
# benchmarks/results/<timestamp>-bench.log.
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
LOG="benchmarks/results/$(date +%Y-%m-%dT%H%M%S)-bench.log"
{
  echo "== machine"; sysctl -n machdep.cpu.brand_string; echo "$(( $(sysctl -n hw.memsize) / 1073741824 )) GB"; sw_vers -productVersion
  echo "== git"; git rev-parse --short HEAD
  echo "== sync"; uv sync --group bench
  echo "== native"; uv run --group bench python -m benchmarks.run
  echo "== aer"; uv run --group bench python -m benchmarks.run_aer
  echo "== done"
} 2>&1 | tee "$LOG"
