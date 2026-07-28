#!/usr/bin/env bash
#
# run_selected_benchmarks.sh
# ----------------------------
# Runs the specific set of docker compose services/scripts you listed,
# in order, one at a time. Continues past failures instead of stopping
# (set -uo pipefail, deliberately no -e), logs each to logs/<name>.log,
# and clears GPU memory between runs.
#
# Usage: ./run_selected_benchmarks.sh

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

run() {
  local name="$1"; shift
  echo "============================================================"
  echo "Running: $name"
  echo "  $ $*"
  echo "============================================================"
  t0=$(date +%s)
  if "$@" 2>&1 | tee "logs/$name.log"; then
    echo ">>> $name: OK ($(( $(date +%s) - t0 ))s)"
  else
    echo ">>> $name: FAILED ($(( $(date +%s) - t0 ))s)"
  fi
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  echo
}

run "faster_whisper"                docker compose run --rm faster_whisper
run "faster_whisper_chunked"        docker compose run --rm faster_whisper python notebook_faster_whisper_chunked.py

run "faster_whisper_int8"           docker compose run --rm faster_whisper_int8
run "faster_whisper_int8_chunked"   docker compose run --rm faster_whisper_int8 python notebook_faster_whisper_int8_chunked.py

run "faster_whisper_turbo"          docker compose run --rm faster_whisper_turbo
run "faster_whisper_turbo_chunked"  docker compose run --rm faster_whisper_turbo python notebook_faster_whisper_turbo_int8_chunked.py

run "voxtral_mini"                  docker compose run --rm voxtral_mini

run "voxtral_small"                 docker compose run --rm voxtral_small
run "voxtral_small_chunked"         docker compose run --rm voxtral_small python notebook_voxtral_small_chunked.py

run "whisper_large"                 docker compose run --rm whisper_large

run "whisperx"                      docker compose run --rm whisperx
run "whisperx_chunked"              docker compose run --rm whisperx python notebook_whisperx_chunked.py

echo "All done. Logs in ./logs/, results should be under ./results/ as usual."
