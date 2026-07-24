#!/usr/bin/env bash
#
# run_kyutai_benchmark.sh
# ------------------------
# One command to: install prerequisites (uv, Rust/cargo, build deps),
# install moshi-server (Rust, CUDA build), fetch the STT config from
# kyutai-labs/delayed-streams-modeling, start the server, wait for it
# to be ready, run benchmark_kyutai_streaming.py against it, then
# shut the server down.
#
# Usage:
#   ./run_kyutai_benchmark.sh --dataset dataset/test_set_frozen.json --audio-dir audio_wav/ [any other benchmark_kyutai_streaming.py flags...]
#
# Assumes benchmark_kyutai_streaming.py, normalizer.py, metrics.py are
# already on this machine (you said you'd handle moving those).
#
# Env vars you can override:
#   MOSHI_PORT       (default 8080)
#   MOSHI_CONFIG     (default: fetched configs/config-stt-en_fr-hf.toml from kyutai repo)
#   DSM_REPO_DIR     (default ./delayed-streams-modeling — where the config gets cloned)
#   BENCHMARK_SCRIPT (default ./benchmark_kyutai_streaming.py)
#   HF_TOKEN          set this if the model repo requires auth
#   READY_TIMEOUT_S   how long to wait for the server to come up (default 900s — first run downloads weights)

set -euo pipefail

MOSHI_PORT="${MOSHI_PORT:-8080}"
DSM_REPO_DIR="${DSM_REPO_DIR:-$PWD/delayed-streams-modeling}"
MOSHI_CONFIG="${MOSHI_CONFIG:-$DSM_REPO_DIR/configs/config-stt-en_fr-hf.toml}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-$PWD/benchmark_kyutai_streaming.py}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
SERVER_LOG="$PWD/moshi-server.log"

log() { printf '\n[run_kyutai_benchmark] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. System build deps (needed to compile moshi-server with CUDA support)
# ---------------------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  log "nvcc not found — installing build prerequisites via apt (needs sudo)."
  sudo apt-get update -y
  sudo apt-get install -y build-essential pkg-config libssl-dev clang curl git nvidia-cuda-toolkit
else
  log "nvcc found: $(command -v nvcc)"
fi

# ---------------------------------------------------------------------------
# 2. uv (runs the python benchmark script + its inline deps)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  log "uv found: $(command -v uv)"
fi

# ---------------------------------------------------------------------------
# 3. Rust / cargo
# ---------------------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  log "Installing Rust toolchain via rustup..."
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  source "$HOME/.cargo/env"
else
  log "cargo found: $(command -v cargo)"
  source "$HOME/.cargo/env" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4. moshi-server itself (CUDA build)
# ---------------------------------------------------------------------------
if ! command -v moshi-server >/dev/null 2>&1; then
  log "Installing moshi-server (cargo install --features cuda moshi-server)... this compiles from source and can take several minutes."
  cargo install --features cuda moshi-server
  export PATH="$HOME/.cargo/bin:$PATH"
else
  log "moshi-server found: $(command -v moshi-server)"
fi

# ---------------------------------------------------------------------------
# 5. Config file — clone the kyutai repo (shallow) if we don't have it
# ---------------------------------------------------------------------------
if [ ! -f "$MOSHI_CONFIG" ]; then
  log "Fetching STT config from kyutai-labs/delayed-streams-modeling..."
  if [ -d "$DSM_REPO_DIR/.git" ]; then
    git -C "$DSM_REPO_DIR" pull --ff-only
  else
    git clone --depth 1 https://github.com/kyutai-labs/delayed-streams-modeling "$DSM_REPO_DIR"
  fi
fi

if [ ! -f "$MOSHI_CONFIG" ]; then
  log "ERROR: config not found at $MOSHI_CONFIG after clone. Check DSM_REPO_DIR/MOSHI_CONFIG."
  exit 1
fi
log "Using config: $MOSHI_CONFIG"

# ---------------------------------------------------------------------------
# 6. Start moshi-server in the background
# ---------------------------------------------------------------------------
log "Starting moshi-server on port $MOSHI_PORT (logging to $SERVER_LOG)..."
: > "$SERVER_LOG"
moshi-server worker --config "$MOSHI_CONFIG" --port "$MOSHI_PORT" >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping moshi-server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 7. Wait until the server is accepting connections
#    (first run also downloads model weights from HF, hence the long default timeout)
# ---------------------------------------------------------------------------
log "Waiting for moshi-server to become ready (timeout ${READY_TIMEOUT_S}s)..."
elapsed=0
until (exec 3<>"/dev/tcp/127.0.0.1/$MOSHI_PORT") 2>/dev/null; do
  exec 3>&- 2>/dev/null || true
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "ERROR: moshi-server process died. Last log lines:"
    tail -n 40 "$SERVER_LOG"
    exit 1
  fi
  if [ "$elapsed" -ge "$READY_TIMEOUT_S" ]; then
    log "ERROR: timed out waiting for moshi-server on port $MOSHI_PORT. Last log lines:"
    tail -n 40 "$SERVER_LOG"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  if [ $((elapsed % 30)) -eq 0 ]; then
    log "...still waiting (${elapsed}s elapsed). Tail of server log:"
    tail -n 5 "$SERVER_LOG"
  fi
done
exec 3>&- 2>/dev/null || true
log "moshi-server is accepting connections."

# ---------------------------------------------------------------------------
# 8. Run the benchmark, forwarding all extra CLI args through
# ---------------------------------------------------------------------------
log "Running benchmark: $BENCHMARK_SCRIPT $*"
uv run "$BENCHMARK_SCRIPT" --url "ws://127.0.0.1:$MOSHI_PORT" "$@"
BENCH_EXIT=$?

log "Benchmark finished with exit code $BENCH_EXIT."
exit $BENCH_EXIT