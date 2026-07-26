# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
#     "sphn",
#     "websockets",
#     "pandas",
#     "jiwer",
# ]
# ///
"""
benchmark_kyutai_streaming.py — Benchmark: Kyutai STT 1B (en/fr), REAL streaming, Rust server
=================================================================================================
This wraps the exact same WebSocket protocol as the official
scripts/stt_from_file_rust_server.py (same message types, same audio framing,
same silence-padding/marker handshake) so the connection logic is proven, not
guessed — only the metrics/CSV layer is new.

Unlike every other notebook in this benchmark, this one talks to GENUINE
Kyutai streaming inference: moshi-server (Rust), running locally on your GPU,
fed audio at a configurable real-time factor (rtf). This is NOT the
transformers one-shot batch call used elsewhere.

Requires:
  - moshi-server already running in another terminal:
      moshi-server worker --config configs/config-stt-en_fr-hf.toml
  - Audio files converted to WAV first if they're .m4a/.mp3 (sphn's codec
    support for those formats is unreliable — see chat history). Use:
      ffmpeg -i input.m4a -ar 24000 -ac 1 output.wav

What's measured, and how it differs from your batch-model notebooks:
  - wall_clock_latency_s : total time from connection open to receiving the
    final Marker back (includes the rtf-paced send time, NOT just compute
    time — this is intentional, since at rtf~1.0 this approximates "how long
    you'd wait in a real live conversation before the full transcript settles")
  - first_word_latency_s : time from stream start to the FIRST word received
    — this is the actual "does it feel live" number, not available at all
    for your batch models since they only return one complete result
  - mean_word_latency_s  : average per-word gap, a rough proxy for the
    model's ~500ms streaming delay claim

Run:
    uv run benchmark_kyutai_streaming.py --dataset dataset/test_set_frozen.json --audio-dir audio_wav/
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import msgpack
import pandas as pd
import sphn
import websockets

sys.path.insert(0, "/app/src")  # adjust if your normalizer/metrics live elsewhere
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

SAMPLE_RATE = 24000
FRAME_SIZE = 1920


def load_and_process_audio(file_path: str):
    pcm_data, _ = sphn.read(file_path, sample_rate=SAMPLE_RATE)
    return pcm_data[0]


async def receive_messages(websocket, timing: dict):
    """Same message handling as the official script, plus timing capture."""
    transcript = []
    t0 = timing["stream_start"]

    async for message in websocket:
        data = msgpack.unpackb(message, raw=False)

        if data["type"] == "Step":
            continue

        if data["type"] == "Word":
            now = time.time()
            if "first_word_time" not in timing:
                timing["first_word_time"] = now
            timing.setdefault("word_recv_times", []).append(now)
            transcript.append({
                "text": data["text"],
                "timestamp": [data["start_time"], data["start_time"]],
            })

        if data["type"] == "EndWord":
            if len(transcript) > 0:
                transcript[-1]["timestamp"][1] = data["stop_time"]

        if data["type"] == "Marker":
            timing["end_time"] = time.time()
            break

    return transcript


async def send_messages(websocket, in_file: str, rtf: float, duration_s: float = 0.0):
    audio_data = load_and_process_audio(in_file)

    async def send_audio(audio):
        await websocket.send(
            msgpack.packb({"type": "Audio", "pcm": [float(x) for x in audio]}, use_single_float=True)
        )

    await send_audio([0.0] * SAMPLE_RATE)

    start_time = time.time()
    for i in range(0, len(audio_data), FRAME_SIZE):
        await send_audio(audio_data[i: i + FRAME_SIZE])
        expected_send_time = start_time + (i + 1) / SAMPLE_RATE / rtf
        current_time = time.time()
        if current_time < expected_send_time:
            await asyncio.sleep(expected_send_time - current_time)
        else:
            await asyncio.sleep(0.001)

    # Give the decoder time to drain its backlog BEFORE sending the Marker.
    # Earlier we scaled the *post*-Marker silence with duration and it made
    # no difference — the server logs show recv_loop finishing right as the
    # Marker is processed, not when audio input runs out. That means
    # everything sent after the Marker is irrelevant to how much time
    # send_loop gets: recv_loop is already done by then, which is exactly
    # what triggers the premature close/backlog-drop race on long files.
    # So the real fix is padding BEFORE the Marker, giving the decoder a
    # chance to actually catch up while recv_loop is still legitimately busy.
    pre_marker_silence_s = min(150, max(5, int(duration_s * 0.2)))
    for _ in range(pre_marker_silence_s):
        await send_audio([0.0] * SAMPLE_RATE)

    await websocket.send(msgpack.packb({"type": "Marker", "id": 0}, use_single_float=True))

    # Small buffer after the Marker too, in case recv_loop's completion isn't
    # purely Marker-triggered on every server version — cheap insurance.
    for _ in range(10):
        await send_audio([0.0] * SAMPLE_RATE)


async def stream_audio(url: str, api_key: str, in_file: str, rtf: float, duration_s: float = 0.0):
    headers = {"kyutai-api-key": api_key}
    timing = {"stream_start": time.time()}

    async with websockets.connect(url, additional_headers=headers) as websocket:
        send_task = asyncio.create_task(send_messages(websocket, in_file, rtf, duration_s))
        receive_task = asyncio.create_task(receive_messages(websocket, timing))
        _, transcript = await asyncio.gather(send_task, receive_task)

    return transcript, timing


def compute_timing_metrics(timing: dict) -> dict:
    """Derive the streaming-specific numbers from raw timestamps."""
    start = timing["stream_start"]
    end = timing.get("end_time")
    first_word = timing.get("first_word_time")
    word_times = timing.get("word_recv_times", [])

    wall_clock_latency_s = (end - start) if end else None
    first_word_latency_s = (first_word - start) if first_word else None

    mean_word_gap_s = None
    if len(word_times) > 1:
        gaps = [word_times[i] - word_times[i - 1] for i in range(1, len(word_times))]
        mean_word_gap_s = sum(gaps) / len(gaps)

    return {
        "wall_clock_latency_s": round(wall_clock_latency_s, 3) if wall_clock_latency_s else None,
        "first_word_latency_s": round(first_word_latency_s, 3) if first_word_latency_s else None,
        "mean_word_gap_s": round(mean_word_gap_s, 3) if mean_word_gap_s else None,
        "n_words_received": len(word_times),
    }


async def run_one_file(url, api_key, audio_path, rtf, duration_s: float = 0.0):
    transcript, timing = await stream_audio(url, api_key, str(audio_path), rtf, duration_s)
    full_text = " ".join(w["text"] for w in transcript).strip()
    timing_metrics = compute_timing_metrics(timing)
    return full_text, timing_metrics


def looks_truncated(timing_metrics: dict, duration_s: float) -> bool:
    """
    Heuristic for the 'send_loop died mid-flight, backlog lost' failure mode:
    the stream ran close to the full audio duration (so it wasn't a fast,
    clean disconnect) but produced far too few words to be real speech.
    Only flags recordings long enough that a false positive is very unlikely
    (a short, quiet segment could legitimately have few words).
    """
    n_words = timing_metrics.get("n_words_received", 0)
    wall_clock = timing_metrics.get("wall_clock_latency_s") or 0
    if duration_s < 60:
        return False
    ran_nearly_full_length = wall_clock >= duration_s * 0.8
    far_too_few_words = n_words < max(20, duration_s / 15)
    return ran_nearly_full_length and far_too_few_words


def is_channel_exhaustion_error(exc: Exception) -> bool:
    """
    moshi-server 0.6.4 has a race where 'send_loop' dying from the
    Sending-after-closing error leaks its channel slot instead of releasing
    it back to the worker pool. The next connection attempt then fails with
    'no free channels' server-side, which surfaces client-side as an
    abnormal close (websockets code 1005, no status received). Retrying
    against an exhausted pool is pointless — only a server restart clears
    the leaked channels — so we detect this specific failure and stop
    immediately instead of wasting the remaining retry attempts.
    """
    msg = str(exc)
    return "1005" in msg or "no status received" in msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to test_set_frozen.json (same format as your other notebooks)")
    parser.add_argument("--audio-dir", required=True, help="Directory containing WAV versions of the audio files (convert m4a->wav first)")
    parser.add_argument("--url", default="ws://127.0.0.1:8080")
    parser.add_argument("--api-key", default="public_token")
    parser.add_argument("--rtf", type=float, default=1.01, help="1.01 = realistic live-mic pace; use a large value (e.g. 1000) for max-speed batch-style timing instead")
    parser.add_argument("--results-path", default="results/results_kyutai_streaming_rust.csv")
    parser.add_argument("--segment-id", default=None,
                         help="Run only these segment_id(s) (for fast iteration while tuning "
                              "server config). Comma-separated for multiple, e.g. "
                              "seg_0005,seg_0006")
    parser.add_argument("--max-retries", type=int, default=2,
                         help="Retries if the stream appears to have dropped its backlog "
                              "mid-flight (ran the full duration but returned far too few "
                              "words). Default 2. Automatically skipped if the server's "
                              "channel pool looks exhausted, since retrying that is futile.")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    segments = dataset["segments"]
    if args.segment_id:
        wanted = {s.strip() for s in args.segment_id.split(",") if s.strip()}
        segments = [s for s in segments if s["segment_id"] in wanted]
        found = {s["segment_id"] for s in segments}
        missing = wanted - found
        if missing:
            print(f"Warning: no segment found for segment_id(s): {sorted(missing)}")
        if not segments:
            print(f"No segments found for segment_id={args.segment_id}")
            return
    fingerprint = dataset["dataset_fingerprint"]

    norm = MedicalNormalizer()
    metrics = BenchmarkMetrics()
    audio_dir = Path(args.audio_dir)
    url = f"{args.url}/api/asr-streaming"

    records = []
    for idx, seg in enumerate(segments):
        seg_id = seg["segment_id"]
        duration_s = seg["duration_s"]
        gt_norm = seg["ground_truth_normalized"]
        # Expect a .wav with the same stem as the original audio_file
        wav_name = Path(seg["audio_file"]).stem + ".wav"
        audio_path = audio_dir / wav_name

        print(f"[{idx+1}/{len(segments)}] {seg_id} — {wav_name}")

        if not audio_path.exists():
            print(f"  WAV not found: {audio_path} — skipping (convert with ffmpeg first)")
            continue

        try:
            raw_text, timing_metrics = asyncio.run(
                run_one_file(url, args.api_key, audio_path, args.rtf, duration_s)
            )
        except Exception as e:
            print(f"  Streaming failed: {e}")
            raw_text, timing_metrics = "[ERROR]", {}

        attempt = 1
        server_needs_restart = False
        while looks_truncated(timing_metrics, duration_s) and attempt <= args.max_retries:
            print(f"  ⚠️  Only {timing_metrics.get('n_words_received', 0)} words for "
                  f"{duration_s:.0f}s of audio (ran {timing_metrics.get('wall_clock_latency_s')}s) "
                  f"— looks like the server dropped its backlog mid-stream. "
                  f"Retrying (attempt {attempt}/{args.max_retries})...")
            try:
                raw_text, timing_metrics = asyncio.run(
                    run_one_file(url, args.api_key, audio_path, args.rtf, duration_s)
                )
            except Exception as e:
                if is_channel_exhaustion_error(e):
                    print(f"  ✗ Retry hit a closed connection (1005) — the previous failure "
                          f"almost certainly leaked a channel slot in moshi-server's worker "
                          f"pool (known bug: 'send_loop' dying from the Sending-after-closing "
                          f"race doesn't release its channel). Retrying against an exhausted "
                          f"pool won't help — RESTART moshi-server, then re-run this segment.")
                    raw_text, timing_metrics = "[ERROR]", {}
                    server_needs_restart = True
                    break
                print(f"  Streaming failed on retry: {e}")
                raw_text, timing_metrics = "[ERROR]", {}
            attempt += 1

        if server_needs_restart:
            pass  # already printed a clear message above; don't pile on with the generic one
        elif looks_truncated(timing_metrics, duration_s):
            print(f"  ⚠️  Still truncated after {args.max_retries} retries — "
                  f"keeping the partial result, but this segment needs a manual look "
                  f"(check moshi-server logs for 'Sending after closing is not allowed')")

        hyp_norm = norm.normalize(raw_text)
        result = metrics.compute(
            ref=gt_norm,
            hyp=hyp_norm,
            latency_s=timing_metrics.get("wall_clock_latency_s", -1.0),
            audio_duration_s=duration_s,
            cost_per_minute=0.0,
        )

        print(f"  WER={result.wer:.3f} | CER={result.cer:.3f} | "
              f"first_word={timing_metrics.get('first_word_latency_s')}s | "
              f"mean_word_gap={timing_metrics.get('mean_word_gap_s')}s")

        records.append({
            "model": "kyutai-stt-1b-en_fr-RUST-STREAMING",
            "device": "cuda",
            "serving_mode": f"real-streaming-rtf-{args.rtf}",
            "segment_id": seg_id,
            "audio_file": seg["audio_file"],
            "duration_s": duration_s,
            "hypothesis_raw": raw_text,
            "hypothesis_norm": hyp_norm,
            "reference_norm": gt_norm,
            "dataset_fingerprint": fingerprint,
            **result.to_dict(),
            **timing_metrics,
        })

    df = pd.DataFrame(records)
    df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)

    results_file = Path(args.results_path)
    if args.segment_id and results_file.exists():
        # Partial rerun: merge these rows into the existing results file rather
        # than overwriting the other segments' results.
        existing = pd.read_csv(results_file)
        existing = existing[~existing["segment_id"].isin(df["segment_id"])]
        df = pd.concat([existing, df], ignore_index=True)
        df = df.sort_values("segment_id").reset_index(drop=True)
        print(f"Merged {len(records)} segment(s) into existing {len(existing)}-row results file")

    df.to_csv(args.results_path, index=False, encoding="utf-8")
    print(f"\nResults saved -> {args.results_path}")

    print("\n" + "=" * 60)
    print(f"RESULTS — Kyutai STT 1B (en/fr) / REAL streaming via Rust server (rtf={args.rtf})")
    print("=" * 60)
    print(f"  Segments:              {len(df)}")
    print(f"  Mean WER:              {df['wer'].mean():.3f}")
    print(f"  Mean CER:              {df['cer'].mean():.3f}")
    print(f"  Mean first-word delay: {df['first_word_latency_s'].mean():.3f}s")
    print(f"  Mean per-word gap:     {df['mean_word_gap_s'].mean():.3f}s")
    print(f"  Mean wall-clock time:  {df['wall_clock_latency_s'].mean():.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
