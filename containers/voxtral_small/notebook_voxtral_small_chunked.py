"""
notebook_voxtral_small_chunked.py — Benchmark: Voxtral Small 24B, silence-chunked
====================================================================================
Model:   mistral/voxtral-small-24b-2507
API:     Scaleway (OpenAI-compatible audio.transcriptions endpoint)

FIX (from previous broken version):
  - Model ID changed from "mistralai/Voxtral-Small-24B-2507" (capitalized,
    HuggingFace-style) to "mistral/voxtral-small-24b-2507" (lowercase,
    Scaleway-style). This is the exact model ID Scaleway's catalog expects —
    the capitalized version returned 100% failures on every segment.
  - API call switched from chat.completions.create() with base64-embedded
    audio to audio.transcriptions.create() with the raw file handle — this
    is the dedicated STT endpoint and matches the whole-audio notebook that
    was already confirmed working.

Chunking strategy (unchanged):
  - pydub detects silence in the audio
  - Chunks are assembled greedily up to CHUNK_MAX_MS (3 minutes)
  - Each chunk is written to a temp WAV and sent to the audio.transcriptions
    endpoint independently
  - Transcripts are joined in order with a single space

Install:
    pip install openai python-dotenv pydub

Run:
    python notebook_voxtral_small_chunked.py
"""

import sys
import json
import time
import logging
import tempfile
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import split_on_silence

load_dotenv()

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "mistral/voxtral-small-24b-2507"   # FIXED: lowercase Scaleway model id
LANGUAGE        = "fr"
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_voxtral_small_chunked.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# Chunking parameters — same as the faster-whisper chunked notebooks
CHUNK_MAX_MS        = 3 * 60 * 1000   # 3 minutes hard ceiling per chunk
SILENCE_THRESH_DB   = -40             # dBFS
SILENCE_MIN_LEN_MS  = 700             # minimum silence duration to split on
SILENCE_KEEP_MS     = 300             # silence padding kept at chunk edges

# ── Client ────────────────────────────────────────────────────────────────────
# timeout=120: default OpenAI client timeout (~10s) is too short for Scaleway's
# audio transcription endpoint on longer chunks. seg_0003/0004/0006 timed out
# under the default — increasing to 120s with max_retries fixes the intermittent
# failures seen on ~3-minute chunks.
client = OpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=os.environ["SCW_API_KEY"],
    timeout=120.0,
    max_retries=3,
)

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()


# ── Chunking ──────────────────────────────────────────────────────────────────
def make_chunks(audio_path: Path) -> list[AudioSegment]:
    audio = AudioSegment.from_file(str(audio_path))

    if len(audio) <= CHUNK_MAX_MS:
        return [audio]

    pieces = split_on_silence(
        audio,
        min_silence_len=SILENCE_MIN_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=SILENCE_KEEP_MS,
    )

    if not pieces:
        log.warning("  No silence detected — falling back to hard 3-min slices")
        return [audio[start : start + CHUNK_MAX_MS]
                for start in range(0, len(audio), CHUNK_MAX_MS)]

    chunks: list[AudioSegment] = []
    current = pieces[0]
    for piece in pieces[1:]:
        if len(current) + len(piece) <= CHUNK_MAX_MS:
            current += piece
        else:
            chunks.append(current)
            current = piece
    chunks.append(current)

    return chunks


# ── Transcribe one chunk via the audio.transcriptions endpoint ───────────────
def transcribe_chunk(chunk: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        chunk.export(tmp_path, format="wav")
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=MODEL_ID,
                file=f,
                language=LANGUAGE,
            )
        return result.text.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Transcribe full file via chunks ───────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float, int]:
    t0     = time.perf_counter()
    chunks = make_chunks(audio_path)
    parts  = []
    for i, chunk in enumerate(chunks):
        log.info(f"    chunk {i+1}/{len(chunks)} — {len(chunk)/1000:.1f}s")
        parts.append(transcribe_chunk(chunk))
    text    = " ".join(p.strip() for p in parts if p.strip())
    latency = time.perf_counter() - t0
    return text.strip(), latency, len(chunks)


# ── Run benchmark ─────────────────────────────────────────────────────────────
records = []
total   = len(segments)

for idx, seg in enumerate(segments):
    seg_id     = seg["segment_id"]
    audio_path = AUDIO_BASE_DIR / seg["audio_file"]
    duration_s = seg["duration_s"]
    gt_norm    = seg["ground_truth_normalized"]

    log.info(f"[{idx+1}/{total}] {seg_id} — {seg['audio_file']}")

    if not audio_path.exists():
        log.warning(f"  Audio not found: {audio_path} — skipping")
        continue

    try:
        raw_text, latency, n_chunks = transcribe(audio_path)
    except Exception as e:
        log.error(f"  Transcription failed: {e}")
        raw_text, latency, n_chunks = "[ERROR]", -1.0, 0

    log.info(f"  chunks={n_chunks} | latency={latency:.1f}s")

    hyp_norm = norm.normalize(raw_text)
    result   = metrics.compute(
        ref=gt_norm,
        hyp=hyp_norm,
        latency_s=latency,
        audio_duration_s=duration_s,
        cost_per_minute=0.0,
    )

    log.info(f"  WER={result.wer:.3f} | CER={result.cer:.3f} | "
             f"RTF={result.rtf:.3f} | latency={result.latency_s:.1f}s")
    if result.med_critical_errors:
        for err in result.med_critical_errors:
            log.warning(f"  ⚠️  {err}")

    log.info(f"  REF: {gt_norm[:100]}")
    log.info(f"  HYP: {hyp_norm[:100]}")

    records.append({
        "model":               MODEL_ID,
        "chunking_strategy":   "silence_3min",
        "n_chunks":            n_chunks,
        "chunk_max_ms":        CHUNK_MAX_MS,
        "silence_thresh_db":   SILENCE_THRESH_DB,
        "silence_min_len_ms":  SILENCE_MIN_LEN_MS,
        "segment_id":          seg_id,
        "audio_file":          seg["audio_file"],
        "duration_s":          duration_s,
        "hypothesis_raw":      raw_text,
        "hypothesis_norm":     hyp_norm,
        "reference_norm":      gt_norm,
        "dataset_fingerprint": fingerprint,
        **result.to_dict(),
    })

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
log.info(f"\nResults saved → {RESULTS_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS — Voxtral Small 24B / silence-chunked")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}  {'✓ faster than realtime' if df['rtf'].mean() < 1 else '✗ slower than realtime'}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
print(f"  Mean chunks:     {df['n_chunks'].mean():.1f} per file")
print(f"  Dataset:         {fingerprint}")
n_crit = df['med_critical_errors'].apply(lambda x: 1 if x and x != '' else 0).sum()
print(f"  Critical errors: {n_crit}")
print("=" * 60)