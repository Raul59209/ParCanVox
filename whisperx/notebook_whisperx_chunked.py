"""
notebook_whisperx_chunked.py — Benchmark: WhisperX 3.8.4, silence-chunked
============================================================================
Model:   WhisperX large-v3
Backend: whisperx (faster-whisper + forced alignment under the hood)

FIX (from un-chunked notebook's confirmed OOM bug):
  seg_0002 crashed with "CUDA failed with error out of memory", and the
  VRAM fragmentation it left behind immediately broke seg_0003 with
  "cudaErrorInvalidDevice: invalid device ordinal" — a corrupted CUDA
  context cascading from the first failure. Fix: clear torch's CUDA cache
  after every chunk (success or failure), and retry once at batch_size=1
  if a chunk OOMs instead of giving up immediately.

Run:
    python notebook_whisperx_chunked.py
"""

import sys
import json
import time
import logging
import tempfile
from pathlib import Path

import pandas as pd
import torch
import whisperx
from pydub import AudioSegment
from pydub.silence import split_on_silence

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_SIZE     = "large-v3"
LANGUAGE       = "fr"
DATASET_PATH   = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR = Path("audio")
RESULTS_DIR    = Path("results")
RESULTS_PATH   = RESULTS_DIR / "results_whisperx_chunked.csv"
RESULTS_DIR.mkdir(exist_ok=True)

CHUNK_MAX_MS        = 3 * 60 * 1000
SILENCE_THRESH_DB   = -40
SILENCE_MIN_LEN_MS  = 700
SILENCE_KEEP_MS     = 300

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

log.info(f"Loading WhisperX {MODEL_SIZE}...")
try:
    model = whisperx.load_model(MODEL_SIZE, device="cuda", compute_type="float16", language=LANGUAGE)
    DEVICE, COMPUTE_TYPE, BATCH_SIZE = "cuda", "float16", 8
    log.info("Loaded on cuda/float16 ✓")
except Exception as e:
    log.warning(f"GPU failed: {e} — falling back to CPU")
    model = whisperx.load_model(MODEL_SIZE, device="cpu", compute_type="int8", language=LANGUAGE)
    DEVICE, COMPUTE_TYPE, BATCH_SIZE = "cpu", "int8", 1
    log.info("Loaded on cpu/int8 ✓")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()


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


def transcribe_chunk(chunk: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        chunk.export(tmp_path, format="wav")
        audio = whisperx.load_audio(tmp_path)
        try:
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("    OOM on chunk — clearing cache and retrying with batch_size=1")
                torch.cuda.empty_cache()
                result = model.transcribe(audio, batch_size=1, language=LANGUAGE)
            else:
                raise
        return " ".join(s["text"].strip() for s in result.get("segments", []))
    finally:
        # Clear cache after every chunk (success or failure) to stop VRAM
        # fragmentation from one chunk's OOM cascading into the next chunk's
        # "invalid device ordinal" crash.
        torch.cuda.empty_cache()
        Path(tmp_path).unlink(missing_ok=True)


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


records = []
total   = len(segments)

for idx, seg in enumerate(segments):
    seg_id     = seg["segment_id"]
    audio_path = AUDIO_BASE_DIR / seg["audio_file"]
    duration_s = seg["duration_s"]
    gt_norm    = seg["ground_truth_normalized"]

    log.info(f"[{idx+1}/{total}] {seg_id}")

    if not audio_path.exists():
        log.warning(f"  Audio not found: {audio_path} — skipping")
        continue

    try:
        raw_text, latency, n_chunks = transcribe(audio_path)
    except Exception as e:
        log.error(f"  Failed: {e}")
        raw_text, latency, n_chunks = "[ERROR]", -1.0, 0

    log.info(f"  chunks={n_chunks} | latency={latency:.1f}s")

    hyp_norm = norm.normalize(raw_text)
    result   = metrics.compute(
        ref=gt_norm, hyp=hyp_norm,
        latency_s=latency, audio_duration_s=duration_s,
        cost_per_minute=0.0,
    )

    log.info(f"  WER={result.wer:.3f} | CER={result.cer:.3f} | RTF={result.rtf:.3f}")
    if result.med_critical_errors:
        for err in result.med_critical_errors:
            log.warning(f"  ⚠️  {err}")
    log.info(f"  REF: {gt_norm[:100]}")
    log.info(f"  HYP: {hyp_norm[:100]}")

    records.append({
        "model":               f"whisperx-{MODEL_SIZE}-chunked",
        "device":              DEVICE,
        "compute_type":        COMPUTE_TYPE,
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

df = pd.DataFrame(records)
df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
log.info(f"Results saved → {RESULTS_PATH}")

print("\n" + "=" * 60)
print(f"RESULTS — whisperx-{MODEL_SIZE} / silence-chunked ({DEVICE}/{COMPUTE_TYPE})")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
print(f"  Mean chunks:     {df['n_chunks'].mean():.1f} per file")
n_crit = df['med_critical_errors'].apply(lambda x: 1 if x and x != '' else 0).sum()
print(f"  Critical errors: {n_crit}")
print(f"  Dataset:         {fingerprint}")
print("=" * 60)