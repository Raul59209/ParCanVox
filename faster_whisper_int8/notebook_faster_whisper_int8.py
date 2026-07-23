"""
notebook_faster_whisper_int8.py — Benchmark: faster-whisper large-v3 int8
==========================================================================
Model:   faster-whisper large-v3, forced int8 quantization
Backend: CTranslate2

Difference from notebook_faster_whisper.py:
  - COMPUTE_TYPE is hardcoded to "int8" (no float16 fallback)
  - Results saved to results_faster_whisper_int8.csv
  - Everything else identical — whole audio file passed at once

Run:
    python notebook_faster_whisper_int8.py
"""

import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
LANGUAGE        = "fr"
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_faster_whisper_int8.csv"
RESULTS_DIR.mkdir(exist_ok=True)

COMPUTE_TYPE    = "int8"    # forced — this is what we're benchmarking
DEVICE          = "cuda"    # int8 runs on GPU too; falls back to cpu below

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

# ── Load model ────────────────────────────────────────────────────────────────
log.info(f"Loading faster-whisper {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model loaded ✓")
except Exception as e:
    log.warning(f"Failed with {DEVICE}/{COMPUTE_TYPE}: {e}")
    log.info("Retrying with cpu/int8...")
    DEVICE = "cpu"
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model loaded on CPU ✓")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()

# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        initial_prompt=(
            "Transcription médicale en français. "
            "Termes: mg, ml, narine, polypes, cortisone, Nasonex."
        ),
    )
    text = " ".join(seg.text.strip() for seg in segments_gen)
    latency = time.perf_counter() - t0
    return text.strip(), latency

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
        raw_text, latency = transcribe(audio_path)
    except Exception as e:
        log.error(f"  Transcription failed: {e}")
        raw_text, latency = "[ERROR]", -1.0

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
        "model":               f"faster-whisper-{MODEL_SIZE}-int8-whole",
        "device":              DEVICE,
        "compute_type":        COMPUTE_TYPE,
        "chunking_strategy":   "whole",
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
print(f"RESULTS — faster-whisper {MODEL_SIZE} int8 / whole audio ({DEVICE})")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}  {'✓ faster than realtime' if df['rtf'].mean() < 1 else '✗ slower than realtime'}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
print(f"  Dataset:         {fingerprint}")
n_crit = df['med_critical_errors'].apply(lambda x: 1 if x and x != '' else 0).sum()
print(f"  Critical errors: {n_crit}")
print("=" * 60)