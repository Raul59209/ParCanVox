"""
notebook_whisper_large_v3.py — Benchmark: openai/whisper large-v3
==================================================================
Install:
    pip install openai-whisper

Run:
    python notebook_whisper_large_v3.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd

# Force CPU — RTX 5060 (sm_120) not yet supported by PyTorch CUDA kernels
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer 
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "large-v3"
LANGUAGE        = "fr"
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_whisper_large_v3.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

# ── Load model ────────────────────────────────────────────────────────────────
log.info(f"Loading openai-whisper {MODEL_NAME} (CPU)...")
import whisper
import torch
device = "cpu"
model  = whisper.load_model(MODEL_NAME, device=device)
log.info("Model loaded ✓")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()

# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=(
            "Transcription médicale en français. "
            "Termes: mg, ml, narine, polypes, cortisone, Nasonex."
        ),
    )
    latency = time.perf_counter() - t0
    return result["text"].strip(), latency

# ── Run benchmark ─────────────────────────────────────────────────────────────
records = []
for idx, seg in enumerate(segments):
    seg_id     = seg["segment_id"]
    audio_path = AUDIO_BASE_DIR / seg["audio_file"]
    duration_s = seg["duration_s"]
    gt_norm    = seg["ground_truth_normalized"]

    log.info(f"[{idx+1}/{len(segments)}] {seg_id}")

    if not audio_path.exists():
        log.warning(f"  Audio not found: {audio_path} — skipping")
        continue

    try:
        raw_text, latency = transcribe(audio_path)
    except Exception as e:
        log.error(f"  Failed: {e}")
        raw_text, latency = "[ERROR]", -1.0

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

    records.append({
        "model":               f"whisper-{MODEL_NAME}",
        "device":              device,
        "compute_type":        "fp32",
        "segment_id":          seg_id,
        "audio_file":          seg["audio_file"],
        "duration_s":          duration_s,
        "hypothesis_raw":      raw_text,
        "hypothesis_norm":     hyp_norm,
        "reference_norm":      gt_norm,
        "dataset_fingerprint": fingerprint,
        **result.to_dict(),
    })

# ── Save & summarise ──────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
log.info(f"Results saved → {RESULTS_PATH}")

print("\n" + "=" * 60)
print(f"RESULTS — whisper {MODEL_NAME} ({device})")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
n_crit = df['med_critical_errors'].apply(lambda x: 1 if x and x != '' else 0).sum()
print(f"  Critical errors: {n_crit}")
print(f"  Dataset:         {fingerprint}")
print("=" * 60)