"""
notebook_whisper_large_v3_scaleway.py — Benchmark: Whisper large-v3 via Scaleway API
======================================================================================
Model:   openai/whisper-large-v3
API:     Scaleway Generative APIs (OpenAI-compatible /v1/audio/transcriptions)

This is the simplest possible path for Whisper large-v3 — no local model,
no VRAM, no Docker complexity. Just an API call, same pattern as Voxtral Small.

Scaleway hosts Whisper large-v3 as a managed endpoint, confirmed in their
model catalog. Uses the standard OpenAI audio transcriptions endpoint format.

Run:
    python notebook_whisper_large_v3_scaleway.py
"""

import sys
import json
import time
import logging
import tempfile
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "whisper-large-v3"
LANGUAGE        = "fr"
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_whisper_large_v3_scaleway.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Client ────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=os.environ["SCW_API_KEY"],
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


# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=MODEL_ID,
            file=f,
            language=LANGUAGE,
            response_format="text",
        )
    latency = time.perf_counter() - t0
    text = response.strip() if isinstance(response, str) else response.text.strip()
    return text, latency


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
        "model":               MODEL_ID,
        "serving":             "scaleway-api",
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
valid = df[df["hypothesis_raw"] != "[ERROR]"]
print("\n" + "=" * 60)
print(f"RESULTS — {MODEL_ID} via Scaleway API")
print("=" * 60)
print(f"  Segments:        {len(df)} ({len(df) - len(valid)} errors)")
print(f"  Mean WER:        {valid['wer'].mean():.3f}")
print(f"  Mean CER:        {valid['cer'].mean():.3f}")
print(f"  Med entity acc:  {valid['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {valid['rtf'].mean():.3f}")
print(f"  Mean latency:    {valid['latency_s'].mean():.1f}s")
print(f"  Dataset:         {fingerprint}")
n_crit = valid["med_critical_errors"].apply(
    lambda x: 1 if x and str(x).strip() not in ("", "[]", "nan") else 0
).sum()
print(f"  Critical errors: {n_crit}")
print("=" * 60)