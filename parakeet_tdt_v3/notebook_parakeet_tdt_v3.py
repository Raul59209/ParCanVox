"""
notebook_parakeet_tdt_v3.py — Benchmark: nvidia/parakeet-tdt-0.6b-v3
======================================================================
Model:   nvidia/parakeet-tdt-0.6b-v3
Backend: NeMo 2.4 (FastConformer-TDT decoder)

Why this model:
  - 25 European languages including French (fr WER 5.15% on FLEURS)
  - Automatic language detection — no prompt needed
  - Punctuation and capitalisation preserved in output
  - Up to 24 minutes with full attention (longer with local attention)
  - 0.6B params, runs fine on 8GB VRAM

Audio requirements:
  - 16kHz mono WAV or FLAC
  - Convert from .m4a first:
      ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
  (Note: Kyutai needed 24kHz; Parakeet needs 16kHz — don't reuse
   the same converted files from the Kyutai benchmark run.)

Run:
    python notebook_parakeet_tdt_v3.py
"""

import sys
import json
import time
import logging
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SR       = 16000   # Parakeet expects 16kHz (NOT 24kHz like Kyutai)
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_parakeet_tdt_v3.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

# ── Load model ────────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_ID} via NeMo...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
model.eval()

# Move to GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Switch from full (global, quadratic-memory) attention to local attention
# with a global token. NVIDIA's own spec for this model caps full-attention
# long-form audio at 24 minutes, but only "on A100 80GB" — on smaller GPUs
# the real ceiling is far lower, which matches every segment over ~400-450s
# failing outright in this dataset. Local attention scales close to linearly
# with audio length instead, raising the practical ceiling to ~3 hours on
# the same hardware, and needs no manual audio chunking (unlike Canary,
# whose attention-decoder architecture has no equivalent option).
model.change_attention_model(
    self_attention_model="rel_pos_local_attn",
    att_context_size=[256, 256],
)

# The attention change alone doesn't cover everything: the convolutional
# subsampling module at the very front of the encoder operates directly on
# the raw, un-downsampled audio sequence and can OOM independently of
# whatever attention mechanism is used, on long enough files. This is
# NVIDIA's own documented second half of the long-form-audio fix — without
# it, only the mid-length segments were clearing (self-attention was the
# bottleneck there); the genuinely longest files were still crashing in the
# subsampling step itself. 1 = auto-select chunk size.
model.change_subsampling_conv_chunking_factor(1)

log.info(f"Model loaded ✓ (device={device})")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()


# ── Audio preprocessing ───────────────────────────────────────────────────────
def prepare_wav(audio_path: Path) -> str:
    """
    Use ffmpeg to decode any audio format (m4a, wav, mp3, flac, ogg) to
    a 16kHz mono WAV, then return the temp file path for NeMo to read.

    We use ffmpeg here rather than soundfile because soundfile can't decode
    m4a/AAC (the format our benchmark audio files use) without libsndfile
    being compiled with AAC support, which it typically isn't. ffmpeg has
    no such limitation — it handles AAC natively.
    """
    import subprocess
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-f", "wav",
        tmp.name,
    ], capture_output=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[-300:]}")
    return tmp.name


# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    tmp_path = prepare_wav(audio_path)
    try:
        t0 = time.perf_counter()
        output = model.transcribe([tmp_path])
        latency = time.perf_counter() - t0
        # NeMo returns a list of Hypothesis objects; .text is the transcript
        text = output[0].text if hasattr(output[0], "text") else str(output[0])
        return text.strip(), latency
    finally:
        Path(tmp_path).unlink(missing_ok=True)


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
        "device":              device,
        "compute_type":        "fp32",
        "chunking_strategy":   "whole_local_attn_256",
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
print(f"RESULTS — {MODEL_ID} ({device})")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}  "
      f"{'✓ faster than realtime' if df['rtf'].mean() < 1 else '✗ slower than realtime'}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
print(f"  Dataset:         {fingerprint}")
n_crit = df["med_critical_errors"].apply(
    lambda x: 1 if x and str(x).strip() not in ("", "[]", "nan") else 0
).sum()
print(f"  Critical errors: {n_crit}")
print("=" * 60)