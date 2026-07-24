"""
notebook_kyutai_stt_1b_trfs.py — Benchmark: Kyutai STT 1B (en/fr), whole audio
=================================================================================
Model:   kyutai/stt-1b-en_fr-trfs
Backend: transformers >= 4.53.0 (native support, NOT the moshi CLI)

FIX (root cause of 100% segment failures, 'NoneType' object has no attribute 'shape'):
  processor(audio_array) was called with NO keyword arguments. Per the official
  HuggingFace model card example, the processor must be called with
  return_tensors="pt" (and padding=True for batches) — without it, the
  processor does not reliably return real tensors, so something downstream
  ends up None and model.generate() crashes trying to read its .shape.
  See: https://huggingface.co/docs/transformers/en/model_doc/kyutai_speech_to_text

Install:
    pip install -U transformers torch torchaudio

Run:
    python notebook_kyutai_stt_1b_trfs.py
"""

import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd
import torch
import torchaudio

# Disable torch.compile/dynamo — Triton's JIT step tries to link against
# libcuda.so at compile time, which isn't available in this container even
# though the GPU itself is visible via the Docker GPU runtime. Forcing
# eager mode avoids the "cannot find -lcuda" linker crash during generate().
torch._dynamo.config.disable = True

from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "kyutai/stt-1b-en_fr-trfs"   # bilingual en/fr — NOT stt-2.6b-en (English only)
LANGUAGE        = "fr"   # informational only — model is bilingual, auto-detects
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_kyutai_stt_1b_trfs.csv"
RESULTS_DIR.mkdir(exist_ok=True)

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR    = 24000  # Kyutai STT expects 24kHz audio (per model card example)

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

# ── Load model ────────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_ID} on {TORCH_DEVICE}...")
processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_ID)
model     = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map=TORCH_DEVICE, torch_dtype="auto"
)
log.info("Model loaded ✓")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()


# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:  # downmix stereo to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    audio_array = waveform.squeeze(0).numpy()

    # FIX: must pass return_tensors="pt" (and padding=True) — without these,
    # the processor does not produce real tensors and model.generate() crashes
    # downstream with 'NoneType' object has no attribute 'shape'.
    # Match the OFFICIAL documented call pattern exactly for single-audio input:
    # processor(audio_array, return_tensors="pt") — audio is positional,
    # padding=True is only used in the multi-file BATCH examples, not here.
    # CRITICAL: audio must be passed as a KEYWORD argument. This processor's
    # __call__ signature is (images=None, text=None, videos=None, audio=None,
    # **kwargs) — audio is the FOURTH parameter, not the first. Passing the
    # array positionally silently binds it to `images` instead, the real
    # `audio` kwarg stays None, and the processor returns an empty
    # BatchFeature with no error — which is exactly what caused every
    # 'NoneType' object has no attribute 'shape' crash so far.
    inputs = processor(audio=audio_array, return_tensors='pt')
    inputs = inputs.to(TORCH_DEVICE)

    with torch.no_grad():
        output_tokens = model.generate(**inputs)

    text = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
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
             f"RTF={result.rtf:.3f} | latency={result.latency_s:.1f}s "
             f"(whole-file one-shot — not real streaming latency)")
    if result.med_critical_errors:
        for err in result.med_critical_errors:
            log.warning(f"  ⚠️  {err}")

    log.info(f"  REF: {gt_norm[:100]}")
    log.info(f"  HYP: {hyp_norm[:100]}")

    records.append({
        "model":               MODEL_ID,
        "device":              TORCH_DEVICE,
        "serving_mode":        "one-shot-whole-file (transformers, NOT moshi CLI)",
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
print(f"RESULTS — Kyutai STT 1B (en/fr) / whole audio, transformers ({TORCH_DEVICE})")
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