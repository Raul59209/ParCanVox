"""
notebook_voxtral_realtime_4b_trfs.py — Benchmark: Voxtral Mini 4B Realtime via transformers
==============================================================================================
Model:   mistralai/Voxtral-Mini-4B-Realtime-2602
Backend: transformers >= 5.2.0 (VoxtralRealtimeForConditionalGeneration)

IMPORTANT — what this notebook actually does:
  This uses the transformers batch path, NOT the streaming WebSocket API.
  The model is architecturally a streaming model (~480ms per-word delay),
  but transformers runs it one-shot: whole audio in, whole transcript out.
  Latency numbers here are NOT comparable to real streaming use — treat
  WER/CER/medical-entity accuracy as the meaningful metrics, not RTF.

  True streaming would require vLLM's Realtime API (WebSocket), but that
  needs 16GB+ VRAM for serving. This transformers path has no such
  requirement — device_map="auto" will use your GPU where it fits and
  fall back to CPU for the rest.

French WER on FLEURS (from model card):
  - 480ms delay: 6.42%
  - 2400ms delay: 5.23%

Install:
    pip install "transformers>=5.2.0" "mistral-common[audio]" soundfile soxr librosa

Run:
    python notebook_voxtral_realtime_4b_trfs.py
"""

import os
import sys
import json
import time
import logging
import tempfile
import base64
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

sys.path.insert(0, "/app")
from model_loader import load_voxtral_components

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "mistralai/Voxtral-Mini-4B-Realtime-2602"
MODEL_DIR       = os.environ.get("VOXTRAL_MODEL_DIR")
DEFAULT_VLLM_URL = "http://voxtral_vllm:8000/v1"
VLLM_BASE_URL   = os.environ.get("VLLM_BASE_URL", DEFAULT_VLLM_URL).strip()
USE_TRANSFORMERS = os.environ.get("VOXTRAL_USE_TRANSFORMERS", "true").strip().lower() in {"1", "true", "yes", "on"}
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_voxtral_realtime_4b_trfs.csv"
MAX_SEGMENTS    = int(os.environ.get("VOXTRAL_MAX_SEGMENTS", "0").strip() or "0")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

# ── Load model or client ────────────────────────────────────────────────────
processor = None
model = None
client = None

def wait_for_vllm(client, max_retries: int = 12, delay_s: float = 5.0) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            client.models.list()
            log.info("vLLM server is ready")
            return
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"vLLM server at {VLLM_BASE_URL} did not become ready: {exc}") from exc
            log.warning(f"vLLM not ready yet ({attempt}/{max_retries}): {exc}")
            time.sleep(delay_s)


if VLLM_BASE_URL and not USE_TRANSFORMERS:
    from openai import OpenAI
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "dummy"))
    log.info(f"Using vLLM endpoint at {VLLM_BASE_URL}")
    wait_for_vllm(client)
    log.info("VLLM backend selected; sending requests to the remote server")
else:
    log.info(f"Loading {MODEL_ID} via transformers...")
    if MODEL_DIR:
        log.info(f"Using local model directory: {MODEL_DIR}")

    try:
        processor, model = load_voxtral_components(MODEL_ID, model_dir=MODEL_DIR)
    except Exception as exc:
        log.error(f"Failed to load Voxtral model: {exc}")
        raise

    model.eval()

    # When device_map="auto" splits the model across GPU+CPU, model.device
    # raises AttributeError since there's no single device. Use this instead:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Get sampling rate safely — different processor types expose it differently
    try:
        SAMPLING_RATE = processor.feature_extractor.sampling_rate
    except AttributeError:
        try:
            SAMPLING_RATE = processor.sampling_rate
        except AttributeError:
            SAMPLING_RATE = 16000  # Voxtral realtime default per model card
            log.warning(f"Could not read sampling_rate from processor, defaulting to {SAMPLING_RATE}Hz")

    log.info(f"Model loaded ✓ (primary device={DEVICE}, sampling_rate={SAMPLING_RATE}Hz)")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()

if MAX_SEGMENTS > 0:
    log.info(f"Debug mode: processing only the first {MAX_SEGMENTS} segment(s)")
else:
    log.info("Full benchmark mode: processing all segments")


# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
    log.info(f"Starting transcription for {audio_path.name} ({audio_size_mb:.1f} MB)")

    if client is not None:
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
        suffix    = audio_path.suffix.lstrip(".").lower()
        mime      = f"audio/{suffix}" if suffix != "mp3" else "audio/mpeg"

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": suffix,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transcris exactement ce qui est dit dans cet audio médical en français. "
                                "Retourne uniquement la transcription, sans commentaire."
                            ),
                        },
                    ],
                }
            ],
            timeout=600,
        )
        latency = time.perf_counter() - t0
        text = response.choices[0].message.content.strip()
        return text, latency

    """
    Chunked transcription via transformers.
    Uses ffmpeg to convert audio to WAV first, since Audio.from_file()
    in mistral-common can't decode .m4a/AAC directly — same issue we
    hit with Parakeet and torchaudio earlier in this project.

    Audio is split into CHUNK_LEN_S-second pieces before transcription
    rather than processed whole. Two independent reasons:
      1. It bounds max_new_tokens per call to a small, fixed number
         regardless of total file length, instead of one huge generate()
         call needing up to 4096 tokens for a 12-minute file — this matters
         a lot if any part of the model ends up CPU-offloaded (see
         model_loader.py), since that overhead is paid per generated token.
      2. It matches the approach already used for Canary in this project,
         keeping per-model latency numbers roughly comparable.
    """
    import subprocess
    import shutil
    import numpy as np
    import soundfile as sf

    CHUNK_LEN_S = 30
    CHUNK_MAX_NEW_TOKENS = 512  # generous headroom for ~30s of French speech

    # Convert to WAV via ffmpeg at the model's expected sampling rate
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    chunk_dir = None
    try:
        log.info("Converting audio to WAV via ffmpeg...")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", str(SAMPLING_RATE), "-ac", "1", "-f", "wav", tmp_path,
        ], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[-200:]}")
        log.info("ffmpeg conversion complete")

        chunk_dir = tempfile.mkdtemp()
        pattern = str(Path(chunk_dir) / "chunk_%04d.wav")
        result = subprocess.run([
            "ffmpeg", "-y", "-nostdin", "-i", tmp_path,
            "-f", "segment", "-segment_time", str(CHUNK_LEN_S),
            "-c", "copy", pattern,
        ], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg chunking failed: {result.stderr.decode()[-200:]}")
        chunk_paths = sorted(Path(chunk_dir).glob("chunk_*.wav"))
        log.info(f"Split into {len(chunk_paths)} chunk(s) of ≤{CHUNK_LEN_S}s")

        model_dtype = next(model.parameters()).dtype
        texts = []
        for i, chunk_path in enumerate(chunk_paths):
            audio_array, _ = sf.read(str(chunk_path), dtype="float32")

            # CRITICAL: pass audio as explicit keyword
            inputs = processor(audio=audio_array, return_tensors="pt")
            # Cast to model dtype (bfloat16 when loaded with device_map="auto")
            # to avoid "Input type (float) and bias type (c10::BFloat16)" errors
            inputs = {k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
                      for k, v in inputs.items()}

            log.info(f"Generating chunk {i+1}/{len(chunk_paths)}")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=CHUNK_MAX_NEW_TOKENS, do_sample=False)
            chunk_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            texts.append(chunk_text.strip())

        text = " ".join(t for t in texts if t)
        latency = time.perf_counter() - t0
        return text.strip(), latency
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if chunk_dir is not None:
            shutil.rmtree(chunk_dir, ignore_errors=True)


# ── Run benchmark ─────────────────────────────────────────────────────────────
records = []
total   = len(segments)

for idx, seg in enumerate(segments):
    if MAX_SEGMENTS > 0 and idx >= MAX_SEGMENTS:
        log.info(f"Stopping after {MAX_SEGMENTS} segment(s) as requested")
        break

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
        "serving_mode":        "transformers-batch (NOT streaming)",
        "chunking_strategy":   "whole" if client is not None else "chunked_30s",
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
print(f"RESULTS — {MODEL_ID} / transformers batch")
print("  ⚠️  Latency/RTF not meaningful — batch mode, not real streaming")
print("=" * 60)
print(f"  Segments:        {len(df)} ({len(df) - len(valid)} errors)")
if len(valid):
    print(f"  Mean WER:        {valid['wer'].mean():.3f}")
    print(f"  Mean CER:        {valid['cer'].mean():.3f}")
    print(f"  Med entity acc:  {valid['med_entity_acc'].mean():.3f}")
    print(f"  Mean RTF:        {valid['rtf'].mean():.3f}")
    print(f"  Mean latency:    {valid['latency_s'].mean():.1f}s")
    n_crit = valid["med_critical_errors"].apply(
        lambda x: 1 if x and str(x).strip() not in ("", "[]", "nan") else 0
    ).sum()
    print(f"  Critical errors: {n_crit}")
print(f"  Dataset:         {fingerprint}")
print("=" * 60)