"""
notebook_whisperx.py — Benchmark: WhisperX 3.8.4
"""

import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd
import whisperx

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
RESULTS_PATH   = RESULTS_DIR / "results_whisperx.csv"
RESULTS_DIR.mkdir(exist_ok=True)

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

log.info(f"Loading WhisperX {MODEL_SIZE}...")
try:
    model = whisperx.load_model(MODEL_SIZE, device="cuda", compute_type="float16", language=LANGUAGE)
    # Reduced from 8 to 4: seg_0002 (Recording_1001.m4a) consistently OOMs at
    # batch_size=8 even immediately after a full cache clear + synchronize,
    # meaning peak VRAM usage at batch_size=8 is genuinely too close to the
    # ceiling for at least one file in this dataset. Lowering the batch size
    # for the whole run reduces peak usage on every segment, not just the
    # one that happened to fail first.
    DEVICE, COMPUTE_TYPE, BATCH_SIZE = "cuda", "float16", 4
    log.info("Loaded on cuda/float16 ✓")
except Exception as e:
    log.warning(f"GPU failed: {e} — falling back to CPU")
    model = whisperx.load_model(MODEL_SIZE, device="cpu", compute_type="int8", language=LANGUAGE)
    DEVICE, COMPUTE_TYPE, BATCH_SIZE = "cpu", "int8", 1
    log.info("Loaded on cpu/int8 ✓")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()

import torch

def transcribe(audio_path: Path) -> tuple[str, float]:
    # Clear cache BEFORE starting too, not just after — residual fragmentation
    # left over from the previous segment's allocations can eat into the
    # margin available for this segment even if that previous segment
    # succeeded cleanly.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    t0    = time.perf_counter()
    audio = whisperx.load_audio(str(audio_path))
    try:
        result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.warning(f"  OOM — synchronizing, clearing cache, and retrying with batch_size=1")
            # empty_cache() alone only frees cached allocations — it does not
            # resolve a CUDA context left in a bad state mid-kernel-launch.
            # synchronize() forces all pending GPU work to actually finish
            # (or fail explicitly) before we touch the device again, which
            # is what was missing last time (retry crashed with
            # "invalid device ordinal" instead of succeeding).
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                result = model.transcribe(audio, batch_size=1, language=LANGUAGE)
            except RuntimeError as e2:
                # Context is unrecoverable in-process — fail this segment
                # cleanly instead of letting it poison the next one.
                log.error(f"  Retry also failed ({e2}) — skipping segment")
                raise
        else:
            raise
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    text  = " ".join(s["text"].strip() for s in result.get("segments", []))
    return text.strip(), time.perf_counter() - t0

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
    log.info(f"  REF: {gt_norm[:100]}")
    log.info(f"  HYP: {hyp_norm[:100]}")

    records.append({
        "model":               f"whisperx-{MODEL_SIZE}",
        "device":              DEVICE,
        "compute_type":        COMPUTE_TYPE,
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
print(f"RESULTS — whisperx-{MODEL_SIZE} ({DEVICE}/{COMPUTE_TYPE})")
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