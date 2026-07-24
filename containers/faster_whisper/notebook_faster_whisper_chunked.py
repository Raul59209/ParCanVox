"""
notebook_faster_whisper_chunked.py — Benchmark: faster-whisper, silence-chunked
==================================================================================
Model:   faster-whisper large-v3 (CTranslate2 backend)
Repo:    https://github.com/SYSTRAN/faster-whisper

Same model/device logic as notebook_faster_whisper.py (float16 on GPU,
falls back to cpu/int8 if that fails) — the only difference is this notebook
pre-splits each file into <=3min silence-based chunks before transcribing,
instead of handing the whole file to faster-whisper in one call.

Chunking strategy (same params as your other chunked notebooks):
  - pydub detects silence in the audio
  - Chunks are assembled greedily up to CHUNK_MAX_MS (3 minutes)
  - Each chunk is written to a temp WAV file and transcribed independently
  - Transcripts are joined in order with a single space

Install:
    pip install faster-whisper pydub

Run:
    python notebook_faster_whisper_chunked.py
"""

import sys
import json
import time
import logging
import tempfile
import argparse
from pathlib import Path

import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--segment-id", default=None,
                     help="Comma-separated segment_id(s) to (re)run, e.g. "
                          "seg_0016,seg_0017. Omit to run the full dataset.")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SIZE      = "large-v3"
LANGUAGE        = "fr"
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_faster_whisper_chunked.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# faster-whisper compute type — try "float16" first (GPU), fall back to "int8" (CPU)
COMPUTE_TYPE    = "float16"   # change to "int8" if GPU fails
DEVICE          = "cuda"      # change to "cpu" if needed

# Chunking parameters — same as your other chunked notebooks
CHUNK_MAX_MS        = 3 * 60 * 1000   # 3 minutes hard ceiling per chunk
SILENCE_THRESH_DB   = -40             # dBFS
SILENCE_MIN_LEN_MS  = 700             # minimum silence duration to split on
SILENCE_KEEP_MS     = 300             # silence padding kept at chunk edges

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]

if args.segment_id:
    wanted = {s.strip() for s in args.segment_id.split(",") if s.strip()}
    segments = [s for s in segments if s["segment_id"] in wanted]
    found = {s["segment_id"] for s in segments}
    missing = wanted - found
    if missing:
        log.warning(f"No segment found for segment_id(s): {sorted(missing)}")
    if not segments:
        log.error(f"None of the requested segments were found in the dataset — exiting")
        sys.exit(1)
    log.info(f"Filtering to {len(segments)} segment(s): {sorted(found)}")

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
    DEVICE, COMPUTE_TYPE = "cpu", "int8"
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model loaded on CPU ✓")

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

    # Safety net: split_on_silence can return a single piece spanning almost
    # the whole file when the audio has no long-enough silence gap (e.g.
    # continuous dictation). The merge loop above only caps length when
    # COMBINING pieces — it never checks whether an individual piece is
    # already oversized on its own. Without this pass, such a piece slips
    # through untouched and gets fed to Whisper as one multi-minute chunk,
    # which is exactly what triggers the long-form repetition hallucination
    # this chunking strategy exists to avoid.
    final_chunks: list[AudioSegment] = []
    for c in chunks:
        if len(c) <= CHUNK_MAX_MS:
            final_chunks.append(c)
        else:
            log.warning(f"  Chunk of {len(c)/1000:.1f}s still exceeds "
                        f"{CHUNK_MAX_MS/1000:.0f}s cap after silence-split — "
                        f"hard-slicing it further")
            for start in range(0, len(c), CHUNK_MAX_MS):
                final_chunks.append(c[start:start + CHUNK_MAX_MS])

    return final_chunks


# ── Transcribe one chunk (temp WAV file) ─────────────────────────────────────
def transcribe_chunk(chunk: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        chunk.export(tmp_path, format="wav")
        segments_gen, info = model.transcribe(
            tmp_path,
            language=LANGUAGE,
            beam_size=5,
            # A tuple (not a single float) enables faster-whisper's built-in
            # hallucination recovery: if compression_ratio_threshold or
            # logprob_threshold flags a segment as degenerate (a repetition
            # loop compresses extremely well, i.e. a high compression ratio),
            # it automatically retries that segment at the next, higher
            # temperature instead of keeping the garbage output.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            # Prevents one hallucinated segment from poisoning Whisper's own
            # internal context for every segment that follows it within this
            # chunk — each segment is decoded independently of prior ones.
            condition_on_previous_text=False,
            vad_filter=True,
            initial_prompt=(
                "Transcription médicale en français. "
                "Termes: mg, ml, narine, polypes, cortisone, Nasonex."
            ),
        )
        return " ".join(s.text.strip() for s in segments_gen)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Transcribe full file via chunks ───────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float, int]:
    """Returns (full_text, total_latency_s, n_chunks)."""
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
        "model":               f"faster-whisper-{MODEL_SIZE}-chunked",
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

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()

if args.segment_id and RESULTS_PATH.exists():
    # Partial rerun: merge these rows into the existing results file rather
    # than overwriting the other segments' results.
    existing = pd.read_csv(RESULTS_PATH)
    existing = existing[~existing["segment_id"].isin(df["segment_id"])]
    df = pd.concat([existing, df], ignore_index=True)
    df = df.sort_values("segment_id").reset_index(drop=True)
    log.info(f"Merged {len(records)} segment(s) into existing {len(existing)}-row results file")

df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
log.info(f"\nResults saved → {RESULTS_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS — faster-whisper {MODEL_SIZE} / silence-chunked ({DEVICE}/{COMPUTE_TYPE})")
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