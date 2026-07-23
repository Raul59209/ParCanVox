"""
prep_ground_truth.py — Bootstrap Ground Truth from Audio
=========================================================
Step 1 in the benchmark pipeline.

What it does:
    1. Scans your audio directory for .wav / .mp3 / .flac files
    2. Runs Whisper large-v3 on each file (GPU accelerated)
    3. Outputs a TSV correction worksheet: you review & fix the transcriptions
    4. Once corrected, run `freeze_dataset.py` to lock the test set

Designed for Scaleway GPU instance (A100 / H100 recommended for large-v3).

Usage:
    pip install openai-whisper num2words
    python prep_ground_truth.py --audio_dir ./audio --output_dir ./dataset

After running:
    → Open dataset/correction_worksheet.tsv in a spreadsheet editor
    → Column A: segment_id (DO NOT EDIT)
    → Column B: audio_file (DO NOT EDIT)
    → Column C: whisper_draft  (Whisper's output — your starting point)
    → Column D: ground_truth   (YOUR CORRECTED TRANSCRIPTION — edit this)
    → Column E: notes          (optional: flag accent, noise, crosstalk, etc.)
    → Save as TSV, then run freeze_dataset.py
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
WHISPER_MODEL = "large-v3"
LANGUAGE = "fr"


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class AudioSegment:
    segment_id: str          # e.g. "seg_001"
    audio_file: str          # relative path from audio_dir
    audio_path: str          # absolute path
    duration_s: float        # audio duration in seconds
    file_hash: str           # SHA256 of audio file (for reproducibility)
    whisper_draft: str       # raw Whisper output
    whisper_normalized: str  # normalized Whisper output (for reference)
    ground_truth: str        # to be filled by human reviewer
    notes: str = ""          # optional reviewer notes
    speaker_id: str = ""     # optional speaker label
    audio_quality: str = ""  # optional: clean / noisy / telephone / etc.


# ---------------------------------------------------------------------------
# AUDIO UTILITIES
# ---------------------------------------------------------------------------

def get_audio_duration(path: str) -> float:
    """Return duration in seconds. Requires ffprobe or torchaudio."""
    try:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return -1.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # first 16 chars is enough for ID purposes


# ---------------------------------------------------------------------------
# WHISPER TRANSCRIPTION
# ---------------------------------------------------------------------------

def load_whisper_model():
    try:
        import whisper
    except ImportError:
        log.error("openai-whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    log.info(f"Loading Whisper {WHISPER_MODEL}...")
    import torch

    # RTX 5060 (Blackwell / sm_120) is not yet supported by any PyTorch build.
    # Force CPU for now — fine for a one-time bootstrap transcription.
    cuda_ok = False
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()  # probe: will throw if no kernel image
            cuda_ok = True
        except Exception as e:
            log.warning(f"CUDA probe failed ({e.__class__.__name__}: {str(e)[:80]})")
            log.warning("Falling back to CPU — your GPU (RTX 5060 Blackwell) is not yet "
                        "supported by the current PyTorch build. CPU is fine for this "
                        "one-time bootstrap step.")

    device = "cuda" if cuda_ok else "cpu"
    log.info(f"Using device: {device}")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    log.info("Whisper model loaded.")
    return model, device


def transcribe_file(model, audio_path: str) -> dict:
    """Returns Whisper result dict with 'text' and 'segments'."""
    result = model.transcribe(
        audio_path,
        language=LANGUAGE,
        task="transcribe",
        verbose=False,
        # Settings tuned for medical French dictation
        temperature=0.0,          # deterministic
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=(
            "Transcription médicale en français. "
            "Termes techniques, médicaments, dosages."
        ),
    )
    return result


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def discover_audio_files(audio_dir: Path) -> list[Path]:
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    files = sorted(files)
    log.info(f"Found {len(files)} audio files in {audio_dir}")
    return files


def run(audio_dir: Path, output_dir: Path, resume: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    worksheet_path = output_dir / "correction_worksheet.tsv"
    metadata_path  = output_dir / "segments_raw.json"
    progress_path  = output_dir / ".progress.json"

    # Load progress (for resume)
    progress = {}
    if resume and progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        log.info(f"Resuming — {len(progress)} segments already transcribed.")

    audio_files = discover_audio_files(audio_dir)
    if not audio_files:
        log.error(f"No audio files found in {audio_dir}. "
                  f"Supported formats: {SUPPORTED_EXTS}")
        sys.exit(1)

    # Load Whisper
    model, device = load_whisper_model()

    # Import normalizer (must be in same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    from normalizer import MedicalNormalizer
    norm = MedicalNormalizer()

    segments = []
    total = len(audio_files)

    for idx, audio_path in enumerate(audio_files):
        seg_id = f"seg_{idx+1:04d}"
        rel_path = str(audio_path.relative_to(audio_dir))

        if seg_id in progress:
            log.info(f"[{idx+1}/{total}] Skipping {seg_id} (already done)")
            segments.append(progress[seg_id])
            continue

        log.info(f"[{idx+1}/{total}] Transcribing {rel_path}...")
        t0 = time.time()

        try:
            result = transcribe_file(model, str(audio_path))
            whisper_text = result["text"].strip()
        except Exception as e:
            log.error(f"Failed to transcribe {audio_path}: {e}")
            whisper_text = "[TRANSCRIPTION ERROR]"

        elapsed = time.time() - t0
        duration = get_audio_duration(str(audio_path))
        rtf = elapsed / duration if duration > 0 else -1

        seg = AudioSegment(
            segment_id=seg_id,
            audio_file=rel_path,
            audio_path=str(audio_path),
            duration_s=round(duration, 2),
            file_hash=sha256_file(str(audio_path)),
            whisper_draft=whisper_text,
            whisper_normalized=norm.normalize(whisper_text),
            ground_truth="",  # to be filled by human
            notes="",
        )

        seg_dict = asdict(seg)
        segments.append(seg_dict)
        progress[seg_id] = seg_dict

        # Save progress after each file
        with open(progress_path, "w") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

        log.info(f"  ✓ {seg_id} | {duration:.1f}s audio | "
                 f"{elapsed:.1f}s transcription | RTF={rtf:.2f}")
        log.info(f"  Draft: {whisper_text[:100]}...")

    # ------------------------------------------------------------------
    # Write correction worksheet (TSV)
    # ------------------------------------------------------------------
    log.info(f"\nWriting correction worksheet → {worksheet_path}")
    with open(worksheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "segment_id",
            "audio_file",
            "duration_s",
            "whisper_draft",
            "ground_truth",          # ← FILL THIS COLUMN
            "whisper_normalized",    # for reference
            "notes",
            "speaker_id",
            "audio_quality",
        ])
        for seg in segments:
            writer.writerow([
                seg["segment_id"],
                seg["audio_file"],
                seg["duration_s"],
                seg["whisper_draft"],
                seg["ground_truth"] or seg["whisper_draft"],  # pre-fill with draft
                seg["whisper_normalized"],
                seg["notes"],
                seg["speaker_id"],
                seg["audio_quality"],
            ])

    # ------------------------------------------------------------------
    # Write raw JSON metadata
    # ------------------------------------------------------------------
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    log.info(f"Raw metadata → {metadata_path}")
    log.info("\n" + "=" * 60)
    log.info("NEXT STEPS:")
    log.info("=" * 60)
    log.info(f"1. Open:  {worksheet_path}")
    log.info("2. Review column D (ground_truth) — correct any Whisper errors")
    log.info("3. Flag noisy/ambiguous segments in column E (notes)")
    log.info("4. Save as TSV (keep the same filename)")
    log.info("5. Run: python freeze_dataset.py --worksheet dataset/correction_worksheet.tsv")
    log.info("=" * 60)
    log.info(f"\nTotal segments: {len(segments)}")
    log.info(f"Total audio:    {sum(s['duration_s'] for s in segments if s['duration_s'] > 0):.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap ground truth transcriptions from audio using Whisper."
    )
    parser.add_argument(
        "--audio_dir", type=Path, required=True,
        help="Directory containing your audio files (searched recursively)."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./dataset"),
        help="Where to write the correction worksheet and metadata."
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Start from scratch (ignore existing progress)."
    )
    args = parser.parse_args()
    run(args.audio_dir, args.output_dir, resume=not args.no_resume)