"""
freeze_dataset.py — Freeze the Benchmark Test Set
===================================================
Run this AFTER you've corrected the correction_worksheet.tsv from prep_ground_truth.py.

What it does:
    1. Reads your corrected TSV
    2. Applies the normalizer to ground_truth column
    3. Computes a dataset fingerprint (SHA256 of all audio hashes + normalized refs)
    4. Writes the frozen dataset: dataset/test_set_frozen.json
    5. Prints a summary with dataset stats

The frozen JSON is the single source of truth for all benchmark notebooks.
Every model notebook loads ONLY this file — never the raw TSV.

Usage:
    python freeze_dataset.py --worksheet dataset/correction_worksheet.tsv
"""

import sys
import csv
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Import normalizer from same directory
sys.path.insert(0, str(Path(__file__).parent))
from normalizer import MedicalNormalizer


def load_worksheet(tsv_path: Path) -> list[dict]:
    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def validate_rows(rows: list[dict]) -> list[str]:
    """Return list of error messages (empty = all good)."""
    errors = []
    ids_seen = set()
    for i, row in enumerate(rows, 1):
        seg_id = row.get("segment_id", "").strip()
        gt = row.get("ground_truth", "").strip()
        audio = row.get("audio_file", "").strip()

        if not seg_id:
            errors.append(f"Row {i}: missing segment_id")
        if seg_id in ids_seen:
            errors.append(f"Row {i}: duplicate segment_id '{seg_id}'")
        ids_seen.add(seg_id)
        if not gt:
            errors.append(f"Row {i} ({seg_id}): ground_truth is empty — "
                          "did you forget to fill column D?")
        if gt == "[TRANSCRIPTION ERROR]":
            errors.append(f"Row {i} ({seg_id}): ground_truth still has placeholder — "
                          "correct or remove this segment.")
        if not audio:
            errors.append(f"Row {i} ({seg_id}): missing audio_file")
    return errors


def compute_fingerprint(records: list[dict]) -> str:
    """Deterministic fingerprint of the frozen dataset."""
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r["segment_id"]):
        h.update(rec["segment_id"].encode())
        h.update(rec["ground_truth_normalized"].encode())
        h.update(rec.get("file_hash", "").encode())
    return h.hexdigest()[:20]


def run(worksheet_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "test_set_frozen.json"

    log.info(f"Loading worksheet: {worksheet_path}")
    rows = load_worksheet(worksheet_path)
    log.info(f"Loaded {len(rows)} rows.")

    # Validate
    errors = validate_rows(rows)
    if errors:
        log.error("Worksheet validation FAILED:")
        for e in errors:
            log.error(f"  ✗ {e}")
        log.error("Fix the worksheet and re-run.")
        sys.exit(1)
    log.info("Worksheet validation passed ✓")

    # Apply normalizer
    norm = MedicalNormalizer()
    records = []
    for row in rows:
        gt_raw = row["ground_truth"].strip()
        gt_norm = norm.normalize(gt_raw)
        records.append({
            "segment_id":             row["segment_id"].strip(),
            "audio_file":             row["audio_file"].strip(),
            "duration_s":             float(row.get("duration_s") or -1),
            "file_hash":              row.get("file_hash", "").strip(),
            "ground_truth_raw":       gt_raw,
            "ground_truth_normalized": gt_norm,
            "speaker_id":             row.get("speaker_id", "").strip(),
            "audio_quality":          (row.get("audio_quality") or "").strip(),
            "notes":                  row.get("notes", "").strip(),
        })

    # Compute dataset fingerprint
    fingerprint = compute_fingerprint(records)

    # Build frozen dataset
    frozen = {
        "dataset_version":    "1.0",
        "frozen_at":          datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "normalizer_version": "medical_fr_v1",
        "language":           "fr",
        "domain":             "medical",
        "n_segments":         len(records),
        "total_duration_s":   sum(r["duration_s"] for r in records if r["duration_s"] > 0),
        "segments":           records,
    }

    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)

    # Summary stats
    durations = [r["duration_s"] for r in records if r["duration_s"] > 0]
    total_s = sum(durations)
    speakers = {r["speaker_id"] for r in records if r["speaker_id"]}
    qualities = {}
    for r in records:
        q = r["audio_quality"] or "unspecified"
        qualities[q] = qualities.get(q, 0) + 1

    log.info("\n" + "=" * 60)
    log.info("TEST SET FROZEN ✓")
    log.info("=" * 60)
    log.info(f"  Output:         {frozen_path}")
    log.info(f"  Fingerprint:    {fingerprint}")
    log.info(f"  Segments:       {len(records)}")
    log.info(f"  Total audio:    {total_s:.1f}s ({total_s/60:.1f} min)")
    if durations:
        log.info(f"  Avg duration:   {total_s/len(durations):.1f}s per segment")
        log.info(f"  Min/Max:        {min(durations):.1f}s / {max(durations):.1f}s")
    if speakers:
        log.info(f"  Speakers:       {len(speakers)} ({', '.join(sorted(speakers))})")
    log.info(f"  Audio quality breakdown: {qualities}")
    log.info("\n  Log this fingerprint in every model notebook to ensure")
    log.info("  you always evaluate against the same test set.")
    log.info("=" * 60)
    log.info("\nNEXT STEP: run a benchmark notebook for each model.")
    log.info("  E.g.: jupyter nbconvert --to notebook --execute notebook_whisper_large_v3.ipynb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Freeze the benchmark test set after human correction."
    )
    parser.add_argument(
        "--worksheet", type=Path, required=True,
        help="Path to the corrected correction_worksheet.tsv"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./dataset"),
        help="Where to write test_set_frozen.json (default: ./dataset)"
    )
    args = parser.parse_args()
    run(args.worksheet, args.output_dir)