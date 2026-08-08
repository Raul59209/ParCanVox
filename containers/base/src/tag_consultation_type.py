#!/usr/bin/env python3
"""
tag_consultation_type.py
--------------------------
Adds a "consultation_type" field to every segment in test_set_frozen.json:
    "consultation" — live doctor-patient dialogue (both people talking)
    "dictation"     — doctor dictating alone (CRO letters, clinical notes,
                       "chers confrères" style, or first-person chart notes)
    "excluded"      — known reference/audio style mismatch, don't include
                       in either aggregate (currently just seg_0010 — its
                       ground truth was written as a third-person narrative
                       summary while the audio is verbatim dialogue, so WER
                       against it isn't a fair measure of transcription
                       quality; see earlier discussion in this project).

This lets every notebook report consultation-WER and dictation-WER as a
standard split, instead of each analysis re-deriving the classification
by hand from segment content (which drifts and is easy to get wrong or
forget as segments get added).

Usage:
    python3 tag_consultation_type.py dataset/test_set_frozen.json

Writes in place after saving a timestamped .bak copy alongside it.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

CONSULTATION_TYPES = {
    "seg_0001": "consultation",  # nasal polyps follow-up — dialogue
    "seg_0002": "consultation",  # cardio/diabetes follow-up — dialogue
    "seg_0003": "consultation",  # hearing loss — dialogue
    "seg_0004": "consultation",  # vertigo (BPPV) — dialogue
    "seg_0005": "consultation",  # otitis externa — dialogue
    "seg_0006": "dictation",     # ORL operative/consult letter — "chers confrères"
    "seg_0007": "dictation",     # vertigo referral letter — "chers confrères"
    "seg_0008": "dictation",     # oncology (tongue lesion) letter
    "seg_0009": "dictation",     # pediatric ORL letter
    "seg_0010": "excluded",      # reference/audio style mismatch — see docstring
    "seg_0011": "consultation",  # facial palsy — dialogue
    "seg_0012": "dictation",     # pediatric ORL letter
    "seg_0014": "consultation",  # cardio follow-up — dialogue
    "seg_0015": "dictation",     # sinus surgery letter
    "seg_0016": "dictation",     # rheumatology chart note — first-person dictation
    "seg_0017": "consultation",  # migraine/headache — dialogue
    "seg_0018": "dictation",     # oncology (dysphagia) letter
    # seg_0013 intentionally omitted — consistently absent from benchmark
    # runs so far; if/when it reappears, classify it before relying on the
    # aggregate split.
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tag_consultation_type.py /path/to/test_set_frozen.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    dataset = json.loads(path.read_text(encoding="utf-8"))

    tagged, missing = 0, []
    for seg in dataset["segments"]:
        sid = seg["segment_id"]
        if sid in CONSULTATION_TYPES:
            seg["consultation_type"] = CONSULTATION_TYPES[sid]
            tagged += 1
        else:
            seg["consultation_type"] = "unclassified"
            missing.append(sid)

    backup_path = path.with_suffix(
        f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    )
    shutil.copy2(path, backup_path)
    print(f"Backed up original -> {backup_path}")

    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tagged {tagged} segment(s).")
    if missing:
        print(f"UNCLASSIFIED (marked 'unclassified', review and add to "
              f"CONSULTATION_TYPES): {missing}")
    print(f"Updated -> {path}")


if __name__ == "__main__":
    main()
