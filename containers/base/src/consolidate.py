"""
consolidate.py — Merge All Results Into Final Comparison Table
==============================================================
Run this after all model notebooks have completed.

Reads:
    results/results_faster_whisper.csv
    results/results_whisper_large_v3.csv
    results/results_whisperx.csv
    results/results_nvidia_conformer.csv
    results/results_voxtral.csv          (optional, if you ran it)

Writes:
    results/results_all.csv              — full row-per-segment table
    results/summary_by_model.csv         — one row per model, all metrics
    results/summary_by_model.md          — markdown table for your report

Run:
    python consolidate.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# All possible result files — missing ones are skipped with a warning
RESULT_FILES = {
    "faster-whisper":    RESULTS_DIR / "results_faster_whisper.csv",
    "faster-whisper-chunked": RESULTS_DIR / "results_faster_whisper_chunked.csv",
    "whisper-large-v3":  RESULTS_DIR / "results_whisper_large_v3.csv",
    "whisperx":             RESULTS_DIR / "results_whisperx.csv",
    "whisperx-chunked":     RESULTS_DIR / "results_whisperx_chunked.csv",
    #"whisper_large_v3_scaleway": RESULTS_DIR / "results_whisper_large_v3_scaleway.csv",
    #"whisper_large_v3_scaleway_chunked": RESULTS_DIR / "results_whisper_large_v3_scaleway_chunked.csv",
    # "nvidia-conformer":  RESULTS_DIR / "results_nvidia_conformer.csv",
    # "voxtral":           RESULTS_DIR / "results_voxtral.csv",
    "faster-whisper-int8":  RESULTS_DIR / "results_faster_whisper_int8.csv",
    "faster-whisper-int8-chunked": RESULTS_DIR / "results_faster_whisper_int8_chunked.csv",
    "faster-whisper-turbo": RESULTS_DIR / "results_faster_whisper_turbo_int8.csv",
    "faster-whisper-turbo": RESULTS_DIR / "results_faster_whisper_turbo_int8_chunked.csv",
    # "voxtral-mini": RESULTS_DIR / "results_voxtral_mini.csv",
    "voxtral-small":         RESULTS_DIR / "results_voxtral_small.csv",
    "voxtral-small-chunked": RESULTS_DIR / "results_voxtral_small_chunked.csv",
    "kyutai_stt":            RESULTS_DIR / "results_kyutai_streaming_delay16.csv",
    
}

METRICS_COLS = [
    "wer", "cer", "med_entity_acc", "med_entity_hits",
    "med_entity_total", "latency_s", "rtf", "cost_usd",
    "cost_per_hour_audio",
]

# ── Load all available results ────────────────────────────────────────────────
dfs = []
for model_key, path in RESULT_FILES.items():
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8")
        dfs.append(df)
        log.info(f"  ✓ {path.name} — {len(df)} segments")
    else:
        log.warning(f"  ✗ {path.name} not found — skipping")

if not dfs:
    log.error("No result files found. Run the model notebooks first.")
    sys.exit(1)

# ── Combine ───────────────────────────────────────────────────────────────────
all_results = pd.concat(dfs, ignore_index=True)
all_results.to_csv(RESULTS_DIR / "results_all.csv", index=False, encoding="utf-8")
log.info(f"\nCombined table → results/results_all.csv ({len(all_results)} rows)")

# ── Verify all share the same dataset fingerprint ─────────────────────────────
fingerprints = all_results["dataset_fingerprint"].unique()
if len(fingerprints) > 1:
    log.error(
        f"FINGERPRINT MISMATCH — results were computed against different datasets!\n"
        f"  Found: {fingerprints}\n"
        f"  Re-run all notebooks against the same test_set_frozen.json."
    )
    sys.exit(1)
else:
    log.info(f"Dataset fingerprint consistent across all models ✓ ({fingerprints[0]})")

# ── Summary by model ──────────────────────────────────────────────────────────
available_metrics = [c for c in METRICS_COLS if c in all_results.columns]

summary = (
    all_results
    .groupby("model")[available_metrics]
    .agg(["mean", "median"])
    .round(4)
)

# Flatten column names: (wer, mean) → wer_mean
summary.columns = ["_".join(col) for col in summary.columns]
summary = summary.reset_index()

# Add critical error count per model
crit_counts = (
    all_results
    .assign(has_critical=all_results["med_critical_errors"]
            .apply(lambda x: 1 if pd.notna(x) and x != "" else 0))
    .groupby("model")["has_critical"]
    .sum()
    .reset_index()
    .rename(columns={"has_critical": "n_critical_errors"})
)
summary = summary.merge(crit_counts, on="model", how="left")

# Add segment count
seg_counts = all_results.groupby("model").size().reset_index(name="n_segments")
summary = summary.merge(seg_counts, on="model", how="left")

summary.to_csv(RESULTS_DIR / "summary_by_model.csv", index=False, encoding="utf-8")
log.info(f"Summary table → results/summary_by_model.csv")

# ── Markdown report table ─────────────────────────────────────────────────────
md_cols = {
    "model":               "Model",
    "wer_mean":            "WER ↓",
    "cer_mean":            "CER ↓",
    "med_entity_acc_mean": "Med entity acc ↑",
    "latency_s_mean":      "Latency (s) ↓",
    "rtf_mean":            "RTF ↓",
    "cost_per_hour_audio_mean": "Cost/hr audio ($)",
    "n_critical_errors":   "Critical errors ↓",
    "n_segments":          "Segments",
}

available_md_cols = {k: v for k, v in md_cols.items() if k in summary.columns}
md_df = summary[list(available_md_cols.keys())].rename(columns=available_md_cols)

# Highlight best value per metric column (bold in markdown)
def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows   = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)

md_content = f"""# STT Benchmark Results — French Medical Audio

Dataset fingerprint: `{fingerprints[0]}`

## Summary: model × metric

{md_table(md_df)}

## Notes
- WER and CER: lower is better. Computed on normalized text (same normalizer applied to both reference and hypothesis).
- Med entity acc: proportion of medical entities (drugs, dosages, routes, frequencies) correctly transcribed. Higher is better.
- RTF (Real-Time Factor): latency / audio duration. RTF < 1.0 = faster than real-time.
- Critical errors: dosage mismatches (e.g. 500 mg vs 5000 mg), missing or hallucinated dosages.
- Cost/hr audio: extrapolated API cost per hour of audio. Local models = $0.
"""

md_path = RESULTS_DIR / "summary_by_model.md"
md_path.write_text(md_content, encoding="utf-8")
log.info(f"Markdown report → results/summary_by_model.md")

# ── Print to terminal ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL BENCHMARK SUMMARY")
print("=" * 70)
print(md_df.to_string(index=False))
print("=" * 70)
print(f"\nDataset fingerprint: {fingerprints[0]}")
print(f"Models compared:     {list(all_results['model'].unique())}")
print(f"\nOutput files:")
print(f"  results/results_all.csv        — full per-segment data")
print(f"  results/summary_by_model.csv   — aggregated metrics")
print(f"  results/summary_by_model.md    — ready to paste into your report")