# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "python-dotenv",
#     "pandas",
#     "jiwer",
# ]
# ///
"""
score_corrected_kyutai.py — Correct + re-score a Kyutai streaming benchmark run
==================================================================================
Takes an existing results CSV from benchmark_kyutai_streaming.py (which has
hypothesis_raw + reference_norm per segment, already produced from a real
streaming run against your GPU server), runs each hypothesis_raw through
build_error_correction_prompt() (the same LLM correction step from
app_demo.py/prompts.py), re-normalizes and re-scores the CORRECTED text
against the same reference, and saves a new CSV in the same row format as
results_all.csv so it's directly comparable to every other model/run.

This does NOT re-run the streaming server — it works off transcripts you've
already captured, so it's fast and doesn't need moshi-server running.

What you get per segment:
  - wer_before / cer_before   : scores on the raw Kyutai streaming transcript
  - wer_after  / cer_after    : scores on the LLM-corrected transcript
  - n_corrections_applied     : how many edits the LLM made
  - qualite_transcription     : the LLM's own self-assessed quality label
  - llm_correction_cost_s     : wall-clock time the correction call took
                                 (a real cost if you'd wire this into the
                                 live pipeline — correction isn't free)

Usage:
    uv run score_corrected_kyutai.py --input results/results_kyutai_streaming_delay16.csv \
                                      --output results/results_kyutai_corrected.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_error_correction_prompt
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics


def call_llm(prompt: str, max_tokens: int = 6000) -> dict:
    client = OpenAI(base_url="https://api.scaleway.ai/v1", api_key=os.environ["SCW_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON invalide : {e}",
            "raw": raw[:1000],
            "raw_length_chars": len(raw),
            "likely_truncated": len(raw) > 0 and not raw.rstrip().endswith("}"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to an existing Kyutai streaming results CSV (must have hypothesis_raw, reference_norm, audio_duration_s, etc.)")
    parser.add_argument("--output", default="results/results_kyutai_corrected.csv")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    norm = MedicalNormalizer()
    metrics = BenchmarkMetrics()

    records = []
    for idx, row in df_in.iterrows():
        seg_id = row["segment_id"]
        raw_text = row["hypothesis_raw"]
        gt_norm = row["reference_norm"]
        duration_s = row["duration_s"]

        print(f"[{idx+1}/{len(df_in)}] {seg_id}")

        if raw_text == "[ERROR]" or not isinstance(raw_text, str) or not raw_text.strip():
            print("  Skipping — no valid raw transcript to correct")
            continue

        # Score BEFORE correction (should match the original run's numbers,
        # recomputed here so before/after sit side by side using identical code)
        hyp_norm_before = norm.normalize(raw_text)
        result_before = metrics.compute(
            ref=gt_norm, hyp=hyp_norm_before,
            latency_s=row.get("latency_s", row.get("wall_clock_latency_s", -1.0)),
            audio_duration_s=duration_s, cost_per_minute=0.0,
        )

        # Run correction
        t0 = time.perf_counter()
        correction_result = call_llm(build_error_correction_prompt(raw_text))
        correction_time_s = time.perf_counter() - t0

        if "error" in correction_result:
            print(f"  Correction LLM call failed: {correction_result['error']}")
            if correction_result.get("likely_truncated"):
                print(f"  -> Likely TRUNCATED response ({correction_result.get('raw_length_chars')} chars) — "
                      f"raise max_tokens further if this keeps happening on long transcripts")
            corrected_text = raw_text  # fall back to uncorrected rather than dropping the row
            n_corrections = 0
            quality_label = None
        else:
            corrected_text = correction_result.get("transcript_corrigee", raw_text)
            n_corrections = len(correction_result.get("corrections_appliquees", []))
            quality_label = correction_result.get("qualite_transcription")

        # Score AFTER correction
        hyp_norm_after = norm.normalize(corrected_text)
        result_after = metrics.compute(
            ref=gt_norm, hyp=hyp_norm_after,
            latency_s=row.get("latency_s", row.get("wall_clock_latency_s", -1.0)),
            audio_duration_s=duration_s, cost_per_minute=0.0,
        )

        print(f"  WER before={result_before.wer:.3f} -> after={result_after.wer:.3f}  "
              f"({n_corrections} corrections, quality={quality_label})")

        records.append({
            "model": "kyutai-stt-1b-en_fr-RUST-STREAMING+LLM-correction",
            "segment_id": seg_id,
            "audio_file": row.get("audio_file"),
            "duration_s": duration_s,
            "hypothesis_raw_before": raw_text,
            "hypothesis_raw_after": corrected_text,
            "reference_norm": gt_norm,
            "wer_before": round(result_before.wer, 4),
            "cer_before": round(result_before.cer, 4),
            "wer_after": round(result_after.wer, 4),
            "cer_after": round(result_after.cer, 4),
            "wer_improvement": round(result_before.wer - result_after.wer, 4),
            "n_corrections_applied": n_corrections,
            "qualite_transcription": quality_label,
            "llm_correction_cost_s": round(correction_time_s, 2),
            "med_entity_acc_after": result_after.med_entity_acc,
            "med_critical_errors_after": result_after.med_critical_errors,
        })

    df_out = pd.DataFrame(records)
    df_out["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nResults saved -> {args.output}")

    print("\n" + "=" * 70)
    print("BEFORE vs AFTER LLM correction")
    print("=" * 70)
    print(f"  Segments:              {len(df_out)}")
    print(f"  Mean WER before:       {df_out['wer_before'].mean():.4f}")
    print(f"  Mean WER after:        {df_out['wer_after'].mean():.4f}")
    print(f"  Mean improvement:      {df_out['wer_improvement'].mean():.4f}")
    print(f"  Files improved:        {(df_out['wer_improvement'] > 0).sum()} / {len(df_out)}")
    print(f"  Files made worse:      {(df_out['wer_improvement'] < 0).sum()} / {len(df_out)}")
    print(f"  Mean correction cost:  {df_out['llm_correction_cost_s'].mean():.2f}s per file")
    print("=" * 70)


if __name__ == "__main__":
    main()