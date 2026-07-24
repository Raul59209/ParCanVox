# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "python-dotenv",
# ]
# ///
"""
correct_kyutai_transcript.py — Run a Kyutai streaming transcript through
the existing build_error_correction_prompt() LLM call from Napoleon's
prompts.py, and show before/after so you can judge whether the prompt
(tuned originally with Kyutai in mind) actually helps on real streaming
output.

This does NOT change anything in app_demo.py or the pipeline — it's a
standalone check so you can decide whether to wire this into the real
pipeline, and whether the prompt needs tuning first.

Usage:
    uv run correct_kyutai_transcript.py "Bonjour, comment allez-vous..."

    # or read from a file:
    uv run correct_kyutai_transcript.py --file transcript.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Adjust this import path to wherever your real prompts.py lives
sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_error_correction_prompt


def call_llm(prompt: str, max_tokens: int = 2000) -> dict:
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
        return {"error": f"JSON invalide : {e}", "raw": raw[:1000]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", nargs="?", help="Transcript text directly on the command line")
    parser.add_argument("--file", help="Path to a text file containing the transcript instead")
    args = parser.parse_args()

    if args.file:
        transcript = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.transcript:
        transcript = args.transcript
    else:
        print("Provide a transcript as an argument or via --file")
        return

    print("=" * 70)
    print("BEFORE (raw Kyutai streaming transcript)")
    print("=" * 70)
    print(transcript)
    print()

    prompt = build_error_correction_prompt(transcript)
    result = call_llm(prompt)

    if "error" in result:
        print("LLM call failed or returned invalid JSON:")
        print(result)
        return

    print("=" * 70)
    print(f"QUALITÉ ÉVALUÉE : {result.get('qualite_transcription', '?')}")
    print("=" * 70)

    corrections = result.get("corrections_appliquees", [])
    print(f"\n{len(corrections)} correction(s) appliquée(s):\n")
    for c in corrections:
        print(f"  [{c.get('type')}] \"{c.get('original')}\" -> \"{c.get('corrige')}\"  "
              f"(confiance: {c.get('confiance')})")
        print(f"      raison: {c.get('raison')}")
        print()

    print("=" * 70)
    print("AFTER (corrected transcript)")
    print("=" * 70)
    print(result.get("transcript_corrigee", "<missing>"))


if __name__ == "__main__":
    main()