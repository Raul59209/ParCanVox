"""
correct_transcriptions.py — LLM correction of benchmark CSVs
=============================================================
Reads any benchmark CSV, runs LLM correction on each hypothesis_raw
using the Scaleway API + drug lookup (CIS_bdpm.txt + lexique phonétique),
recomputes WER/CER/entity accuracy, and saves corrected CSVs.

Usage:
    python correct_transcriptions.py results/results_whisperx.csv
    python correct_transcriptions.py results/*.csv
    python correct_transcriptions.py results/results_whisperx.csv results/results_fw_turbo.csv

Output:
    results/results_whisperx_llm_corrected.csv
    (original filename with _llm_corrected suffix)

Requirements:
    pip install openai python-dotenv jiwer pandas
    CIS_bdpm.txt and Lexique_phonétique_médicaments.json in same folder as this script
    SCW_API_KEY in .env or environment
"""

import sys
import os
import json
import time
import logging
import re
import unicodedata
from pathlib import Path
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Scaleway client ───────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=os.environ["SCW_API_KEY"],
    timeout=120.0,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CIS_PATH      = BASE_DIR / "CIS_bdpm.txt"
PHONETIC_PATH = BASE_DIR / "Lexique_phonétique_médicaments.json"


# ══════════════════════════════════════════════════════════════════════════════
# DRUG LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


@lru_cache(maxsize=1)
def _load_cis() -> list[str]:
    if not CIS_PATH.exists():
        log.warning(f"CIS_bdpm.txt not found at {CIS_PATH}")
        return []
    names = set()
    with open(CIS_PATH, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            raw  = parts[1].strip()
            name = re.split(r"\s+\d|\s*,", raw)[0].strip().upper()
            if len(name) < 4 or len(name) > 40:
                continue
            if any(s in name for s in ("BOIRON", "LEHNING", "WELEDA")):
                continue
            names.add(name)
    log.info(f"Loaded {len(names)} drug names from CIS_bdpm.txt")
    return sorted(names)


@lru_cache(maxsize=1)
def _load_phonetic() -> dict[str, str]:
    if not PHONETIC_PATH.exists():
        log.warning(f"Lexique phonétique not found at {PHONETIC_PATH}")
        return {}
    with open(PHONETIC_PATH, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    entries = data.get("entries", data) if isinstance(data, dict) else data
    if isinstance(entries, list):
        for entry in entries:
            name = entry.get("text") or entry.get("nom") or ""
            ipa  = entry.get("ipa") or entry.get("phoneme") or ""
            if isinstance(ipa, list):
                ipa = " ".join(ipa)
            if name and ipa:
                result[_normalize(ipa)] = name.upper()
    log.info(f"Loaded {len(result)} phonetic entries")
    return result


def get_candidates(transcript: str, max_results: int = 60) -> list[str]:
    all_names = _load_cis()
    phonetic  = _load_phonetic()
    transcript_norm = _normalize(transcript)
    words = set(re.findall(r"[a-z]{4,}", transcript_norm))

    candidates = set()

    for name in all_names:
        name_norm = _normalize(name)
        if name_norm in transcript_norm:
            candidates.add(name)

    for name in all_names:
        name_norm = _normalize(name)
        for word in words:
            if len(word) >= 5 and (
                name_norm.startswith(word[:5]) or
                word.startswith(name_norm[:5])
            ):
                candidates.add(name)

    for word in words:
        if word in phonetic:
            candidates.add(phonetic[word])

    result = sorted(candidates)[:max_results]
    if not result:
        short = [n for n in all_names if len(n) <= 12 and " " not in n]
        result = short[:40]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LLM CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_review_prompt(transcript: str) -> str:
    drug_list_str = ""
    candidates = get_candidates(transcript)
    if candidates:
        drug_list_str = (
            "\nRÉFÉRENTIEL MÉDICAMENTS (base officielle française CIS_bdpm) :\n"
            "Si un mot ressemble à l'un de ces médicaments mais est mal orthographié, "
            "propose une correction.\n"
            + ", ".join(candidates)
            + "\n"
        )

    review_schema = {
        "corrections": [
            {
                "original":  "string — le mot ou groupe de mots EXACTEMENT tel qu'il apparait dans la transcription",
                "corrige":   "string — la correction proposee",
                "type":      "string — medicament | dosage | anatomie | autre",
                "confiance": "string — haute (certain) | moyenne (probable) | faible (possible)",
            }
        ],
        "alertes": [
            {
                "texte": "string — le passage concerne",
                "raison":"string — pourquoi ce passage merite attention du medecin",
            }
        ],
    }

    return f"""Tu es un assistant médical expert en relecture de transcriptions de consultations médicales.

TRANSCRIPTION À VÉRIFIER ET CORRIGER :
{transcript}
{drug_list_str}
INSTRUCTIONS :
- Vérifie et corrige les noms de médicaments mal transcrits. Compare avec le référentiel fourni.
- Vérifie les dosages : sont-ils cohérents et plausibles ?
- Vérifie les termes anatomiques et médicaux.
- Corrige les erreurs phonétiques (ex: "pérendopril" -> "périndopril", "turbastatine" -> "atorvastatine").
- Pour chaque correction, "original" doit être le texte EXACT tel qu'il apparait dans la transcription.
- "corrige" doit être UNIQUEMENT le terme corrigé, en français, tel qu'il doit apparaître dans la
  transcription — jamais une phrase, jamais une explication, jamais une traduction en anglais,
  jamais plusieurs options séparées par "ou". Exemple correct : "corrige": "périndopril".
  Exemple INTERDIT : "corrige": "périndopril ou peut-être un autre IEC, à vérifier".
- Si tu n'es pas sûr à au moins "moyenne" confiance d'UN SEUL terme précis, ne mets PAS cette
  entrée dans "corrections" — mets plutôt ton incertitude dans "alertes".
- Ne liste QUE les erreurs médicales réelles — pas de corrections de style ou de grammaire.
- Si aucune erreur médicale n'est détectée, retourne corrections=[].
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RETOURNER :
{json.dumps(review_schema, ensure_ascii=False, indent=2)}
"""


# Phrases that signal the LLM wrote commentary instead of a clean term —
# reject any "corrige" value containing these, regardless of confidence.
_HEDGE_MARKERS = (
    " ou ", "peut-être", "peut etre", "sans doute", "à vérifier", "a verifier",
    "selon le contexte", "aucune correspondance", "correct mais", "sans précision",
    "sans precision", "nom commercial pour", "n'est pas standard", "n est pas standard",
)


def _looks_like_explanation(original: str, corrected: str) -> bool:
    """Reject corrections that are commentary/alternatives rather than a clean term."""
    low = corrected.lower()
    if any(marker in low for marker in _HEDGE_MARKERS):
        return True
    # A real term correction stays roughly the same length as the original.
    # If the model tripled the word count, it's explaining, not correcting.
    orig_words = max(len(original.split()), 1)
    corr_words = len(corrected.split())
    if corr_words > orig_words * 3 + 2:
        return True
    return False


def apply_surgical_corrections(transcript: str, corrections: list[dict]) -> str:
    """
    Apply only the specific word-level substitutions from the LLM corrections list.
    Does NOT use the LLM's full rewrite — leaves all other text untouched.
    Only applies corrections with confiance=haute or moyenne to avoid false positives,
    and rejects any correction that looks like an explanation rather than a clean term
    (the LLM sometimes ignores the prompt and writes a hedged sentence into "corrige",
    which otherwise gets substituted verbatim into the transcript and inflates WER).
    """
    if not corrections:
        return transcript

    result = transcript
    applied = 0
    for c in corrections:
        original  = c.get("original", "").strip()
        corrected = c.get("corrige", "").strip()
        confiance = c.get("confiance", "faible").lower()

        # Only apply high/medium confidence corrections
        if confiance not in ("haute", "moyenne"):
            continue
        if not original or not corrected or original == corrected:
            continue
        if _looks_like_explanation(original, corrected):
            log.info(f"    ✗ rejected (looks like commentary, not a term): "
                      f"'{original}' → '{corrected}'")
            continue

        # Case-insensitive replacement, preserve surrounding context
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        new_result = pattern.sub(corrected, result, count=1)
        if new_result != result:
            log.info(f"    ✓ '{original}' → '{corrected}' ({confiance})")
            result = new_result
            applied += 1

    if applied == 0:
        log.info(f"    No corrections applied")
    return result


def llm_correct(transcript: str, retries: int = 3) -> str:
    """
    Call LLM to get correction suggestions, then apply them surgically.
    Only substitutes specific flagged words — never rewrites the full transcript.
    Returns corrected text or original on failure.
    """
    if not transcript or transcript == "[ERROR]":
        return transcript

    prompt = build_review_prompt(transcript)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="mistral-small-3.2-24b-instruct-2506",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            last_brace = raw.rfind("}")
            if last_brace != -1:
                raw = raw[:last_brace + 1]
            result = json.loads(raw)

            # Use surgical substitutions from corrections list — NOT the full rewrite
            corrections = result.get("corrections", [])
            corrected = apply_surgical_corrections(transcript, corrections)
            return corrected

        except Exception as e:
            log.warning(f"  Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    log.error("  All retries failed, returning original transcript")
    return transcript


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Compute WER and CER using jiwer."""
    try:
        import jiwer
        if not reference or not hypothesis:
            return 1.0, 1.0
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return round(wer, 4), round(cer, 4)
    except Exception as e:
        log.warning(f"  jiwer error: {e}")
        return 1.0, 1.0


from normalizer import MedicalNormalizer  # same class used to build reference_norm
_MED_NORMALIZER = MedicalNormalizer()

def normalize_text(text: str) -> str:
    """
    Must use the SAME normalizer that produced reference_norm, or WER/CER
    become meaningless — you'd be comparing two different text spaces
    (this used to strip accents and skip number/abbrev expansion while
    reference_norm kept accents and expanded everything).
    """
    if not isinstance(text, str):
        return ""
    return _MED_NORMALIZER.normalize(text)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def process_csv(csv_path: Path) -> Path:
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {csv_path.name}")
    log.info(f"{'='*60}")

    df = pd.read_csv(csv_path)
    log.info(f"  Segments: {len(df)}")

    required = {"hypothesis_raw", "reference_norm"}
    missing = required - set(df.columns)
    if missing:
        log.error(f"  Missing columns: {missing} — skipping")
        return None

    corrected_texts  = []
    corrected_norms  = []
    wers_corrected   = []
    cers_corrected   = []
    wers_original    = list(df["wer"]) if "wer" in df.columns else [None] * len(df)
    cers_original    = list(df["cer"]) if "cer" in df.columns else [None] * len(df)

    for idx, row in df.iterrows():
        seg_id   = row.get("segment_id", f"row_{idx}")
        hyp_raw  = str(row["hypothesis_raw"]) if pd.notna(row["hypothesis_raw"]) else ""
        ref_norm = str(row["reference_norm"]) if pd.notna(row["reference_norm"]) else ""

        log.info(f"  [{idx+1}/{len(df)}] {seg_id}")

        # LLM correction
        corrected = llm_correct(hyp_raw)
        corrected_norm = normalize_text(corrected)

        # Recompute metrics
        wer_new, cer_new = compute_wer_cer(ref_norm, corrected_norm)

        corrected_texts.append(corrected)
        corrected_norms.append(corrected_norm)
        wers_corrected.append(wer_new)
        cers_corrected.append(cer_new)

        wer_orig = row.get("wer", None)
        delta = f"{wer_new - wer_orig:+.3f}" if wer_orig is not None else "N/A"
        log.info(f"    WER: {wer_orig:.3f} → {wer_new:.3f} ({delta})")

        time.sleep(0.5)  # rate limit courtesy

    # Add new columns
    df["hypothesis_llm_corrected"]      = corrected_texts
    df["hypothesis_llm_corrected_norm"] = corrected_norms
    df["wer_original"]                  = wers_original
    df["cer_original"]                  = cers_original
    df["wer_llm_corrected"]             = wers_corrected
    df["cer_llm_corrected"]             = cers_corrected
    df["wer_delta"]                     = [
        round(w - o, 4) if o is not None else None
        for w, o in zip(wers_corrected, wers_original)
    ]

    # Save
    out_path = csv_path.parent / (csv_path.stem + "_llm_corrected.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"\n  Saved → {out_path.name}")

    # Summary
    mean_wer_orig = pd.Series(wers_original).dropna().mean()
    mean_wer_corr = pd.Series(wers_corrected).mean()
    mean_cer_orig = pd.Series(cers_original).dropna().mean()
    mean_cer_corr = pd.Series(cers_corrected).mean()
    log.info(f"  WER: {mean_wer_orig:.3f} → {mean_wer_corr:.3f} ({mean_wer_corr - mean_wer_orig:+.3f})")
    log.info(f"  CER: {mean_cer_orig:.3f} → {mean_cer_corr:.3f} ({mean_cer_corr - mean_cer_orig:+.3f})")

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python correct_transcriptions.py results/results_*.csv")
        sys.exit(1)

    # Load drug lookup caches upfront
    log.info("Loading drug reference databases...")
    _load_cis()
    _load_phonetic()

    csv_files = [Path(p) for p in sys.argv[1:]]
    outputs   = []

    for csv_path in csv_files:
        if not csv_path.exists():
            log.error(f"File not found: {csv_path}")
            continue
        out = process_csv(csv_path)
        if out:
            outputs.append(out)

    print(f"\nDone. {len(outputs)} file(s) corrected:")
    for o in outputs:
        print(f"  {o}")