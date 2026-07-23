"""
drug_lookup.py — Référentiel médicaments pour Napoleon
=======================================================
Charge le fichier CIS_bdpm.txt (base officielle française) et le lexique
phonétique, et expose deux fonctions :

  get_candidates(transcript)  → liste de noms de médicaments présents ou
                                 phonétiquement proches dans la transcription
  get_all_names()             → liste complète des noms (pour usage direct)

Utilisé par build_review_prompt() dans prompts.py pour enrichir la vérification
sans surcharger le contexte LLM.
"""

import re
import json
import unicodedata
from pathlib import Path
from functools import lru_cache

# ── Paths — cherche les fichiers à côté de ce script ─────────────────────────
_BASE = Path(__file__).parent
CIS_PATH      = _BASE / "CIS_bdpm.txt"
PHONETIC_PATH = _BASE / "Lexique_phonétique_médicaments.json"


def _normalize(s: str) -> str:
    """Lowercase, strip accents, remove non-alpha chars — for fuzzy matching."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


@lru_cache(maxsize=1)
def _load_cis() -> list[str]:
    """Load and cache drug names from CIS_bdpm.txt."""
    if not CIS_PATH.exists():
        return []
    names = set()
    with open(CIS_PATH, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            raw  = parts[1].strip()
            name = re.split(r"\s+\d|\s*,", raw)[0].strip().upper()
            # Skip homeopathic and very short/long names
            if len(name) < 4 or len(name) > 40:
                continue
            if any(s in name for s in ("BOIRON", "LEHNING", "WELEDA")):
                continue
            names.add(name)
    return sorted(names)


@lru_cache(maxsize=1)
def _load_phonetic() -> dict[str, str]:
    """Load phonetic lexicon — returns {normalized_phoneme: original_name}."""
    if not PHONETIC_PATH.exists():
        return {}
    with open(PHONETIC_PATH, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    # Handle both list and dict formats
    if isinstance(data, list):
        for entry in data:
            name  = entry.get("nom") or entry.get("name") or ""
            phon  = entry.get("phoneme") or entry.get("ipa") or entry.get("phonetique") or ""
            if name and phon:
                result[_normalize(phon)] = name.upper()
    elif isinstance(data, dict):
        for name, info in data.items():
            phon = ""
            if isinstance(info, dict):
                phon = info.get("phoneme") or info.get("ipa") or info.get("phonetique") or ""
            elif isinstance(info, str):
                phon = info
            if phon:
                result[_normalize(phon)] = name.upper()
    return result


def get_all_names() -> list[str]:
    """Return the full list of drug names from CIS_bdpm.txt."""
    return _load_cis()


def get_candidates(transcript: str, max_results: int = 60) -> list[str]:
    """
    Return drug names from the reference database that are likely relevant
    to this transcript — either because the name appears directly, or because
    a word in the transcript phonetically resembles a known drug name.

    Keeps the list short (max_results) so it fits comfortably in a prompt.
    """
    all_names  = _load_cis()
    phonetic   = _load_phonetic()

    transcript_norm = _normalize(transcript)
    words = set(re.findall(r"[a-z]{4,}", transcript_norm))

    candidates = set()

    # 1. Direct substring match — drug name appears in transcript
    for name in all_names:
        name_norm = _normalize(name)
        if name_norm in transcript_norm:
            candidates.add(name)

    # 2. Fuzzy word match — transcript word is close to a drug name
    for name in all_names:
        name_norm = _normalize(name)
        for word in words:
            # Simple prefix match (catches spelling variants)
            if len(word) >= 5 and (
                name_norm.startswith(word[:5]) or
                word.startswith(name_norm[:5])
            ):
                candidates.add(name)

    # 3. Phonetic match — transcript word matches a phonetic entry
    for word in words:
        if word in phonetic:
            candidates.add(phonetic[word])

    # Sort and cap
    result = sorted(candidates)[:max_results]

    # If nothing found, return the 40 most common short names as a safety net
    if not result:
        short = [n for n in all_names if len(n) <= 12 and " " not in n]
        result = short[:40]

    return result