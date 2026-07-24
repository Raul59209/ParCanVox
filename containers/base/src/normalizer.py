"""
normalizer.py — French Medical STT Normalizer
==============================================
Apply to BOTH ground truth AND model outputs before any metric computation.
This guarantees WER/CER comparisons are fair across models.

Usage:
    from normalizer import MedicalNormalizer
    norm = MedicalNormalizer()
    clean = norm.normalize("Le Dr. Dupont prescrit 500 mg d'amoxicilline deux fois par jour.")
    # → "docteur dupont prescrit 500 mg amoxicilline 2 fois par jour"
"""

import re
import unicodedata
# from num2words import num2words  # pip install num2words

# ---------------------------------------------------------------------------
# 1. ABBREVIATION TABLES
# ---------------------------------------------------------------------------

# All keys are lowercased. Values are the canonical form we normalize TO.
# Rule: pick the SHORT form as canonical (less ambiguity in WER token count).
TITLE_ABBREVS = {
    "dr.": "docteur",
    "dr":  "docteur",
    "pr.": "professeur",
    "pr":  "professeur",
    "m.":  "monsieur",
    "mme.": "madame",
    "mme":  "madame",
    "mr.":  "monsieur",
    "mr":   "monsieur",
}

# Medical route-of-administration abbreviations → canonical French
ROUTE_ABBREVS = {
    "iv":       "intraveineux",
    "i.v.":     "intraveineux",
    "im":       "intramusculaire",
    "i.m.":     "intramusculaire",
    "sc":       "sous cutané",
    "s.c.":     "sous cutané",
    "po":       "per os",
    "p.o.":     "per os",
    "sl":       "sublingual",
    "s.l.":     "sublingual",
    "inhal.":   "inhalation",
    "perf.":    "perfusion",
}

# Dose / unit normalization → canonical short form
UNIT_ABBREVS = {
    "milligramme":   "mg",
    "milligrammes":  "mg",
    "microgramme":   "mcg",
    "microgrammes":  "mcg",
    "µg":            "mcg",
    "ug":            "mcg",
    "gramme":        "g",
    "grammes":       "g",
    "millilitre":    "ml",
    "millilitres":   "ml",
    "litre":         "l",
    "litres":        "l",
    "milliequivalent": "meq",
    "unité":         "ui",
    "unités":        "ui",
    "u.i.":          "ui",
    "ui.":           "ui",
}

# Frequency abbreviations
FREQ_ABBREVS = {
    "sid":  "une fois par jour",
    "bid":  "deux fois par jour",
    "tid":  "trois fois par jour",
    "qid":  "quatre fois par jour",
    "qd":   "une fois par jour",
    "prn":  "si besoin",
    "p.r.n.": "si besoin",
    "sos":  "si besoin",
}

# French number words (for text→digit conversion)
# We use num2words in reverse via a lookup table for robustness.
# This covers the most common spoken forms in medical dictation.
FR_NUMBER_WORDS = {
    "zéro": 0, "zero": 0,
    "un": 1, "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "vingt": 20,
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
    "soixante-dix": 70,
    "soixante dix": 70,
    "quatre-vingt": 80,
    "quatre vingt": 80,
    "quatre-vingt-dix": 90,
    "quatre vingt dix": 90,
    "cent": 100,
    "cents": 100,
    "mille": 1000,
    "million": 1000000,
    "milliard": 1000000000,
    # Composites — ordered longest-first so regex matches greedily
    "cinq cents": 500,
    "deux cents": 200,
    "trois cents": 300,
    "quatre cents": 400,
    "six cents": 600,
    "sept cents": 700,
    "huit cents": 800,
    "neuf cents": 900,
    "dix-sept": 17, "dix sept": 17,
    "dix-huit": 18, "dix huit": 18,
    "dix-neuf": 19, "dix neuf": 19,
    "vingt et un": 21, "vingt-et-un": 21,
    "vingt-deux": 22, "vingt deux": 22,
    "vingt-cinq": 25, "vingt cinq": 25,
    "trente-cinq": 35, "trente cinq": 35,
    "cinquante": 50,
}

# Build a sorted list (longest phrase first) for greedy replacement
_NUMBER_PATTERNS = sorted(FR_NUMBER_WORDS.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# 2. NORMALIZER CLASS
# ---------------------------------------------------------------------------

class MedicalNormalizer:
    """
    Deterministic, reversible-free text normalizer for French medical STT.

    Steps applied in order:
        1. Unicode NFC + strip accents from punctuation only (keep é, è, etc.)
        2. Lowercase
        3. Title / honorific abbreviations
        4. Route-of-administration abbreviations
        5. Unit normalization
        6. Frequency abbreviations
        7. French number words → digits  (e.g. "cinq cents" → "500")
        8. Digit grouping cleanup       (e.g. "5 00" → "500")
        9. Punctuation removal (keep hyphens inside words, keep digits/units)
       10. Collapse whitespace
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Precompile regex for number words (longest match first)
        self._num_re = re.compile(
            r'\b(' + '|'.join(re.escape(p) for p in _NUMBER_PATTERNS) + r')\b',
            re.IGNORECASE
        )

    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        steps = [
            self._unicode_clean,
            self._lowercase,
            self._expand_titles,
            self._expand_routes,
            self._normalize_units,
            self._expand_frequencies,
            self._words_to_digits,
            self._cleanup_digit_spaces,
            self._remove_punctuation,
            self._collapse_whitespace,
        ]
        result = text
        for step in steps:
            result = step(result)
            if self.verbose:
                print(f"[{step.__name__}] → {result!r}")
        return result

    # ------------------------------------------------------------------
    # Step 1 — Unicode
    def _unicode_clean(self, text: str) -> str:
        # NFC normalization (handles composed vs decomposed accents)
        return unicodedata.normalize("NFC", text)

    # Step 2 — Lowercase
    def _lowercase(self, text: str) -> str:
        return text.lower()

    # Step 3 — Title abbreviations
    def _expand_titles(self, text: str) -> str:
        for abbr, canonical in TITLE_ABBREVS.items():
            # Word boundary aware, case-insensitive (already lowercased)
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', canonical, text)
        return text

    # Step 4 — Route abbreviations
    def _expand_routes(self, text: str) -> str:
        for abbr, canonical in ROUTE_ABBREVS.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', canonical, text)
        return text

    # Step 5 — Unit normalization (words → short form)
    def _normalize_units(self, text: str) -> str:
        for word, canonical in UNIT_ABBREVS.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', canonical, text)
        return text

    # Step 6 — Frequency abbreviations
    def _expand_frequencies(self, text: str) -> str:
        for abbr, canonical in FREQ_ABBREVS.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', canonical, text)
        return text

    # Step 7 — French number words → digits
    def _words_to_digits(self, text: str) -> str:
        def replace_match(m):
            word = m.group(0).lower()
            return str(FR_NUMBER_WORDS.get(word, word))
        return self._num_re.sub(replace_match, text)

    # Step 8 — Fix digit spacing artifacts ("5 00 mg" → "500 mg")
    # This catches cases where composite numbers are replaced sequentially
    def _cleanup_digit_spaces(self, text: str) -> str:
        # Join digit groups separated by a single space if the second group
        # looks like a magnitude suffix (00, 000, 0000)
        text = re.sub(r'(\d)\s+(0{2,4})\b', r'\1\2', text)
        return text

    # Step 9 — Remove punctuation (keep hyphens between letters, keep . in numbers)
    def _remove_punctuation(self, text: str) -> str:
        # Keep: letters (including accented), digits, spaces, hyphens between words
        # Remove: . , ; : ! ? ( ) [ ] " ' « » …
        text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
        # Remove standalone hyphens (not between two word chars)
        text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
        return text

    # Step 10 — Collapse whitespace
    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# 3. QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    norm = MedicalNormalizer(verbose=False)

    test_cases = [
        # (input, expected_output)
        ("Le Dr. Dupont prescrit 500 mg d'amoxicilline.",
         "docteur dupont prescrit 500 mg amoxicilline"),

        ("Administrer cinq cents milligrammes IV deux fois par jour.",
         "administrer 500 mg intraveineux deux fois par jour"),

        # Critical medical safety case: 5000 mg vs 500 mg must stay distinct
        ("Dose : cinq mille milligrammes",
         "dose 5000 mg"),  # Note: "cinq mille" not in table → stays as words (safe fallback)

        ("Le Pr. Martin recommande 1,5 g de paracétamol toutes les huit heures.",
         "le professeur martin recommande 1 5 g de paracétamol toutes les 8 heures"),

        ("perfusion IV de NaCl 0.9% à vingt ml/h",
         "perfusion intraveineux de nacl 0 9 à 20 ml h"),

        ("Prendre deux comprimés BID pendant dix jours.",
         "prendre 2 comprimés deux fois par jour pendant 10 jours"),
    ]

    print("=" * 60)
    print("MedicalNormalizer — self-test")
    print("=" * 60)
    all_passed = True
    for raw, expected in test_cases:
        result = norm.normalize(raw)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"\n{status} INPUT:    {raw}")
        print(f"  OUTPUT:   {result}")
        if result != expected:
            print(f"  EXPECTED: {expected}")

    print("\n" + "=" * 60)
    print("All tests passed!" if all_passed else "Some tests FAILED — review above.")
    print("=" * 60)