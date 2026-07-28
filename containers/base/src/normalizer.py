"""
normalizer.py — French Medical STT Normalizer
==============================================
Apply to BOTH ground truth AND model outputs before any metric computation.
This guarantees WER/CER comparisons are fair across models.

Usage:
    from normalizer import MedicalNormalizer
    norm = MedicalNormalizer()
    clean = norm.normalize("Le Dr. Dupont prescrit 500 mg d'amoxicilline deux fois par jour.")
    # → "le docteur dupont prescrit 500 mg amoxicilline deux fois par jour"
"""

import re
import unicodedata
# from num2words import num2words  # pip install num2words

# Bump this whenever normalization behavior changes in a way that would
# alter output text — anything reading this module (e.g. freeze_dataset.py)
# should record it in the frozen dataset's metadata, not hardcode a string.
# v2: fixed elision handling (d'/l'/j'/... no longer leave stray letters),
#     fixed composite number combination ("cinq mille" -> 5000, not "5 1000"),
#     fixed step ordering so "X fois par jour" is never partially digitized.
NORMALIZER_VERSION = "medical_fr_v2"

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

# Base values used to COMBINE runs of adjacent number words that aren't
# already covered by an explicit composite entry above (e.g. "cinq mille").
# This is the same underlying data as FR_NUMBER_WORDS restricted to the
# single-word forms, used by the accumulator in _words_to_digits.
_NUMBER_BASE = {k: v for k, v in FR_NUMBER_WORDS.items() if " " not in k and "-" not in k}

# Elided determiners/pronouns: d' l' j' n' s' m' t' c' qu' jusqu' lorsqu' puisqu'
# These are function words, not content — for WER fairness we drop the
# elided particle entirely rather than leaving a stray one-letter token
# ("d'amoxicilline" → "amoxicilline", not "d amoxicilline").
_ELISION_RE = re.compile(
    r"\b(jusqu|lorsqu|puisqu|qu|[ldjnsmtc])['’]",
    re.IGNORECASE
)

# Frequency phrases whose leading number word must NOT be digitized
# (e.g. "deux fois par jour" must stay as words, matching metrics.py's
# _FREQ_TOKENS which looks for the literal word form).
_FREQ_PHRASE_GUARD = re.compile(r"\s+fois\s+par\s+jour\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 2. NORMALIZER CLASS
# ---------------------------------------------------------------------------

class MedicalNormalizer:
    """
    Deterministic, reversible-free text normalizer for French medical STT.

    Steps applied in order:
        1. Unicode NFC + strip accents from punctuation only (keep é, è, etc.)
        2. Lowercase
        3. Elision handling (d' l' j' n' qu' ... → drop the elided particle)
        4. Title / honorific abbreviations
        5. Route-of-administration abbreviations
        6. Unit normalization
        7. French number words → digits (e.g. "cinq cents" → "500"),
           protecting "<word> fois par jour" phrases from digitization
        8. Frequency abbreviations (BID/TID/... → "deux fois par jour" etc.)
           — runs AFTER step 7 so its output phrases are never re-digitized
        9. Digit grouping cleanup       (e.g. "5 00" → "500")
       10. Punctuation removal (keep hyphens inside words, keep digits/units)
       11. Collapse whitespace
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
            self._handle_elisions,
            self._expand_titles,
            self._expand_routes,
            self._normalize_units,
            self._words_to_digits,
            self._expand_frequencies,
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

    # Step 3 — Elision handling
    # Drop elided determiners/pronouns entirely instead of leaving a stray
    # one-letter token behind ("d'amoxicilline" → "amoxicilline", not
    # "d amoxicilline"). Both ref and hyp go through this identically, so
    # comparisons stay fair; it just stops elisions from inflating WER with
    # phantom single-character tokens.
    def _handle_elisions(self, text: str) -> str:
        return _ELISION_RE.sub("", text)

    # Step 4 — Title abbreviations
    def _expand_titles(self, text: str) -> str:
        for abbr, canonical in TITLE_ABBREVS.items():
            # Word boundary aware, case-insensitive (already lowercased)
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', canonical, text)
        return text

    # Step 5 — Route abbreviations
    def _expand_routes(self, text: str) -> str:
        for abbr, canonical in ROUTE_ABBREVS.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', canonical, text)
        return text

    # Step 6 — Unit normalization (words → short form)
    def _normalize_units(self, text: str) -> str:
        for word, canonical in UNIT_ABBREVS.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', canonical, text)
        return text

    # Step 7 — French number words → digits
    #
    # Runs BEFORE frequency-abbreviation expansion (step 8) so that once
    # step 8 introduces a canonical phrase like "deux fois par jour" it is
    # never re-digitized.
    #
    # Also combines adjacent number-word tokens that aren't already an
    # explicit composite entry in FR_NUMBER_WORDS — e.g. "cinq mille" isn't
    # in the table, but "cinq" (5) + "mille" (1000) are adjacent single-word
    # matches, so they're combined into 5000 instead of being substituted
    # independently into the dangerous "5 1000" (which metrics.py's dosage
    # regex would then misread as "1000mg").
    #
    # A run is skipped (left as words) if it's immediately followed by
    # "fois par jour", since that phrase must stay as words to match
    # metrics.py's _FREQ_TOKENS lookup.
    def _words_to_digits(self, text: str) -> str:
        matches = list(self._num_re.finditer(text))
        if not matches:
            return text

        out = []
        cursor = 0
        i = 0
        while i < len(matches):
            # Grow a run of matches separated only by whitespace
            run = [matches[i]]
            j = i + 1
            while j < len(matches):
                between = text[run[-1].end():matches[j].start()]
                # Adjacent number words joined only by whitespace and/or
                # hyphens (e.g. "soixante-quinze", "quatre vingt douze")
                # belong to the same compound number.
                if re.fullmatch(r"[\s\-]*", between):
                    run.append(matches[j])
                    j += 1
                else:
                    break

            run_start, run_end = run[0].start(), run[-1].end()

            # Protect "<...> fois par jour" — leave the whole run as words
            if _FREQ_PHRASE_GUARD.match(text, run_end):
                out.append(text[cursor:run_end])
                cursor = run_end
                i = j
                continue

            values = [FR_NUMBER_WORDS[m.group(0).lower()] for m in run]
            combined = self._combine_number_run(values)

            out.append(text[cursor:run_start])
            out.append(str(combined))
            cursor = run_end
            i = j

        out.append(text[cursor:])
        return "".join(out)

    @staticmethod
    def _combine_number_run(values: list) -> int:
        """Standard French cardinal accumulation: groups of hundreds/thousands
        multiply into the running total, plain units/tens add within a group.
        Single-value runs just return that value unchanged."""
        total = 0
        group = 0
        for v in values:
            if v >= 1000:
                total += (group or 1) * v
                group = 0
            elif v == 100:
                group = (group or 1) * 100
            else:
                group += v
        return total + group

    # Step 8 — Frequency abbreviations (BID/TID/... → words)
    # Runs AFTER _words_to_digits so the phrases it introduces are safe.
    def _expand_frequencies(self, text: str) -> str:
        for abbr, canonical in FREQ_ABBREVS.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', canonical, text)
        return text

    # Step 9 — Fix digit spacing artifacts ("5 00 mg" → "500 mg")
    # This catches cases where composite numbers are replaced sequentially
    def _cleanup_digit_spaces(self, text: str) -> str:
        # Join digit groups separated by a single space if the second group
        # looks like a magnitude suffix (00, 000, 0000)
        text = re.sub(r'(\d)\s+(0{2,4})\b', r'\1\2', text)
        return text

    # Step 10 — Remove punctuation (keep hyphens between letters, keep . in numbers)
    def _remove_punctuation(self, text: str) -> str:
        # Keep: letters (including accented), digits, spaces, hyphens between words
        # Remove: . , ; : ! ? ( ) [ ] " ' « » …
        text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
        # Remove standalone hyphens (not between two word chars)
        text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
        return text

    # Step 11 — Collapse whitespace
    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# 3. QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    norm = MedicalNormalizer(verbose=False)

    test_cases = [
        # (input, expected_output)
        # NOTE: original expected value here dropped "le", inconsistent with
        # test case 4 below which keeps "le" for the identical "Le Pr. X"
        # pattern. Articles are not stripped anywhere in this normalizer, so
        # the correct expectation is "le docteur ...". Fixed accordingly.
        ("Le Dr. Dupont prescrit 500 mg d'amoxicilline.",
         "le docteur dupont prescrit 500 mg amoxicilline"),

        ("Administrer cinq cents milligrammes IV deux fois par jour.",
         "administrer 500 mg intraveineux deux fois par jour"),

        # Critical medical safety case: 5000 mg vs 500 mg must stay distinct
        ("Dose : cinq mille milligrammes",
         "dose 5000 mg"),  # "cinq mille" now correctly combined → 5000, not "5 1000"

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
