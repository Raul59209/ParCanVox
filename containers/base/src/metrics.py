"""
metrics.py — STT Benchmark Metrics
====================================
Computes all metrics defined in the benchmark spec:

    1. WER  — Word Error Rate (jiwer)
    2. CER  — Character Error Rate (jiwer)
    3. Medical entity accuracy — medications, dosages, critical numbers
    4. Operational — latency, RTF, cost per hour of audio

IMPORTANT: Always pass NORMALIZED text to compute_wer / compute_cer.
Run normalizer.normalize() on both ref and hyp before calling anything here.

Usage:
    from metrics import BenchmarkMetrics
    m = BenchmarkMetrics()
    result = m.compute(
        ref="docteur dupont prescrit 500 mg amoxicilline 2 fois par jour",
        hyp="docteur dupont prescrit 5000 mg amoxicilline 2 fois par jour",
        latency_s=1.23,
        audio_duration_s=8.5,
        cost_per_minute=0.001,
    )
    print(result)
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from jiwer import wer, cer


# ---------------------------------------------------------------------------
# 1. RESULT DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class SegmentMetrics:
    # Core ASR metrics
    wer: float                    # 0.0 = perfect, 1.0+ = very bad
    cer: float

    # Medical entity metrics
    med_entities_ref:  list       # entities found in reference
    med_entities_hyp:  list       # entities found in hypothesis
    med_entity_hits:   int        # correctly transcribed entities
    med_entity_total:  int        # total entities in reference
    med_entity_acc:    float      # hit rate (0.0–1.0)
    med_critical_errors: list     # list of critical mismatches (e.g. 500mg vs 5000mg)

    # Operational metrics
    latency_s:         float
    audio_duration_s:  float
    rtf:               float      # latency / audio_duration (lower = faster than realtime)
    cost_usd:          float
    cost_per_hour_audio: float    # extrapolated cost for 1 hour of audio

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten lists to strings for CSV output
        d["med_entities_ref"]    = "; ".join(str(e) for e in self.med_entities_ref)
        d["med_entities_hyp"]    = "; ".join(str(e) for e in self.med_entities_hyp)
        d["med_critical_errors"] = "; ".join(str(e) for e in self.med_critical_errors)
        return d


# ---------------------------------------------------------------------------
# 2. MEDICAL ENTITY EXTRACTOR
# ---------------------------------------------------------------------------

# Dosage pattern: number + unit  e.g. "500 mg", "1.5 g", "20 ml", "2 ui"
_DOSAGE_RE = re.compile(
    r'\b(\d+(?:[.,]\d+)?)\s*(mg|mcg|g|ml|l|ui|meq|mmol|mol)\b',
    re.IGNORECASE
)

# Standalone critical numbers (without unit) — e.g. dilution ratios, concentrations
_NUMBER_RE = re.compile(r'\b(\d{2,}(?:[.,]\d+)?)\b')

# Frequency tokens — already normalized by normalizer.py
_FREQ_TOKENS = {
    "une fois par jour", "deux fois par jour", "trois fois par jour",
    "quatre fois par jour", "si besoin", "per os", "toutes les heures",
}

# Route tokens — already normalized
_ROUTE_TOKENS = {
    "intraveineux", "intramusculaire", "sous cutané",
    "sublingual", "inhalation", "perfusion",
}

# How many characters back to look, from the start of a dosage match, when
# searching for the drug name it belongs to. Tuned to span "drug NAME 75 mg"
# style constructions (including a brand name + a few connecting words)
# without reaching back far enough to accidentally grab an unrelated
# medication mentioned earlier in a long sentence.
_DRUG_ASSOC_WINDOW = 60

# Common French medical drug stems (partial list — extend for your domain)
_DRUG_STEMS = [
    "amoxicilline", "paracétamol", "paracetamol", "ibuprofène", "metformine", "aspirine",
    "oméprazole", "atorvastatine", "metoprolol", "lisinopril", "amlodipine",
    "furosémide", "losartan", "simvastatine", "ramipril", "bisoprolol",
    "prednisone", "prednisolone", "dexamethasone", "cortisone", "insuline",
    "héparine", "warfarine", "clopidogrel", "rivaroxaban", "apixaban",
    "amiodarone", "digoxine", "atenolol", "propranolol", "verapamil",
    "morphine", "codeine", "tramadol", "oxycodone", "fentanyl",
    "diazepam", "lorazepam", "alprazolam", "midazolam", "clonazepam",
    "fluoxetine", "sertraline", "escitalopram", "venlafaxine", "mirtazapine",
    "amoxicilline", "azithromycine", "ciprofloxacine", "doxycycline",
    "metronidazole", "trimethoprime", "cefalexine", "augmentin",
    "salbutamol", "salmeterol", "tiotropium", "fluticasone", "beclometasone",
    "methotrexate", "cyclophosphamide", "tamoxifene", "letrozole",
    "levothyroxine", "carbimazole", "propylthiouracile",
    # ── French consultation drugs (from benchmark audio) ──────────────────
    "périndopril", "perindopril",       # ACE inhibitor — pérendopril is a common whisper error
    "indapamide",                        # diuretic — endopamide is a common whisper error
    "kardégic", "kardegic",             # aspirin 75mg — cardégique is a common whisper error
    "doliprane",                         # paracetamol brand — Dolivier is a common whisper error
    "tanganil",                          # acétylleucine — vertigo treatment
    "acétylleucine", "acetylleucine",   # tanganil DCI
    "lercanidipine",                     # calcium channel blocker
    "valsartan",                         # ARB antihypertensive
    "irbesartan",                        # ARB antihypertensive
    "olmesartan",                        # ARB antihypertensive
    "rosuvastatine", "rosuvastatin",    # statin — turbastatine is a common whisper error
    "ezetimibe",                         # cholesterol
    "allopurinol",                       # gout
    "pantoprazole",                      # PPI
    "esomeprazole",                      # PPI
    "levothyrox",                        # thyroid brand name
    "betahistine", "bétahistine",       # vertigo/meniere
    "serc",                              # betahistine brand
    "métoject", "metoject",             # methotrexate injection — méta-inject is a common whisper error
    "auflox",                            # antibiotic ear drops — eau flossette is a common whisper error
    "amoxicilline",
    "augmentin",                         # amoxicillin/clavulanate
    "clamoxyl",                          # amoxicillin brand
    "celestene", "célestène",           # betamethasone
    "solupred",                          # prednisolone brand
]
_DRUG_RE = re.compile(
    r'\b(' + '|'.join(re.escape(d) for d in _DRUG_STEMS) + r')\b',
    re.IGNORECASE
)


@dataclass
class MedicalEntity:
    type: str        # "dosage" | "drug" | "frequency" | "route" | "number"
    value: str       # normalized string
    raw: str          # original match
    drug: str = ""    # for type=="dosage": nearest associated drug name (lowercase),
                       # "" if none found within _DRUG_ASSOC_WINDOW. Not part of
                       # equality/hash — two dosage entities with the same value
                       # still count as the "same" entity for entity-accuracy
                       # purposes regardless of which drug they were tied to;
                       # `drug` is only used for smarter critical-error pairing.

    def __eq__(self, other):
        return self.type == other.type and self.value == other.value

    def __hash__(self):
        return hash((self.type, self.value))

    def __str__(self):
        return f"{self.type}:{self.value}"


def extract_medical_entities(text: str) -> list[MedicalEntity]:
    """Extract all medical entities from normalized text."""
    entities = []

    # Find all drug name matches first (position + name), so each dosage can
    # look up the nearest PRECEDING drug name to associate with. This is what
    # lets find_critical_errors() compare "périndopril's dose" against
    # "périndopril's dose" instead of against whatever dosage happens to come
    # out of a set first.
    drug_matches = [(m.start(), m.group(0).lower()) for m in _DRUG_RE.finditer(text)]

    # Dosages (number + unit) — highest priority
    for m in _DOSAGE_RE.finditer(text):
        number = m.group(1).replace(",", ".")
        unit = m.group(2).lower()

        assoc_drug = ""
        best_dist = None
        for pos, name in drug_matches:
            if pos < m.start():
                dist = m.start() - pos
                if dist <= _DRUG_ASSOC_WINDOW and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    assoc_drug = name

        entities.append(MedicalEntity(
            type="dosage",
            value=f"{number}{unit}",
            raw=m.group(0),
            drug=assoc_drug,
        ))

    # Drug names
    for pos, name in drug_matches:
        entities.append(MedicalEntity(
            type="drug",
            value=name,
            raw=name,
        ))

    # Frequencies
    for freq in _FREQ_TOKENS:
        if freq in text:
            entities.append(MedicalEntity(type="frequency", value=freq, raw=freq))

    # Routes
    for route in _ROUTE_TOKENS:
        if route in text:
            entities.append(MedicalEntity(type="route", value=route, raw=route))

    return entities


def _split_dosage(dosage_value: str) -> tuple[float, str]:
    """Split '500mg' → (500.0, 'mg'). Returns (0.0, '') on failure."""
    m = re.match(r'^([\d.]+)([a-z]+)$', dosage_value)
    if not m:
        return 0.0, ""
    try:
        return float(m.group(1)), m.group(2)
    except ValueError:
        return 0.0, m.group(2)


def find_critical_errors(
    ref_entities: list[MedicalEntity],
    hyp_entities: list[MedicalEntity],
) -> list[str]:
    """
    Find medically critical mismatches — especially dosage errors.
    A "500mg" vs "5000mg" confusion counts as only 1 WER error
    but is clinically dangerous. We surface these explicitly.

    Dosages are compared PER-DRUG whenever a drug association is available
    (see extract_medical_entities), instead of just grabbing an arbitrary
    same-unit dosage from a set. This stops nonsensical comparisons like
    "périndopril's 4mg" being flagged against "paracétamol's 500mg" just
    because both happen to be in "mg" and one came first out of set iteration
    order. When no drug is known for a given number (e.g. it sits too far
    from any recognized drug name), we fall back to matching it against the
    numerically closest same-unit dosage rather than a random one — safer,
    but still not as reliable as a real drug association, so treat those
    specifically as lower-confidence if you're triaging critical_errors.
    """
    errors = []

    ref_dosages = [e for e in ref_entities if e.type == "dosage"]
    hyp_dosages = [e for e in hyp_entities if e.type == "dosage"]
    ref_set = set(ref_dosages)
    hyp_set = set(hyp_dosages)

    def group_by_drug(entities):
        groups: dict[str, list] = {}
        for e in entities:
            groups.setdefault(e.drug, []).append(e)
        return groups

    ref_by_drug = group_by_drug(ref_dosages)
    hyp_by_drug = group_by_drug(hyp_dosages)

    matched_hyp_ids = set()  # id() of hyp entities already used, so one hyp
                              # dosage can't be "reused" to explain away
                              # multiple different ref mismatches

    for drug, ref_list in ref_by_drug.items():
        for ref_e in ref_list:
            if ref_e in hyp_set:
                continue  # an exact (value) match exists somewhere in hyp — fine

            ref_num, ref_unit = _split_dosage(ref_e.value)
            drug_label = f" [{drug}]" if drug else ""

            if drug:
                # Known drug on the ref side: ONLY compare against hyp dosages
                # associated with that SAME drug. If hyp has no dosage tied to
                # this drug at all, that's a MISSING dosage — never silently
                # substitute a different drug's number.
                same_drug_candidates = [
                    e for e in hyp_by_drug.get(drug, [])
                    if _split_dosage(e.value)[1] == ref_unit and id(e) not in matched_hyp_ids
                ]
                candidates = same_drug_candidates
            else:
                # No drug could be associated with this ref number (too far
                # from any recognized drug name). Fall back to the closest
                # same-unit hyp dosage among ALL unmatched hyp dosages —
                # better than arbitrary set order, but flagged as such via
                # the empty drug_label so it's visibly lower-confidence.
                candidates = [
                    e for e in hyp_dosages
                    if _split_dosage(e.value)[1] == ref_unit and id(e) not in matched_hyp_ids
                ]

            if candidates:
                hyp_e = min(candidates, key=lambda e: abs(_split_dosage(e.value)[0] - ref_num))
                matched_hyp_ids.add(id(hyp_e))
                hyp_num, _ = _split_dosage(hyp_e.value)
                if ref_num > 0:
                    errors.append(
                        f"DOSAGE_MISMATCH{drug_label}: ref={ref_e.value} hyp={hyp_e.value} "
                        f"(ratio={hyp_num/ref_num:.1f}x)"
                    )
                else:
                    errors.append(f"DOSAGE_MISMATCH{drug_label}: ref={ref_e.value} hyp={hyp_e.value}")
            else:
                errors.append(f"DOSAGE_MISSING{drug_label}: ref={ref_e.value} not found in hypothesis")

    # Dosages in hyp but not in ref at all (hallucinated dosages), and not
    # already consumed as the "corrected value" side of a mismatch above.
    ref_units = {_split_dosage(e.value)[1] for e in ref_dosages}
    for hyp_e in hyp_dosages:
        if id(hyp_e) in matched_hyp_ids:
            continue
        if hyp_e in ref_set:
            continue
        hyp_unit = _split_dosage(hyp_e.value)[1]
        if hyp_unit not in ref_units:
            drug_label = f" [{hyp_e.drug}]" if hyp_e.drug else ""
            errors.append(f"DOSAGE_HALLUCINATED{drug_label}: hyp={hyp_e.value} not in reference")

    return errors


# ---------------------------------------------------------------------------
# 3. MAIN METRICS CLASS
# ---------------------------------------------------------------------------

class BenchmarkMetrics:
    """
    Compute all benchmark metrics for a single segment.

    Both ref and hyp must already be normalized (via MedicalNormalizer)
    before being passed here.
    """

    def compute(
        self,
        ref: str,
        hyp: str,
        latency_s: float,
        audio_duration_s: float,
        cost_per_minute: float = 0.0,   # set to 0 for local models
    ) -> SegmentMetrics:

        # ── WER / CER ──────────────────────────────────────────────────────
        segment_wer = wer(ref, hyp) if ref.strip() else 0.0
        segment_cer = cer(ref, hyp) if ref.strip() else 0.0

        # ── Medical entities ───────────────────────────────────────────────
        ref_entities = extract_medical_entities(ref)
        hyp_entities = extract_medical_entities(hyp)

        ref_set = set(ref_entities)
        hyp_set = set(hyp_entities)
        hits = len(ref_set & hyp_set)
        total = len(ref_set)
        acc = hits / total if total > 0 else 1.0  # no entities = trivially correct

        critical_errors = find_critical_errors(ref_entities, hyp_entities)

        # ── Operational ────────────────────────────────────────────────────
        rtf = latency_s / audio_duration_s if audio_duration_s > 0 else -1.0
        cost_usd = (audio_duration_s / 60) * cost_per_minute
        cost_per_hour = (cost_usd / audio_duration_s) * 3600 if audio_duration_s > 0 else 0.0

        return SegmentMetrics(
            wer=round(segment_wer, 4),
            cer=round(segment_cer, 4),
            med_entities_ref=ref_entities,
            med_entities_hyp=hyp_entities,
            med_entity_hits=hits,
            med_entity_total=total,
            med_entity_acc=round(acc, 4),
            med_critical_errors=critical_errors,
            latency_s=round(latency_s, 3),
            audio_duration_s=round(audio_duration_s, 2),
            rtf=round(rtf, 3),
            cost_usd=round(cost_usd, 6),
            cost_per_hour_audio=round(cost_per_hour, 4),
        )

    def compute_corpus(self, results: list[SegmentMetrics]) -> dict:
        """
        Aggregate per-segment metrics into corpus-level summary.
        Uses macro average (mean of per-segment scores).
        """
        if not results:
            return {}

        n = len(results)
        total_ref_words = 0   # for micro-average WER (optional)

        wers  = [r.wer for r in results]
        cers  = [r.cer for r in results]
        accs  = [r.med_entity_acc for r in results]
        rtfs  = [r.rtf for r in results if r.rtf >= 0]
        costs = [r.cost_per_hour_audio for r in results if r.cost_per_hour_audio > 0]

        total_audio  = sum(r.audio_duration_s for r in results)
        total_cost   = sum(r.cost_usd for r in results)
        all_critical = [e for r in results for e in r.med_critical_errors]

        return {
            "n_segments":           n,
            "total_audio_s":        round(total_audio, 1),
            "total_audio_min":      round(total_audio / 60, 2),

            # WER
            "wer_mean":             round(sum(wers) / n, 4),
            "wer_median":           round(sorted(wers)[n // 2], 4),
            "wer_worst":            round(max(wers), 4),
            "wer_best":             round(min(wers), 4),

            # CER
            "cer_mean":             round(sum(cers) / n, 4),
            "cer_median":           round(sorted(cers)[n // 2], 4),

            # Medical entities
            "med_entity_acc_mean":  round(sum(accs) / n, 4),
            "n_critical_errors":    len(all_critical),
            "critical_errors":      all_critical,

            # Operational
            "rtf_mean":             round(sum(rtfs) / len(rtfs), 3) if rtfs else -1,
            "rtf_median":           round(sorted(rtfs)[len(rtfs) // 2], 3) if rtfs else -1,
            "total_cost_usd":       round(total_cost, 4),
            "cost_per_hour_audio":  round(costs[0], 4) if costs else 0.0,
        }


# ---------------------------------------------------------------------------
# 4. SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from normalizer import MedicalNormalizer
    norm = MedicalNormalizer()
    m    = BenchmarkMetrics()

    print("=" * 60)
    print("metrics.py — self-test")
    print("=" * 60)

    test_cases = [
        {
            "name": "Perfect match",
            "ref": "docteur dupont prescrit 500 mg amoxicilline",
            "hyp": "docteur dupont prescrit 500 mg amoxicilline",
            "expect_wer": 0.0,
            "expect_critical": 0,
        },
        {
            "name": "Critical dosage error (500mg → 5000mg)",
            "ref": "administrer 500 mg de morphine intraveineux",
            "hyp": "administrer 5000 mg de morphine intraveineux",
            "expect_wer": 0.167,   # 1 word wrong out of 6
            "expect_critical": 1,
        },
        {
            "name": "Drug name misheard",
            "ref": "prendre 2 comprimés de paracétamol",
            "hyp": "prendre 2 comprimés de paracétamol",
            "expect_wer": 0.0,
            "expect_critical": 0,
        },
        {
            "name": "Missing dosage",
            "ref": "insuline 10 ui sous cutané deux fois par jour",
            "hyp": "insuline sous cutané deux fois par jour",
            "expect_wer": 0.375,
            "expect_critical": 1,
        },
        {
            "name": "Two drugs, same unit — dosages must NOT cross-match",
            "ref": "périndopril 4 mg le matin et paracétamol 500 mg le soir",
            "hyp": "périndopril 5 mg le matin et paracétamol 500 mg le soir",
            "expect_wer": 0.083,
            "expect_critical": 1,   # only périndopril's dose is wrong;
                                     # paracétamol's correct 500mg must NOT
                                     # get dragged into a false mismatch
        },
    ]

    all_passed = True
    results_for_corpus = []

    for tc in test_cases:
        ref_n = norm.normalize(tc["ref"])
        hyp_n = norm.normalize(tc["hyp"])
        result = m.compute(
            ref=ref_n,
            hyp=hyp_n,
            latency_s=1.5,
            audio_duration_s=5.0,
            cost_per_minute=0.001,
        )
        results_for_corpus.append(result)

        wer_ok  = abs(result.wer - tc["expect_wer"]) < 0.05
        crit_ok = len(result.med_critical_errors) == tc["expect_critical"]
        ok = wer_ok and crit_ok
        if not ok:
            all_passed = False

        status = "✓" if ok else "✗"
        print(f"\n{status} {tc['name']}")
        print(f"  REF: {ref_n}")
        print(f"  HYP: {hyp_n}")
        print(f"  WER={result.wer:.3f} (expected ~{tc['expect_wer']:.3f}) | "
              f"CER={result.cer:.3f}")
        print(f"  Med entities REF: {[str(e) for e in result.med_entities_ref]}")
        print(f"  Med entities HYP: {[str(e) for e in result.med_entities_hyp]}")
        print(f"  Entity acc: {result.med_entity_acc:.2f} | "
              f"Critical errors: {result.med_critical_errors}")
        print(f"  RTF={result.rtf:.2f} | Cost/hr=${result.cost_per_hour_audio:.4f}")

    # Corpus summary
    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    corpus = m.compute_corpus(results_for_corpus)
    for k, v in corpus.items():
        if k != "critical_errors":
            print(f"  {k}: {v}")
    if corpus.get("critical_errors"):
        print(f"  critical_errors:")
        for e in corpus["critical_errors"]:
            print(f"    ⚠️  {e}")

    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests FAILED ✗"))