"""
notebook_whisperx_chunked.py — Benchmark: WhisperX 3.8.4, silence-chunked
============================================================================
Model:   WhisperX large-v3
Backend: whisperx (faster-whisper + forced alignment under the hood)

FIX (from un-chunked notebook's confirmed OOM bug):
  seg_0002 crashed with "CUDA failed with error out of memory", and the
  VRAM fragmentation it left behind immediately broke seg_0003 with
  "cudaErrorInvalidDevice: invalid device ordinal" — a corrupted CUDA
  context cascading from the first failure. Fix: clear torch's CUDA cache
  after every chunk (success or failure), and retry once at batch_size=1
  if a chunk OOMs instead of giving up immediately.

Run:
    python notebook_whisperx_chunked.py
"""

import sys
import os
import json
import jiwer
import time
import logging
import tempfile
from pathlib import Path

import pandas as pd
import torch
import whisperx
from pydub import AudioSegment
from pydub.silence import split_on_silence
from openai import OpenAI

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import argparse

MODEL_SIZE     = "large-v3"
LANGUAGE       = "fr"
DATASET_PATH   = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR = Path("audio")
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SILENCE_THRESH_DB   = -40
SILENCE_MIN_LEN_MS  = 700
SILENCE_KEEP_MS     = 300

# ---------------------------------------------------------------------------
# initial_prompt variants — pick ONE per run via --prompt-variant.
# Whisper's initial_prompt has a hard ~224-token cap, so these can NOT be
# combined with each other or with the generic prompt; each is already
# tuned right up against that limit on its own (token counts noted below,
# per the source files, using openai-whisper's multilingual tokenizer).
# ---------------------------------------------------------------------------
PROMPT_VARIANTS = {
    # Original generic cross-specialty prompt (baseline — not ORL-specific,
    # covers common drugs across cardio/rheum/neuro/ORL seen in this dataset).
    "generic": (
        "Transcription d'une consultation médicale en français. "
        "Médicaments courants : périndopril, indapamide, metformine, atorvastatine, "
        "clopidogrel, méthotrexate, prednisolone, amoxicilline, oflocet, nasonex, "
        "doliprane, ibuprofène, oméprazole, lévothyroxine, amlodipine. "
        "Termes médicaux : otoscopie, adénoïdectomie, amygdalectomie, "
        "aérateurs trans-tympaniques, tympanométrie, audiogramme, presbyacousie, "
        "polyarthrite rhumatoïde, anti-CCP, DAS28-CRP, biothérapie anti-TNF, "
        "paralysie faciale périphérique, vertige positionnel paroxystique bénin, "
        "manœuvre de Semont, otite externe, otite séromuqueuse, dysphonie, "
        "laryngite, reflux gastro-œsophagien, nodules vocaux. "
        "Unités : mg, g, UI, ml, mg/L."
    ),

    # v2 of the generic prompt — informed by the 'error_words' output of the
    # first generic run. Two changes from v1:
    #   1. Added "antécédents" — the single most frequent real substitution
    #      error across that run (3x), a common ORL/general-medicine term
    #      that wasn't in any of the three ORL-specific lists either.
    #   2. Added an explicit sentence explaining that "point" / "à la ligne"
    #      are spoken dictation punctuation, not content — these were
    #      consistently mishandled (deleted, misplaced, or hallucinated as
    #      real words) in the dictation-style segments specifically. Framed
    #      as natural prose, matching the format that outperformed the
    #      comma-list ORL prompts on dictation content in the earlier test.
    "generic_v2": (
        "Transcription d'une consultation ou d'une dictée médicale en français. "
        "Le médecin dicte parfois des lettres en disant à voix haute des "
        "instructions de ponctuation comme point ou à la ligne, qui ne font "
        "pas partie du contenu médical. "
        "Antécédents du patient, médicaments courants : périndopril, "
        "indapamide, metformine, atorvastatine, clopidogrel, méthotrexate, "
        "prednisolone, amoxicilline, oflocet, nasonex, doliprane, "
        "ibuprofène, oméprazole, lévothyroxine, amlodipine. "
        "Termes médicaux : otoscopie, adénoïdectomie, amygdalectomie, "
        "aérateurs trans-tympaniques, tympanométrie, audiogramme, "
        "presbyacousie, polyarthrite rhumatoïde, anti-CCP, DAS28-CRP, "
        "biothérapie anti-TNF, paralysie faciale périphérique, vertige "
        "positionnel paroxystique bénin, manœuvre de Semont, otite externe, "
        "otite séromuqueuse, dysphonie, laryngite, reflux "
        "gastro-œsophagien, nodules vocaux. Unités : mg, g, UI, ml, mg/L."
    ),

    # generic_v3 — the "merge instead of route" approach. Since the auto-
    # classifier experiment showed routing per-file to a specialist prompt
    # costs 2x compute and, when misclassification happens (which it did,
    # 4/7 dictation segments routed wrong), performs WORSE than just using
    # one good prompt for everything — this folds the highest-value ORL
    # terms directly into generic_v2 as ONE prompt, natural-prose format
    # (the format that has consistently won), no routing/classification
    # needed at all.
    # Term selection priority:
    #   1. Terms with DIRECT evidence from error_words analysis: "confrère"
    #      (2x substitution error — whisperx struggles with this word even
    #      un-prompted) and "endobuccal" (2x substitution error).
    #   2. High-frequency ORL clinical terms across the three specialist
    #      lists, prioritizing ones NOT already in generic_v2.
    # NOTE: word/char count is a rough proxy, not Whisper's actual BPE
    # tokenizer — verify against the ~224 token cap before relying on this;
    # if over budget, trim "otospongiose"/"cholestéatome" first (lowest-
    # confidence additions, no direct error-data backing) before touching
    # anything with direct evidence behind it.
    "generic_v3": (
        "Transcription d'une consultation ou d'une dictée médicale en français. "
        "Le médecin dicte parfois des lettres à des confrères, en disant à voix "
        "haute des instructions de ponctuation comme point ou à la ligne, qui ne "
        "font pas partie du contenu médical. Antécédents du patient, examen "
        "endobuccal, médicaments courants : périndopril, indapamide, metformine, "
        "atorvastatine, clopidogrel, méthotrexate, prednisolone, amoxicilline, "
        "oflocet, nasonex, doliprane, ibuprofène, oméprazole, lévothyroxine, "
        "amlodipine. Termes ORL fréquents : acouphènes, hypoacousie, otalgie, "
        "rhinosinusite, cholestéatome, otospongiose, vertige positionnel "
        "paroxystique bénin, otoscopie, audiométrie, tympanométrie, "
        "tympanoplastie, myringotomie, adénoïdectomie, amygdalectomie, "
        "aérateurs trans-tympaniques, presbyacousie. Autres termes : "
        "polyarthrite rhumatoïde, anti-CCP, DAS28-CRP, biothérapie anti-TNF, "
        "paralysie faciale périphérique, otite externe, otite séromuqueuse, "
        "dysphonie, laryngite, reflux gastro-œsophagien. Unités : mg, g, UI, ml, mg/L."
    ),

    # ---- v2: natural-sentence rewrites of the three ORL lists -------------
    # Same underlying vocabulary as orl_dictee/orl_assistant/orl_cro above,
    # restructured as prose instead of a comma-separated list. Directly
    # tests the working hypothesis from the generic-vs-ORL comparison: that
    # list-format prompts may signal "expect clipped, non-conversational
    # speech" to the decoder, actively working against natural dictation —
    # which would explain why generic (prose) beat all three ORL lists even
    # on the dictation content orl_cro was built for.
    # NOTE: token counts below are estimates (word/char-based), not run
    # through Whisper's actual tokenizer — verify against the ~224 token
    # cap before relying on these; trim the tail clause first if over.
    "orl_dictee_v2": (
        "Consultation ORL : le médecin évoque des traitements comme la "
        "bétaméthasone, la mométasone, la triamcinolone, la fluticasone, "
        "la desloratadine, la cétirizine et le solupred, ainsi que des "
        "gouttes auriculaires telles que l'otipax et le panotile. Il "
        "prescrit ou commente des examens : audiométrie tonale, "
        "audiométrie vocale, tympanométrie, impédancemétrie, "
        "vidéonystagmographie, nasofibroscopie, réflexe stapédien, "
        "potentiels évoqués auditifs et otoémissions acoustiques, ainsi "
        "que la manœuvre de Dix-Hallpike. Il mentionne parfois le cavum, "
        "le méat moyen, un nystagmus ou une névrite."
    ),

    "orl_assistant_v2": (
        "Le patient consulte pour des symptômes ORL : rhinosinusite, "
        "acouphènes, hypoacousie, otalgie, dysphonie, dysphagie, "
        "odynophagie ou épistaxis. Le médecin évoque des diagnostics "
        "possibles comme un cholestéatome, une adénopathie, un reflux "
        "pharyngolaryngé, la maladie de Ménière, une otospongiose, une "
        "rhinorrhée, une anosmie, un vertige positionnel paroxystique "
        "bénin (VPPB), un neurinome de l'acoustique, une presbyacousie ou "
        "une otite séromuqueuse. Il examine le cavum et le méat moyen, et "
        "recherche un nystagmus ou une névrite."
    ),

    "orl_cro_v2": (
        "Chers confrères, je vous adresse ce compte-rendu opératoire. "
        "L'intervention réalisée était une amygdalectomie, une "
        "adénoïdectomie, une septoplastie, une turbinectomie, une "
        "méatotomie moyenne, une ethmoïdectomie ou la pose d'un aérateur "
        "transtympanique. D'autres gestes possibles incluent une "
        "tympanoplastie, une myringotomie, une mastoïdectomie, une "
        "stapédectomie, une parotidectomie, une thyroïdectomie avec curage "
        "ganglionnaire, ou une microchirurgie laryngée en laryngoscopie "
        "en suspension. Le compte-rendu décrit l'anatomie rencontrée : "
        "cornet inférieur, nerf récurrent, sterno-cléido-mastoïdien."
    ),

    # Source: "Prompt WhisperX — Mode dictée ORL.md" (222/224 tokens)
    # Meds, devices, tests — best fit for routine ORL consultations.
    "orl_dictee": (
        "Bétaméthasone, Mométasone, Triamcinolone, Fluticasone, Desloratadine, "
        "Cétirizine, Bilastine, Solupred, Rhinofluimucil, Otipax, Panotile, "
        "Aérius, Audiométrie tonale, Audiométrie vocale, Tympanométrie, "
        "Impédancemétrie, Vidéonystagmographie, Nasofibroscopie, Réflexe "
        "stapédien, Potentiels évoqués auditifs, Otoémissions acoustiques, "
        "Manœuvre de Dix-Hallpike, EU-TIRADS, Score d'Epworth, PEA, Weber, "
        "RGO, Cavum, Méat moyen, Bethesda, VNS, VNG, Nystagmus, Névrite, "
        "Sténon, Wharton, Nasonex, Rhinomaxil, Rhinocort, Nasacort, Derinox, "
        "Aturgyl, Oflocet, Ciloxadex"
    ),

    # Source: "Prompt WhisperX — Mode assistant ORL.md" (218/224 tokens)
    # Symptoms/diagnoses — best fit for narrative diagnostic segments.
    "orl_assistant": (
        "Rhinosinusite, Acouphènes, Hypoacousie, Otalgie, Dysphonie, "
        "Dysphagie, Odynophagie, Épistaxis, SAOS, Cholestéatome, Adénopathie, "
        "Reflux pharyngolaryngé, Maladie de Ménière, Otospongiose, Cophose, "
        "Rhinorrhée, Anosmie, Dysgueusie, Papillomatose laryngée, VPPB, "
        "Neurinome de l'acoustique, Presbyacousie, Otite séromuqueuse, "
        "Otorrhée, Otomycose, Laryngomalacie, Papillome inversé, Carcinome "
        "épidermoïde, Paralysie récurrentielle, RGO, Cavum, Méat moyen, "
        "EU-TIRADS, Bethesda, VNS, VNG, Nystagmus, Névrite, Sténon, Wharton, "
        "Nasofibroscopie"
    ),

    # Source: "Prompt WhisperX — Dictée CRO ORL.md" (206/224 tokens)
    # Surgical/anatomical — best fit for "chers confrères" operative letters
    # (matches seg_0006, seg_0009, seg_0012, seg_0015, seg_0018 closely).
    "orl_cro": (
        "Amygdalectomie, Adénoïdectomie, Septoplastie, Turbinectomie, "
        "Méatotomie moyenne, Ethmoïdectomie, Sphénoïdotomie, "
        "Dacryocystorhinostomie, Tympanoplastie, Myringotomie, Aérateur "
        "transtympanique, Mastoïdectomie, Stapédectomie, Ossiculoplastie, "
        "Parotidectomie, Thyroïdectomie, Loboisthmectomie, Curage "
        "ganglionnaire, Trachéotomie, Laryngectomie, Cordectomie, "
        "Microchirurgie laryngée, Laryngoscopie en suspension, "
        "Septorhinoplastie, Méchage nasal, Nerf récurrent, Cornet inférieur, "
        "Platysma, Mentonnier, Digastrique, Sterno-cléido-mastoïdien, "
        "Pointer, Hypoglosse, Spinal"
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt-variant", choices=sorted(PROMPT_VARIANTS.keys()) + ["auto"], default="generic",
    help="Which initial_prompt to use (default: generic). Each writes to its "
         "own results CSV so runs don't overwrite each other. 'auto' does a "
         "two-pass run per file: a draft transcription with a neutral "
         "prompt, then an LLM (Qwen) classifies the draft's content and "
         "picks the best-matching specialist prompt for a final re-run — "
         "see AUTO_CANDIDATE_VARIANTS and classify_content_type() below. "
         "Roughly doubles transcription time per file.",
)
parser.add_argument(
    "--condition-on-previous-text", action="store_true", default=False,
    help="Default False — prevents a chunk's internal decode windows from "
         "conditioning on prior text, which stops runaway repetition-loop "
         "hallucinations (seen on Recording_1006 / seg_0003) but also means "
         "initial_prompt only directly seeds the FIRST ~30s window of each "
         "chunk, not the whole chunk. Pass this flag to set it True instead "
         "(closer to Whisper's default) for A/B testing whether a wider "
         "prompt reach is worth the hallucination-loop risk.",
)
parser.add_argument(
    "--chunk-max-s", type=int, default=180,
    help="Max chunk length in seconds before a forced split (default: 180, "
         "i.e. the original 3-minute CHUNK_MAX_MS). Shorter chunks mean "
         "initial_prompt gets re-seeded more often relative to audio "
         "length, which matters more the more condition_on_previous_text "
         "is relied on to carry the prompt's influence forward within a "
         "single chunk.",
)
args = parser.parse_args()

CHUNK_MAX_MS = args.chunk_max_s * 1000
VARIANT = args.prompt_variant

# "auto" has no single fixed prompt — it's chosen per audio file at runtime
# (see classify_content_type below). MEDICAL_INITIAL_PROMPT here is only
# used as the DRAFT-pass prompt for auto mode: a neutral, general prompt,
# not one of the specialist ones, since the whole point of the draft pass
# is to figure out which specialist prompt fits before committing to one.
DRAFT_PROMPT_FOR_AUTO = "generic_v2"
MEDICAL_INITIAL_PROMPT = (
    PROMPT_VARIANTS[DRAFT_PROMPT_FOR_AUTO] if VARIANT == "auto"
    else PROMPT_VARIANTS[VARIANT]
)

# Which specialist prompts the LLM classifier is allowed to pick between in
# auto mode. Restricted to the natural-sentence v2 prompts (the format that
# won the generic-vs-ORL-list comparison) plus generic_v2 as the fallback
# for non-ORL / ambiguous content — deliberately NOT including the original
# comma-list v1 ORL prompts, since those underperformed even on the content
# they were built for.
AUTO_CANDIDATE_VARIANTS = ["generic_v2", "orl_dictee_v2", "orl_assistant_v2", "orl_cro_v2"]

# Build a filename/label suffix reflecting any non-default settings, so
# different experimental combinations never silently overwrite each other.
_suffix_bits = []
if args.condition_on_previous_text:
    _suffix_bits.append("cond1")
if args.chunk_max_s != 180:
    _suffix_bits.append(f"chunk{args.chunk_max_s}s")
_suffix = ("_" + "_".join(_suffix_bits)) if _suffix_bits else ""

RESULTS_PATH = RESULTS_DIR / f"results_whisperx_chunked_prompt_{VARIANT}{_suffix}.csv"
MODEL_LABEL  = f"whisperx-{MODEL_SIZE}-chunked-prompt_{VARIANT}{_suffix}"

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")
log.info(f"condition_on_previous_text={args.condition_on_previous_text} | "
         f"chunk_max_s={args.chunk_max_s}")

# See --condition-on-previous-text help text above for the trade-off this
# controls. Default False (safer against hallucination loops); pass the
# flag to test True instead.
# DIAGNOSTIC: temporarily back to a single-key dict, matching the exact
# structure that demonstrably worked in the orl_cro/orl_assistant/orl_dictee
# runs. Both "condition_on_previous_text": False AND True produced output
# byte-identical to a run with a completely different initial_prompt —
# meaning something about adding that second key appears to silently break
# initial_prompt too, not just fail to apply the conditioning setting on its
# own. Isolating back to one key to confirm initial_prompt alone still
# works before reintroducing the hallucination-loop fix a different way.
if args.condition_on_previous_text:
    log.warning("--condition-on-previous-text was passed but is currently "
                "DISABLED in code (diagnostic mode) — see comment near "
                "load_whisperx_model(). Every model load uses initial_prompt only.")

# Model loading is wrapped in a function (rather than the old top-level
# one-shot load) so 'auto' mode can load a DIFFERENT model per classified
# prompt variant during the run, without reloading the same variant twice
# for different files that get classified the same way. Non-auto runs still
# only ever load one model, same as before — this is a strict superset of
# the old behavior, not a change to it.
_model_cache: dict[str, tuple] = {}

def load_whisperx_model(prompt_text: str, cache_key: str):
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    asr_options = {"initial_prompt": prompt_text}
    log.info(f"Loading WhisperX {MODEL_SIZE} for variant '{cache_key}'...")
    try:
        m = whisperx.load_model(
            MODEL_SIZE, device="cuda", compute_type="float16", language=LANGUAGE,
            asr_options=asr_options,
        )
        dev, ctype, bsize = "cuda", "float16", 8
        log.info("Loaded on cuda/float16 ✓")
    except Exception as e:
        log.warning(f"GPU failed: {e} — falling back to CPU")
        m = whisperx.load_model(
            MODEL_SIZE, device="cpu", compute_type="int8", language=LANGUAGE,
            asr_options=asr_options,
        )
        dev, ctype, bsize = "cpu", "int8", 1
        log.info("Loaded on cpu/int8 ✓")

    _model_cache[cache_key] = (m, dev, ctype, bsize)
    return _model_cache[cache_key]


# ---------------------------------------------------------------------------
# Qwen-based content classifier for --prompt-variant auto.
# Reuses the same Scaleway Generative APIs setup as correct_transcriptions.py
# (SCW_API_KEY env var, OpenAI-compatible client). Given a DRAFT transcript
# (produced with a neutral prompt), asks Qwen to pick which specialist
# prompt best matches the content, restricted to AUTO_CANDIDATE_VARIANTS.
# Falls back to 'generic_v2' on any error, timeout, or invalid response —
# never lets a classification failure crash the run or silently pick an
# unlisted variant.
# ---------------------------------------------------------------------------
_qwen_client = None

def _get_qwen_client():
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = OpenAI(
            api_key=os.environ["SCW_API_KEY"],
            base_url="https://api.scaleway.ai/v1",
        )
    return _qwen_client


def classify_content_type(draft_text: str) -> str:
    fallback = "generic_v2"
    if not draft_text or not draft_text.strip():
        return fallback

    prompt = f"""Tu es un classificateur de contenu médical. Voici un brouillon de
transcription (potentiellement imparfait) d'un audio médical en français.

BROUILLON :
{draft_text[:2000]}

Choisis EXACTEMENT UNE catégorie parmi celles-ci, celle qui correspond le mieux
au contenu :
- "generic_v2" : consultation ou contenu médical général, PAS spécifiquement ORL
  (cardiologie, rhumatologie, neurologie, médecine générale, etc.)
- "orl_dictee_v2" : consultation ORL portant sur des traitements/médicaments/examens
  courants (gouttes, sprays nasaux, audiométrie, tympanométrie...)
- "orl_assistant_v2" : consultation ORL centrée sur les symptômes et le diagnostic
  (acouphènes, vertiges, dysphonie, otalgie...), sans dictée de lettre
- "orl_cro_v2" : lettre ou compte-rendu dicté par le médecin seul, souvent commençant
  par "cher confrère" ou "chers confrères", décrivant une intervention chirurgicale ORL

Réponds UNIQUEMENT avec un objet JSON de cette forme, sans texte avant ni après :
{{"category": "<une des 4 valeurs ci-dessus>"}}
"""
    try:
        response = _get_qwen_client().chat.completions.create(
            model="qwen3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        category = parsed.get("category", "")
        if category in AUTO_CANDIDATE_VARIANTS:
            return category
        log.warning(f"  Classifier returned unrecognized category {category!r} — using {fallback}")
        return fallback
    except Exception as e:
        log.warning(f"  Classification failed ({e}) — using {fallback}")
        return fallback


if VARIANT == "auto":
    # Load the draft-pass model once up front (shared across every file's
    # first pass); specialist models get loaded lazily via load_whisperx_model's
    # cache as classification results come in during the main loop.
    model, DEVICE, COMPUTE_TYPE, BATCH_SIZE = load_whisperx_model(
        MEDICAL_INITIAL_PROMPT, DRAFT_PROMPT_FOR_AUTO
    )
else:
    model, DEVICE, COMPUTE_TYPE, BATCH_SIZE = load_whisperx_model(MEDICAL_INITIAL_PROMPT, VARIANT)

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()


def make_chunks(audio_path: Path) -> list[AudioSegment]:
    audio = AudioSegment.from_file(str(audio_path))
    if len(audio) <= CHUNK_MAX_MS:
        return [audio]
    pieces = split_on_silence(
        audio,
        min_silence_len=SILENCE_MIN_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=SILENCE_KEEP_MS,
    )
    if not pieces:
        log.warning("  No silence detected — falling back to hard 3-min slices")
        return [audio[start : start + CHUNK_MAX_MS]
                for start in range(0, len(audio), CHUNK_MAX_MS)]
    chunks: list[AudioSegment] = []
    current = pieces[0]
    for piece in pieces[1:]:
        if len(current) + len(piece) <= CHUNK_MAX_MS:
            current += piece
        else:
            chunks.append(current)
            current = piece
    chunks.append(current)
    return chunks


def transcribe_chunk(chunk: AudioSegment, whisper_model, batch_size: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        chunk.export(tmp_path, format="wav")
        audio = whisperx.load_audio(tmp_path)
        try:
            result = whisper_model.transcribe(audio, batch_size=batch_size, language=LANGUAGE)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("    OOM on chunk — clearing cache and retrying with batch_size=1")
                torch.cuda.empty_cache()
                result = whisper_model.transcribe(audio, batch_size=1, language=LANGUAGE)
            else:
                raise
        return " ".join(s["text"].strip() for s in result.get("segments", []))
    finally:
        # Clear cache after every chunk (success or failure) to stop VRAM
        # fragmentation from one chunk's OOM cascading into the next chunk's
        # "invalid device ordinal" crash.
        torch.cuda.empty_cache()
        Path(tmp_path).unlink(missing_ok=True)


def transcribe(audio_path: Path, whisper_model, batch_size: int) -> tuple[str, float, int]:
    t0     = time.perf_counter()
    chunks = make_chunks(audio_path)
    parts  = []
    for i, chunk in enumerate(chunks):
        log.info(f"    chunk {i+1}/{len(chunks)} — {len(chunk)/1000:.1f}s")
        parts.append(transcribe_chunk(chunk, whisper_model, batch_size))
    text    = " ".join(p.strip() for p in parts if p.strip())
    latency = time.perf_counter() - t0
    return text.strip(), latency, len(chunks)



records = []
total   = len(segments)

def get_error_words(ref: str, hyp: str, max_items: int = 15) -> str:
    """
    Word-level diff between reference and hypothesis, via jiwer's alignment
    output — NOT just the WER number, but the actual words that were wrong.
    Returns a compact string like:
        "SUB: perindopril->pérennitopril | SUB: bisoprolol->bisoprol | DEL: amoxicilline | INS: catégique"
    Intended for the 'error_words' column: skim this across a batch of
    segments to see which specific terms a model/prompt keeps missing —
    much more actionable for building/tuning an initial_prompt vocabulary
    list than the aggregate WER number alone.
    Capped at max_items entries per segment to keep the CSV cell readable;
    a segment with more errors than that has bigger problems than a
    vocabulary prompt will fix anyway.
    """
    if not ref.strip() or not hyp.strip():
        return ""
    try:
        out = jiwer.process_words(ref, hyp)
    except Exception:
        return ""

    ref_words = ref.split()
    hyp_words = hyp.split()
    items = []
    for alignment in out.alignments[0]:
        if alignment.type == "equal":
            continue
        r = " ".join(ref_words[alignment.ref_start_idx:alignment.ref_end_idx])
        h = " ".join(hyp_words[alignment.hyp_start_idx:alignment.hyp_end_idx])
        if alignment.type == "substitute":
            items.append(f"SUB: {r}->{h}")
        elif alignment.type == "delete":
            items.append(f"DEL: {r}")
        elif alignment.type == "insert":
            items.append(f"INS: {h}")

    if len(items) > max_items:
        items = items[:max_items] + [f"... (+{len(items) - max_items} more)"]
    return " | ".join(items)


for idx, seg in enumerate(segments):
    seg_id     = seg["segment_id"]
    audio_path = AUDIO_BASE_DIR / seg["audio_file"]
    duration_s = seg["duration_s"]
    gt_norm    = seg["ground_truth_normalized"]

    log.info(f"[{idx+1}/{total}] {seg_id}")

    if not audio_path.exists():
        log.warning(f"  Audio not found: {audio_path} — skipping")
        continue

    auto_selected_variant = ""  # only populated when VARIANT == "auto"

    try:
        if VARIANT == "auto":
            # Pass 1: draft transcription with the neutral prompt.
            draft_text, draft_latency, _ = transcribe(audio_path, model, BATCH_SIZE)
            auto_selected_variant = classify_content_type(draft_text)
            log.info(f"  Auto-classified as: {auto_selected_variant}")

            # Pass 2: final transcription with the classified specialist
            # prompt. load_whisperx_model caches by variant name, so a
            # second file classified the same way reuses the already-loaded
            # model instead of reloading large-v3 again.
            final_model, _, _, final_batch = load_whisperx_model(
                PROMPT_VARIANTS[auto_selected_variant], auto_selected_variant
            )
            raw_text, final_latency, n_chunks = transcribe(audio_path, final_model, final_batch)
            latency = draft_latency + final_latency  # total cost, both passes
        else:
            raw_text, latency, n_chunks = transcribe(audio_path, model, BATCH_SIZE)
    except Exception as e:
        log.error(f"  Failed: {e}")
        raw_text, latency, n_chunks = "[ERROR]", -1.0, 0

    log.info(f"  chunks={n_chunks} | latency={latency:.1f}s")

    hyp_norm = norm.normalize(raw_text)
    result   = metrics.compute(
        ref=gt_norm, hyp=hyp_norm,
        latency_s=latency, audio_duration_s=duration_s,
        cost_per_minute=0.0,
    )
    error_words = get_error_words(gt_norm, hyp_norm)

    log.info(f"  WER={result.wer:.3f} | CER={result.cer:.3f} | RTF={result.rtf:.3f}")
    if result.med_critical_errors:
        for err in result.med_critical_errors:
            log.warning(f"  ⚠️  {err}")
    if error_words:
        log.info(f"  Errors: {error_words[:200]}")
    log.info(f"  REF: {gt_norm[:100]}")
    log.info(f"  HYP: {hyp_norm[:100]}")

    records.append({
        "model":               MODEL_LABEL,
        "device":              DEVICE,
        "compute_type":        COMPUTE_TYPE,
        "chunking_strategy":   "silence_3min",
        "n_chunks":            n_chunks,
        "chunk_max_ms":        CHUNK_MAX_MS,
        "silence_thresh_db":   SILENCE_THRESH_DB,
        "silence_min_len_ms":  SILENCE_MIN_LEN_MS,
        "segment_id":          seg_id,
        "consultation_type":   seg.get("consultation_type", "unclassified"),
        "auto_selected_variant": auto_selected_variant,
        "audio_file":          seg["audio_file"],
        "duration_s":          duration_s,
        "hypothesis_raw":      raw_text,
        "hypothesis_norm":     hyp_norm,
        "reference_norm":      gt_norm,
        "error_words":         error_words,
        "dataset_fingerprint": fingerprint,
        **result.to_dict(),
    })

df = pd.DataFrame(records)
df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
log.info(f"Results saved → {RESULTS_PATH}")

print("\n" + "=" * 60)
print(f"RESULTS — {MODEL_LABEL} ({DEVICE}/{COMPUTE_TYPE})")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
print(f"  Mean chunks:     {df['n_chunks'].mean():.1f} per file")
n_crit = df['med_critical_errors'].apply(lambda x: 1 if x and x != '' else 0).sum()
print(f"  Critical errors: {n_crit}")
print(f"  Dataset:         {fingerprint}")
print("=" * 60)

# Consultation (live doctor-patient dialogue) vs dictation ("chers confrères"
# letters / chart notes) — reported separately since they behave very
# differently (dictation WER has consistently run ~2-3x higher than
# consultation WER across every prompt variant tested so far). Segments
# tagged "excluded" (currently just seg_0010 — reference/audio style
# mismatch, not a fair transcription-quality measure) are dropped from
# both. Segments tagged "unclassified" are reported separately so they're
# visible rather than silently folded into either bucket.
print("\nCONSULTATION vs DICTATION:")
for ctype in ["consultation", "dictation"]:
    sub = df[df["consultation_type"] == ctype]
    if len(sub) == 0:
        continue
    print(f"  {ctype:14s} n={len(sub):2d}  mean WER={sub['wer'].mean():.4f}  "
          f"mean CER={sub['cer'].mean():.4f}")

unclassified = df[df["consultation_type"] == "unclassified"]
if len(unclassified) > 0:
    print(f"  UNCLASSIFIED segments (excluded from split above, run "
          f"tag_consultation_type.py to fix): {list(unclassified['segment_id'])}")

excluded = df[df["consultation_type"] == "excluded"]
if len(excluded) > 0:
    print(f"  Excluded (known style mismatch): {list(excluded['segment_id'])}")
print("=" * 60)

# Aggregate word-error frequency across the whole run — this is the part
# meant to directly inform the next initial_prompt revision: which specific
# words does this model/prompt combination keep getting wrong, across
# every segment, not just one. Substitutions matter most for vocabulary
# tuning (the model heard *something* and picked the wrong word); deletions/
# insertions are more often chunk-boundary or silence-detection artifacts.
from collections import Counter
sub_counter = Counter()
for ew in df["error_words"]:
    if not ew:
        continue
    for item in ew.split(" | "):
        if item.startswith("SUB:") and "->" in item:
            ref_word = item[len("SUB: "):].split("->")[0].strip()
            sub_counter[ref_word] += 1

if sub_counter:
    print("\nMost frequently mis-transcribed reference words (substitutions only):")
    for word, count in sub_counter.most_common(15):
        print(f"  {count}x  {word}")
    print("\n(Full per-segment detail — including exact wrong-word replacements "
          f"— is in the 'error_words' column of {RESULTS_PATH})")
