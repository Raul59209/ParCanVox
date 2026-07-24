"""
notebook_canary_1b_multilingual.py — Benchmark: nvidia/canary-1b-multilingual
===============================================================================
Model:   nvidia/canary-1b-multilingual
Backend: NeMo (FastConformer encoder + attention decoder, multilingual)
Lang:    French (fr)

Why this model:
  - Unlike Parakeet (English-only), Canary multilingual supports French,
    German, Spanish and English — making it directly comparable to
    Whisper-family models on our French medical audio.
  - Same NeMo backend as Parakeet so the setup is identical.
  - 1B parameters, expects 16kHz mono audio.

Key difference from Parakeet:
  - model.transcribe() requires override_cfg={'source_lang': 'fr',
    'target_lang': 'fr', 'task': 'asr', 'pnc': 'yes'} to set the
    language explicitly. Without this it defaults to English.

Run:
    python notebook_canary_1b_multilingual.py
"""

import sys
import json
import time
import logging
import tempfile
from pathlib import Path

# ── Make broken optional deps look genuinely absent ────────────────────────────
# apex is installed in this base image but broken/incomplete (missing
# submodules/attributes a real install would have). That "installed but
# broken" state is NOT something torch, transformers, or NeMo defend
# against — they assume "if it imports, it's fully functional," because
# that's the only state a real install is normally in. What they DO handle
# correctly, via their own `try: import X ... except ImportError` blocks,
# is "X is genuinely not installed." So instead of stubbing apex into a
# half-working, importable state, we make it look genuinely absent, so
# NeMo's own (already correct) fallback logic runs instead of us trying to
# out-guess it call by call.
#
# NOTE: triton is deliberately NOT in this list. Earlier the image shipped
# a broken triton and blocking it was the right call — but the Dockerfile
# now pins a real, working triton==3.3.0 matching torch 2.7.0, and NeMo's
# ngram_lm_batched.py has an unguarded `import triton` (it defines actual
# GPU kernels), so it needs triton to genuinely work, not just look absent.
BLOCKED_PREFIXES = ('apex',)

# Evict any already-cached (broken) copies before anything else imports them.
for _k in list(sys.modules.keys()):
    if _k in BLOCKED_PREFIXES or any(_k.startswith(p + '.') for p in BLOCKED_PREFIXES):
        del sys.modules[_k]


class _BlockedModuleFinder:
    """Force apex/triton (and any submodule under them) to raise
    ModuleNotFoundError on import, regardless of what's actually on disk."""
    def find_module(self, name, path=None):
        if name in BLOCKED_PREFIXES or any(name.startswith(p + '.') for p in BLOCKED_PREFIXES):
            return self
        return None

    def load_module(self, name):
        raise ModuleNotFoundError(
            f"No module named {name!r} (intentionally blocked: the real "
            f"install in this image is broken/incomplete, so it's made to "
            f"look absent instead, letting normal optional-dependency "
            f"fallbacks handle it)"
        )


sys.meta_path.insert(0, _BlockedModuleFinder())

# NOTE: we deliberately do NOT stub other optional NeMo deps (transformer_engine,
# megatron, flash_attn, causal_conv1d, mamba_ssm, grouped_gemm, einops, pynvml,
# etc.). If any of those are genuinely not installed, Python already raises a
# clean ModuleNotFoundError on its own — no custom finder needed, and that's
# the exact condition NeMo's fallback code is designed for. If one of them
# turns out to be a hard (non-optional) requirement for this model and causes
# an unguarded crash, the correct fix is to `pip install` it for real in the
# Dockerfile, not to fake its presence — faking presence for a package whose
# functions are actually called at runtime (like einops) risks silently wrong
# results rather than a loud failure.

# ── Regular imports ───────────────────────────────────────────────────────────
import pandas as pd
import torch
import subprocess

sys.path.insert(0, "/app/src")
from normalizer import MedicalNormalizer
from metrics import BenchmarkMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "nvidia/canary-1b"
SOURCE_LANG     = "fr"   # French input audio
TARGET_LANG     = "fr"   # French output transcript (ASR, not translation)
TARGET_SR       = 16000  # Canary expects 16kHz, same as Parakeet
DATASET_PATH    = Path("dataset/test_set_frozen.json")
AUDIO_BASE_DIR  = Path("audio")
RESULTS_DIR     = Path("results")
RESULTS_PATH    = RESULTS_DIR / "results_canary_1b_multilingual.csv"
RESULTS_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Device: {device}")

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

segments    = dataset["segments"]
fingerprint = dataset["dataset_fingerprint"]
log.info(f"Dataset fingerprint: {fingerprint}")
log.info(f"Segments: {len(segments)} | Total audio: {dataset['total_duration_s']:.1f}s")

# ── Load model ────────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_ID} via NeMo...")
log.info("(First run downloads ~2.5 GB from HuggingFace)")
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
model.eval()
model = model.to(device)
log.info("Model loaded ✓")

norm    = MedicalNormalizer()
metrics = BenchmarkMetrics()


# ── Audio preprocessing ───────────────────────────────────────────────────────
# Canary-1B's decoder was only trained on ≤40s of audio context — NVIDIA's own
# NeMo team has confirmed that beyond ~1-2 minutes its attention "loses
# tracking," degrading accuracy, and in practice it also causes unbounded GPU
# memory growth on long clips (this dataset's segments average ~6 minutes).
# NVIDIA's documented fix for long-form audio is to chunk it into ≤40s pieces,
# transcribe each independently, and stitch the text back together — that's
# what CHUNK_LEN_S / chunk_wav() below do, mirroring their own
# speech_to_text_aed_chunked_infer.py reference script.
CHUNK_LEN_S = 30


def prepare_wav(audio_path: Path) -> str:
    """
    Decode to 16kHz mono WAV via ffmpeg, write to a temp file, return the
    path. NeMo requires a file path (not an array) as input.

    Uses ffmpeg rather than soundfile because soundfile wraps libsndfile,
    which only supports WAV/FLAC/OGG/etc — it cannot decode M4A/AAC at all
    ("Format not recognised"), which is what every .m4a file in this
    dataset is. ffmpeg (already installed in the base image) handles
    essentially any input container/codec, so this also protects against
    future dataset segments arriving as mp3, ogg, etc.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(audio_path),
        "-ac", "1",                 # downmix to mono
        "-ar", str(TARGET_SR),      # resample to 16kHz
        "-f", "wav",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        # Keep the tail of stderr — ffmpeg's actual error is usually the
        # last few lines; the rest is banner/config noise.
        raise RuntimeError(
            f"ffmpeg failed to decode {audio_path}: "
            f"{result.stderr.strip()[-500:]}"
        )
    return tmp.name


def chunk_wav(wav_path: str, chunk_len_s: int = CHUNK_LEN_S) -> tuple[str, list[str]]:
    """
    Split a 16kHz mono WAV into non-overlapping chunk_len_s-second pieces
    using ffmpeg's segment muxer. Returns (temp_dir, [chunk paths]) so the
    caller can clean up the whole directory afterward.
    """
    tmp_dir = tempfile.mkdtemp()
    pattern = str(Path(tmp_dir) / "chunk_%04d.wav")
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", wav_path,
        "-f", "segment", "-segment_time", str(chunk_len_s),
        "-c", "copy",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to chunk {wav_path}: {result.stderr.strip()[-500:]}"
        )
    chunk_paths = sorted(str(p) for p in Path(tmp_dir).glob("chunk_*.wav"))
    return tmp_dir, chunk_paths


# ── Transcribe ────────────────────────────────────────────────────────────────
def transcribe(audio_path: Path) -> tuple[str, float]:
    tmp_path = prepare_wav(audio_path)
    chunk_dir = None
    t0 = time.perf_counter()
    try:
        chunk_dir, chunk_paths = chunk_wav(tmp_path, CHUNK_LEN_S)
        texts = []
        for chunk_path in chunk_paths:
            # NeMo's Canary transcribe() API changed: task/source_lang/target_lang/pnc
            # are now passed as direct keyword arguments rather than nested inside
            # an override_cfg dict (the older pattern this script originally used).
            # pnc is also now a bool, not the older "yes"/"no" string.
            output = model.transcribe(
                [chunk_path],
                task=       "asr",         # "asr" = transcription (not "s2t_translation"/"ast")
                source_lang=SOURCE_LANG,   # input audio language
                target_lang=TARGET_LANG,   # output text language
                pnc=        "yes",        # punctuation and capitalisation (string
                                           # literal — the installed model's prompt
                                           # slot validator rejects a Python bool
                                           # despite NVIDIA's docs sample showing True)
            )
            # NeMo returns Hypothesis objects or plain strings depending on version
            raw = output[0] if output else ""
            text = raw.text if hasattr(raw, "text") else str(raw)
            texts.append(text.strip())

        elapsed = time.perf_counter() - t0
        return " ".join(t for t in texts if t).strip(), elapsed

    except Exception as e:
        elapsed = time.perf_counter() - t0
        raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if chunk_dir is not None:
            import shutil
            shutil.rmtree(chunk_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Run benchmark ─────────────────────────────────────────────────────────────
records = []
total   = len(segments)

for idx, seg in enumerate(segments):
    seg_id     = seg["segment_id"]
    audio_path = AUDIO_BASE_DIR / seg["audio_file"]
    duration_s = seg["duration_s"]
    gt_norm    = seg["ground_truth_normalized"]

    log.info(f"[{idx+1}/{total}] {seg_id} — {seg['audio_file']}")

    if not audio_path.exists():
        log.warning(f"  Audio not found: {audio_path} — skipping")
        continue

    try:
        raw_text, latency = transcribe(audio_path)
    except Exception as e:
        log.error(f"  Transcription failed: {e}")
        raw_text, latency = "[ERROR]", -1.0

    hyp_norm = norm.normalize(raw_text)
    result   = metrics.compute(
        ref=gt_norm,
        hyp=hyp_norm,
        latency_s=latency,
        audio_duration_s=duration_s,
        cost_per_minute=0.0,
    )

    log.info(f"  WER={result.wer:.3f} | CER={result.cer:.3f} | "
             f"RTF={result.rtf:.3f} | latency={result.latency_s:.1f}s")
    if result.med_critical_errors:
        for err in result.med_critical_errors:
            log.warning(f"  ⚠️  {err}")
    log.info(f"  REF: {gt_norm[:100]}")
    log.info(f"  HYP: {hyp_norm[:100]}")

    records.append({
        "model":               MODEL_ID,
        "source_lang":         SOURCE_LANG,
        "device":              device,
        "chunking_strategy":   f"chunked_{CHUNK_LEN_S}s",
        "segment_id":          seg_id,
        "audio_file":          seg["audio_file"],
        "duration_s":          duration_s,
        "hypothesis_raw":      raw_text,
        "hypothesis_norm":     hyp_norm,
        "reference_norm":      gt_norm,
        "dataset_fingerprint": fingerprint,
        **result.to_dict(),
    })

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df["run_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
log.info(f"\nResults saved → {RESULTS_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS — {MODEL_ID} / French / whole audio ({device})")
print("=" * 60)
print(f"  Segments:        {len(df)}")
print(f"  Mean WER:        {df['wer'].mean():.3f}")
print(f"  Mean CER:        {df['cer'].mean():.3f}")
print(f"  Med entity acc:  {df['med_entity_acc'].mean():.3f}")
print(f"  Mean RTF:        {df['rtf'].mean():.3f}  {'✓ faster than realtime' if df['rtf'].mean() < 1 else '✗ slower than realtime'}")
print(f"  Mean latency:    {df['latency_s'].mean():.1f}s")
n_err  = (df['hypothesis_raw'] == '[ERROR]').sum()
n_crit = df['med_critical_errors'].apply(lambda x: 1 if x and x != '' else 0).sum()
print(f"  Errors:          {n_err}/{len(df)}")
print(f"  Critical errors: {n_crit}")
print(f"  Dataset:         {fingerprint}")
print("=" * 60)