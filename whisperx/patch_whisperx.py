import site, os

base = site.getsitepackages()[0]

# ─────────────────────────────────────────────
# Patch 1: Fix TranscriptionOptions mismatch
# ─────────────────────────────────────────────
asr_path = os.path.join(base, "whisperx/asr.py")
lines = open(asr_path).readlines()

target = "TranscriptionOptions(**default_asr_options)"
new_lines = []

for line in lines:
    if target in line:
        indent = " " * (len(line) - len(line.lstrip()))

        new_lines.append(indent + 'default_asr_options.setdefault("max_new_tokens", None)\n')
        new_lines.append(indent + 'default_asr_options.setdefault("clip_timestamps", None)\n')
        new_lines.append(indent + 'default_asr_options.setdefault("hallucination_silence_threshold", None)\n')
        new_lines.append(indent + 'default_asr_options.setdefault("hotwords", None)\n')

        new_lines.append(line)
    else:
        new_lines.append(line)

open(asr_path, "w").writelines(new_lines)
print("✅ Patched ASR options")


# ─────────────────────────────────────────────
# Patch 2: DISABLE VAD COMPLETELY
# ─────────────────────────────────────────────
vad_path = os.path.join(base, "whisperx/vad.py")

vad_stub = """
def load_vad_model(*args, **kwargs):
    return None

def merge_chunks(*args, **kwargs):
    return []
"""

open(vad_path, "w").write(vad_stub)
print("✅ Disabled VAD completely")