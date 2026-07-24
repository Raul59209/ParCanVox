import torch, torchaudio
torch._dynamo.config.disable = True
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

processor = KyutaiSpeechToTextProcessor.from_pretrained('kyutai/stt-1b-en_fr-trfs')

import inspect
print("=== Processor class:", type(processor))
print("=== feature_extractor:", getattr(processor, 'feature_extractor', 'NO FEATURE EXTRACTOR ATTR'))
try:
    sig = inspect.signature(processor.__call__)
    print("=== __call__ signature:", sig)
except Exception as e:
    print("Could not inspect signature:", e)

model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
    'kyutai/stt-1b-en_fr-trfs', device_map=device, torch_dtype='auto'
)

waveform, sr = torchaudio.load('/app/audio/Consultation 1.m4a')
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != 24000:
    waveform = torchaudio.functional.resample(waveform, sr, 24000)
audio_array = waveform.squeeze(0).numpy()

print(f"audio_array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
print(f"audio_array min/max: {audio_array.min()}/{audio_array.max()}")
print(f"audio_array has NaN: {torch.isnan(torch.from_numpy(audio_array)).any().item()}")
print(f"audio_array all zero: {(audio_array == 0).all()}")

inputs = processor(audio_array, return_tensors='pt')
print('inputs type:', type(inputs))
print('inputs raw dict:', dict(inputs) if hasattr(inputs, 'keys') else inputs)
print('inputs keys:', inputs.keys() if hasattr(inputs, 'keys') else 'NO KEYS METHOD')

for k in (inputs.keys() if hasattr(inputs, 'keys') else []):
    v = inputs[k]
    shape = getattr(v, 'shape', 'NO SHAPE ATTRIBUTE')
    print(f'  {k}: type={type(v)} shape={shape} value_preview={str(v)[:100] if v is None else ""}')

inputs = inputs.to(device)
print("Calling model.generate()...")
output_tokens = model.generate(**inputs)
print('generate succeeded, output shape:', output_tokens.shape)

text = processor.batch_decode(output_tokens, skip_special_tokens=True)
print('DECODED TEXT:', text)