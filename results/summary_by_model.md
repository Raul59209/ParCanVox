# STT Benchmark Results — French Medical Audio

Dataset fingerprint: `a3e83cf2aa8a84b85142`

## Summary: model × metric

| Model | WER ↓ | CER ↓ | Med entity acc ↑ | Latency (s) ↓ | RTF ↓ | Cost/hr audio ($) | Critical errors ↓ | Segments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster-whisper-deepdml/faster-whisper-large-v3-turbo-ct2-int8-chunked | 0.2960 | 0.1957 | 0.8926 | 13.4224 | 0.0378 | 0.0000 | 2 | 17 |
| faster-whisper-deepdml/faster-whisper-large-v3-turbo-ct2-int8-whole | 0.2941 | 0.1908 | 0.8779 | 5.6136 | 0.0183 | 0.0000 | 2 | 17 |
| faster-whisper-large-v3 | 0.2782 | 0.1919 | 0.9147 | 15.7395 | 0.0499 | 0.0000 | 2 | 17 |
| faster-whisper-large-v3-chunked | 0.2734 | 0.1845 | 0.8706 | 20.9029 | 0.0622 | 0.0000 | 2 | 17 |
| faster-whisper-large-v3-int8-chunked | 0.2830 | 0.1904 | 0.8485 | 21.3441 | 0.0634 | 0.0000 | 3 | 17 |
| mistral/voxtral-small-24b-2507 | 0.3086 | 0.2288 | 0.8292 | 34.1831 | 0.0893 | 0.0000 | 6 | 34 |
| whisperx-large-v3 | 0.2471 | 0.1742 | 0.8608 | 4.4742 | 0.0150 | 0.0000 | 3 | 17 |
| whisperx-large-v3-chunked | 0.2473 | 0.1686 | 0.8608 | 11.9672 | 0.0334 | 0.0000 | 4 | 17 |

## Notes
- WER and CER: lower is better. Computed on normalized text (same normalizer applied to both reference and hypothesis).
- Med entity acc: proportion of medical entities (drugs, dosages, routes, frequencies) correctly transcribed. Higher is better.
- RTF (Real-Time Factor): latency / audio duration. RTF < 1.0 = faster than real-time.
- Critical errors: dosage mismatches (e.g. 500 mg vs 5000 mg), missing or hallucinated dosages.
- Cost/hr audio: extrapolated API cost per hour of audio. Local models = $0.
