"""ASR round-trip Word Error Rate."""

import numpy as np


def compute_wer(adapter, samples, asr_model_size: str = "base"):
    """
    Synthesize each sample, transcribe with Whisper, compute WER vs reference.

    samples: list of dicts with at least "text" and "reference" keys.
    Returns scalar WER (lower is better).
    """
    import whisper
    import jiwer
    import librosa

    asr = whisper.load_model(asr_model_size)
    refs, hyps = [], []

    for s in samples:
        res = adapter.synthesize(s["text"])
        if res.audio.size == 0:
            hyp = ""
        else:
            # Whisper expects float32 @ 16 kHz
            if res.sample_rate != 16000:
                audio16 = librosa.resample(
                    res.audio.astype(np.float32),
                    orig_sr=res.sample_rate,
                    target_sr=16000,
                )
            else:
                audio16 = res.audio.astype(np.float32)
            result = asr.transcribe(audio16, fp16=False)
            hyp = result["text"].strip()
        refs.append(s["reference"])
        hyps.append(hyp)

    return jiwer.wer(refs, hyps)
