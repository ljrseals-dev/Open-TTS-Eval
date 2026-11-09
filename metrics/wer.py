"""ASR round-trip Word Error Rate (uses pre-computed SynthResult objects)."""

from __future__ import annotations

import numpy as np
from adapters.base import SynthResult


def compute_wer(
    samples: list[dict],
    results: list[SynthResult],
    asr_model_size: str = "base",
) -> float:
    """
    Transcribe cached audio with Whisper and compute WER vs reference.

    samples  – list of dicts with at least a "reference" key
    results  – parallel list of SynthResult from a single synthesis pass
    Returns scalar WER (lower is better).
    """
    import whisper
    import jiwer
    import librosa

    if len(samples) != len(results):
        raise ValueError("samples and results must have the same length")

    asr = whisper.load_model(asr_model_size)
    refs, hyps = [], []

    for s, res in zip(samples, results):
        if res.audio.size == 0:
            hyp = ""
        else:
            audio = res.audio.astype(np.float32)
            if res.sample_rate != 16000:
                # librosa is fine; for pure speed on 22.05 kHz → 16 kHz
                # scipy.signal.resample_poly could be substituted later
                audio = librosa.resample(
                    audio, orig_sr=res.sample_rate, target_sr=16000
                )
            result = asr.transcribe(audio, fp16=False)
            hyp = result["text"].strip()
        refs.append(s["reference"])
        hyps.append(hyp)

    return float(jiwer.wer(refs, hyps))
