"""ASR round-trip Word Error Rate (uses pre-computed SynthResult objects)."""

from __future__ import annotations

import numpy as np
from adapters.base import SynthResult

# -80 dBFS in float32 [-1, 1]; near noise floor. Raise to 1e-3 if adapters
# output quiet-but-broken audio that would otherwise reach Whisper.
_SILENCE_RMS = 1e-4


def _is_silent(audio: np.ndarray) -> bool:
    if audio.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) < _SILENCE_RMS


def compute_wer(
    samples: list[dict],
    results: list[SynthResult],
    asr_model_size: str = "base",
    return_per_sample: bool = False,
) -> float | tuple[float, list[dict]]:
    """
    Transcribe cached audio with Whisper and compute WER vs reference.

    samples  – list of dicts with at least a "reference" (or "text") key
    results  – parallel list of SynthResult from a single synthesis pass
    return_per_sample – if True, also return per-utterance records

    Returns scalar WER (lower is better), or (wer, per_sample_list).
    """
    import whisper
    import jiwer
    import librosa

    if len(samples) != len(results):
        raise ValueError("samples and results must have the same length")

    asr = whisper.load_model(asr_model_size)
    refs, hyps = [], []
    per_sample = []

    for s, res in zip(samples, results):
        ref = s.get("reference", s.get("text", ""))
        if not ref:
            raise ValueError(
                f"Sample missing both 'reference' and 'text' keys: {s.get('id', s)}"
            )

        if _is_silent(res.audio):
            hyp = ""
            silent = True
        else:
            silent = False
            audio = np.ascontiguousarray(res.audio.astype(np.float32))
            if res.sample_rate != 16000:
                audio = librosa.resample(
                    audio, orig_sr=res.sample_rate, target_sr=16000
                )
                audio = np.ascontiguousarray(audio)
            result = asr.transcribe(audio, fp16=False)
            hyp = result["text"].strip()

        refs.append(ref)
        hyps.append(hyp)

        if return_per_sample:
            # Per-utterance WER (single-pair)
            utt_wer = float(jiwer.wer([ref], [hyp])) if ref or hyp else 0.0
            per_sample.append(
                {
                    "id": s.get("id"),
                    "text": s.get("text"),
                    "reference": ref,
                    "hypothesis": hyp,
                    "wer": round(utt_wer, 4),
                    "silent": silent,
                    "duration_s": round(
                        len(res.audio) / res.sample_rate if res.sample_rate else 0.0, 3
                    ),
                }
            )

    overall = float(jiwer.wer(refs, hyps))
    if return_per_sample:
        return overall, per_sample
    return overall
