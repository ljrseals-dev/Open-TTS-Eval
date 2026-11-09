"""
Predicted naturalness (UTMOS).

Uses the UTMOS22 predictor via torch.hub (SpeechMOS).
Raises a clear error if the predictor is unavailable — never invents scores.
"""

import numpy as np

_UTMOS = None


def _load():
    global _UTMOS
    if _UTMOS is None:
        try:
            import torch
            _UTMOS = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                "UTMOS predictor unavailable. Install torch and allow "
                f"torch.hub, or vendor the checkpoint. Underlying error: {e}"
            )
    return _UTMOS


def compute_utmos(adapter, samples, target_sr: int = 16000):
    """Return mean UTMOS over synthesized samples."""
    import librosa
    import torch

    model = _load()
    scores = []

    for s in samples:
        res = adapter.synthesize(s["text"])
        if res.audio.size == 0:
            continue
        audio = res.audio.astype(np.float32)
        if res.sample_rate != target_sr:
            audio = librosa.resample(
                audio, orig_sr=res.sample_rate, target_sr=target_sr
            )
        t = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            mos = model(t, target_sr).item()
        scores.append(mos)

    return float(np.mean(scores)) if scores else float("nan")
