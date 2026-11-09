"""
Predicted naturalness (UTMOS).

Uses the UTMOS22 predictor via torch.hub (SpeechMOS).
Raises a clear error if the predictor is unavailable — never invents scores.
Accepts a list of pre-computed SynthResult objects (no re-synthesis).
"""

from __future__ import annotations

import numpy as np
from adapters.base import SynthResult

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


def compute_utmos(
    results: list[SynthResult],
    target_sr: int = 16000,
) -> float:
    """Return mean UTMOS over pre-synthesized samples."""
    import librosa
    import torch

    model = _load()
    scores = []

    for res in results:
        if res.audio.size == 0:
            continue
        audio = np.ascontiguousarray(res.audio.astype(np.float32))
        if res.sample_rate != target_sr:
            audio = librosa.resample(
                audio, orig_sr=res.sample_rate, target_sr=target_sr
            )
            audio = np.ascontiguousarray(audio)
        t = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            mos = model(t, target_sr).item()
        if np.isfinite(mos):
            scores.append(mos)

    if not scores:
        return float("nan")
    return float(np.mean(scores))
