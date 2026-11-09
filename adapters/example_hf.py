import time
import numpy as np
from .base import TTSAdapter, SynthResult


class HFTTSAdapter(TTSAdapter):
    """Wraps any HuggingFace text-to-speech pipeline."""

    def __init__(self, model_id: str, name: str | None = None):
        from transformers import pipeline
        self.pipe = pipeline("text-to-speech", model=model_id)
        self.name = name or model_id

    def synthesize(self, text: str) -> SynthResult:
        t0 = time.perf_counter()
        out = self.pipe(text)
        total = time.perf_counter() - t0
        audio = np.asarray(out["audio"], dtype=np.float32).squeeze()
        sr = out["sampling_rate"]
        # For non-streaming models TTFA ≈ total; real streaming adapters override.
        return SynthResult(
            audio=audio,
            sample_rate=sr,
            ttfa_seconds=total,
            total_seconds=total,
        )
