import time
import numpy as np
from .base import TTSAdapter, SynthResult


class PiperAdapter(TTSAdapter):
    """Wraps a Piper ONNX voice for the eval harness.

    Uses Piper's native streaming generator so TTFA measures actual
    first-chunk latency.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        name: str | None = None,
    ):
        from piper import PiperVoice
        self.voice = PiperVoice.load(model_path, config_path=config_path)
        self.name = name or model_path.split("/")[-1].replace(".onnx", "")

    def synthesize(self, text: str) -> SynthResult:
        t0 = time.perf_counter()
        chunks = []
        first_chunk_time = None

        for chunk in self.voice.synthesize(text):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
            chunks.append(chunk.audio_float_array)

        total = time.perf_counter() - t0
        if not chunks:
            audio = np.zeros(0, dtype=np.float32)
            ttfa = total
        else:
            audio = np.concatenate(chunks).astype(np.float32)
            ttfa = first_chunk_time if first_chunk_time is not None else total

        return SynthResult(
            audio=audio,
            sample_rate=self.voice.config.sample_rate,
            ttfa_seconds=ttfa,
            total_seconds=total,
        )
