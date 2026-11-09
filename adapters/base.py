from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SynthResult:
    """Result of a single synthesis call with honest timings."""
    audio: np.ndarray        # float32, mono
    sample_rate: int
    ttfa_seconds: float      # time to first audio sample / chunk
    total_seconds: float     # total wall-clock inference time


class TTSAdapter(ABC):
    """Implement this to plug any TTS model into the harness."""
    name: str = "base"

    @abstractmethod
    def synthesize(self, text: str) -> SynthResult:
        """Return audio + timings. Must be honest about timing."""
        ...
