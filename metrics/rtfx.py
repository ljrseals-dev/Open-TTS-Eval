"""Real-time factor helpers (computed in the orchestrator)."""

def rtfx(total_audio_seconds: float, total_inference_seconds: float) -> float:
    """audio_s / infer_s. Higher is faster."""
    if total_inference_seconds <= 0:
        return 0.0
    return total_audio_seconds / total_inference_seconds
