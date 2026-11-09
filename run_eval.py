#!/usr/bin/env python3
"""
Open TTS Leaderboard-style evaluation harness.

Usage:
  python run_eval.py [--config config.yaml]

Produces results.json with WER, RTFx, TTFA, UTMOS (as configured).

Design notes
------------
- Synthesis is performed exactly once per sample.
- Cached SynthResult objects are passed to metric functions that need audio.
- TTFA / RTFx are only reported when requested in config["metrics"].
- No fake scores: UTMOS raises if the predictor cannot be loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from adapters.base import SynthResult


def load_samples(path: str | Path) -> list[dict]:
    path = Path(path)
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_adapter(config: dict):
    model_cfg = config["model"]
    adapter_type = model_cfg.get("adapter", "hf").lower()

    if adapter_type == "piper":
        from adapters.piper import PiperAdapter
        return PiperAdapter(
            model_path=model_cfg["model_path"],
            config_path=model_cfg.get("config_path"),
            name=model_cfg.get("name"),
        )
    elif adapter_type in ("hf", "huggingface"):
        from adapters.example_hf import HFTTSAdapter
        return HFTTSAdapter(
            model_id=model_cfg["id"],
            name=model_cfg.get("name"),
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def evaluate(config: dict) -> dict:
    from metrics.wer import compute_wer
    from metrics.utmos import compute_utmos
    from metrics.rtfx import rtfx as compute_rtfx

    samples = load_samples(config["dataset"])
    if not samples:
        raise ValueError(f"No samples found in {config['dataset']}")

    adapter = get_adapter(config)
    metrics = set(config.get("metrics", ["wer", "rtfx", "ttfa"]))
    print(f"Evaluating {adapter.name} on {len(samples)} samples...")
    print(f"Requested metrics: {sorted(metrics)}")

    # ---- single synthesis pass ------------------------------------------------
    results_list: list[SynthResult] = []
    for i, s in enumerate(samples):
        res = adapter.synthesize(s["text"])
        results_list.append(res)
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"  synthesized {i + 1}/{len(samples)}")

    durations = [
        len(r.audio) / r.sample_rate if r.sample_rate > 0 else 0.0
        for r in results_list
    ]
    synth_times = [r.total_seconds for r in results_list]
    ttfas = [r.ttfa_seconds for r in results_list]

    total_audio = float(sum(durations))
    total_synth = float(sum(synth_times))

    results: dict = {
        "model": adapter.name,
        "n_samples": len(samples),
        "total_audio_s": round(total_audio, 3),
        "total_inference_s": round(total_synth, 3),
    }

    if "rtfx" in metrics:
        results["rtfx"] = round(compute_rtfx(total_audio, total_synth), 4)

    if "ttfa" in metrics:
        results["mean_ttfa_s"] = round(float(np.mean(ttfas)), 4)
        results["median_ttfa_s"] = round(float(np.median(ttfas)), 4)

    if "wer" in metrics:
        asr_size = config.get("asr_model", "base")
        print(f"Computing WER (ASR={asr_size}, using cached audio)...")
        results["wer"] = round(
            compute_wer(samples, results_list, asr_model_size=asr_size), 4
        )
        results["asr_model"] = f"whisper-{asr_size}"

    if "utmos" in metrics:
        print("Computing UTMOS (using cached audio)...")
        results["utmos"] = round(compute_utmos(results_list), 4)

    return results


def main():
    parser = argparse.ArgumentParser(description="TTS eval harness")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results = evaluate(cfg)

    out_path = Path(cfg.get("output", "results.json"))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
