#!/usr/bin/env python3
"""
Open TTS Leaderboard-style evaluation harness.

Usage:
  python run_eval.py [--config config.yaml]

Produces results.json with WER, RTFx, TTFA, UTMOS (as configured).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml


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

    samples = load_samples(config["dataset"])
    if not samples:
        raise ValueError(f"No samples found in {config['dataset']}")

    adapter = get_adapter(config)
    print(f"Evaluating {adapter.name} on {len(samples)} samples...")

    durations = []
    synth_times = []
    ttfas = []

    for i, s in enumerate(samples):
        res = adapter.synthesize(s["text"])
        dur = len(res.audio) / res.sample_rate if res.sample_rate > 0 else 0.0
        durations.append(dur)
        synth_times.append(res.total_seconds)
        ttfas.append(res.ttfa_seconds)
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"  synthesized {i + 1}/{len(samples)}")

    total_audio = float(sum(durations))
    total_synth = float(sum(synth_times))
    metrics = set(config.get("metrics", ["wer", "rtfx", "ttfa"]))

    results = {
        "model": adapter.name,
        "n_samples": len(samples),
        "total_audio_s": round(total_audio, 3),
        "total_inference_s": round(total_synth, 3),
        "rtfx": round(total_audio / total_synth, 4) if total_synth > 0 else 0.0,
        "mean_ttfa_s": round(float(np.mean(ttfas)), 4),
        "median_ttfa_s": round(float(np.median(ttfas)), 4),
    }

    if "wer" in metrics:
        print("Computing WER (this re-synthesizes + runs ASR)...")
        results["wer"] = round(compute_wer(adapter, samples), 4)

    if "utmos" in metrics:
        print("Computing UTMOS...")
        results["utmos"] = round(compute_utmos(adapter, samples), 4)

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
