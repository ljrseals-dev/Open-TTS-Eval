#!/usr/bin/env python3
"""
Open TTS Leaderboard-style evaluation harness.

Usage:
  python run_eval.py [--config config.yaml] [--dry-run]

Produces results.json with WER, RTFx, TTFA, UTMOS (as configured).
Optionally writes samples.jsonl with per-utterance scores.

Design notes
------------
- Synthesis is performed exactly once per sample.
- Cached SynthResult objects are passed to metric functions that need audio.
- TTFA / RTFx are only reported when requested in config["metrics"].
- Silence is gated in WER (no Whisper hallucination credit).
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


def evaluate(config: dict, dry_run: bool = False) -> dict:
    from metrics.wer import compute_wer
    from metrics.utmos import compute_utmos, utmos_tag
    from metrics.rtfx import rtfx as compute_rtfx

    samples = load_samples(config["dataset"])
    if not samples:
        raise ValueError(f"No samples found in {config['dataset']}")

    if dry_run:
        samples = samples[:2]
        print(f"[dry-run] Using first {len(samples)} samples only")

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
        "dry_run": dry_run,
    }

    if "rtfx" in metrics:
        results["rtfx"] = round(compute_rtfx(total_audio, total_synth), 4)

    if "ttfa" in metrics:
        if ttfas:
            results["mean_ttfa_s"] = round(float(np.mean(ttfas)), 4)
            results["median_ttfa_s"] = round(float(np.median(ttfas)), 4)
        else:
            results["mean_ttfa_s"] = None
            results["median_ttfa_s"] = None

    per_sample_records: list[dict] = []

    if "wer" in metrics:
        asr_size = config.get("asr_model", "base")
        print(f"Computing WER (ASR={asr_size}, using cached audio)...")
        wer_val, wer_per = compute_wer(
            samples, results_list, asr_model_size=asr_size, return_per_sample=True
        )
        results["wer"] = round(wer_val, 4)
        results["asr_model"] = f"whisper-{asr_size}"
        try:
            import whisper
            results["whisper_version"] = getattr(whisper, "__version__", "unknown")
        except Exception:
            results["whisper_version"] = "unknown"
        for rec in wer_per:
            per_sample_records.append(rec)

    if "utmos" in metrics:
        print("Computing UTMOS (using cached audio)...")
        utmos_val, utmos_scores = compute_utmos(
            results_list, return_per_sample=True
        )
        results["utmos"] = round(utmos_val, 4) if np.isfinite(utmos_val) else None
        results["utmos_tag"] = utmos_tag()

        # Merge UTMOS into per-sample records if WER already created them
        if per_sample_records and len(utmos_scores) == len(per_sample_records):
            for rec, score in zip(per_sample_records, utmos_scores):
                rec["utmos"] = round(score, 4) if np.isfinite(score) else None
        elif not per_sample_records:
            for i, (s, score) in enumerate(zip(samples, utmos_scores)):
                per_sample_records.append(
                    {
                        "id": s.get("id"),
                        "text": s.get("text"),
                        "utmos": round(score, 4) if np.isfinite(score) else None,
                        "duration_s": round(durations[i], 3),
                    }
                )

    # Always write per-sample sidecar when we have any
    if per_sample_records:
        sidecar = Path(config.get("output", "results.json")).with_name("samples.jsonl")
        with sidecar.open("w", encoding="utf-8") as f:
            for rec in per_sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results["per_sample_path"] = str(sidecar)
        print(f"Wrote per-sample scores → {sidecar}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TTS eval harness")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthesize only the first 2 samples and exit (smoke test)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results = evaluate(cfg, dry_run=args.dry_run)

    out_path = Path(cfg.get("output", "results.json"))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
