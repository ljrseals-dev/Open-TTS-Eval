#!/usr/bin/env python3
"""
Turn results.json into a leaderboard-style Markdown report.

Usage:
  python scripts/make_report.py results.json > REPORT.md
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def fmt(v, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}{unit}"
    return f"{v}{unit}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_report.py results.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        r = json.load(f)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Loud banner if this is a smoke-test result, not a real evaluation
    if r.get("dry_run"):
        n = r.get("n_samples", "?")
        print(f"> ⚠️  **DRY RUN** — {n} samples were synthesized.")
        print("> This is a smoke test, **not** a valid leaderboard result.\n")

    print(f"# TTS Eval Report — {r.get('model', '?')}\n")
    print(f"_Generated {now}, {r.get('n_samples', '?')} samples_\n")

    print("| Metric | Value | Notes |")
    print("|---|---|---|")

    if "wer" in r:
        print(f"| WER | {fmt(r.get('wer'))} | ASR round-trip (lower better) |")
    if "rtfx" in r:
        print(f"| RTFx | {fmt(r.get('rtfx'))} | audio-s / infer-s (higher better) |")
    if "mean_ttfa_s" in r:
        print(f"| TTFA (mean) | {fmt(r.get('mean_ttfa_s'), ' s')} | time-to-first-audio |")
    if "median_ttfa_s" in r:
        print(f"| TTFA (median) | {fmt(r.get('median_ttfa_s'), ' s')} | |")
    if "utmos" in r:
        print(f"| UTMOS | {fmt(r.get('utmos'))} | predicted naturalness (not human MOS) |")

    print(f"| Audio total | {fmt(r.get('total_audio_s'), ' s')} | |")
    print(f"| Inference total | {fmt(r.get('total_inference_s'), ' s')} | |")

    # ---- Reproducibility: only show provenance for metrics that ran ----
    prov_lines = []
    if "wer" in r:
        prov_lines.append(
            f"- ASR model: `{r.get('asr_model', '—')}` "
            f"(whisper {r.get('whisper_version', '—')})"
        )
    if "utmos" in r:
        prov_lines.append(f"- UTMOS checkpoint: `{r.get('utmos_tag', '—')}`")
    if r.get("per_sample_path"):
        prov_lines.append(f"- Per-sample scores: `{r['per_sample_path']}`")
    if r.get("dry_run"):
        prov_lines.append("- **Dry run:** yes (results are not a valid benchmark)")

    if prov_lines:
        print("\n## Reproducibility\n")
        for line in prov_lines:
            print(line)

    print("\n## Honesty notes\n")
    print("- UTMOS is a **learned predictor**, not human MOS.")
    print("- WER depends on the ASR model used; the model and version are recorded above.")
    print("- RTFx / TTFA are hardware-dependent; record CPU, threads, and batch size.")
    print("- Timings and WER/UTMOS come from the **same single synthesis pass**;")
    print("  per-utterance scores are in `samples.jsonl`.")
    print("- Speaker similarity (SECS) is **not** measured by this harness.")


if __name__ == "__main__":
    main()
