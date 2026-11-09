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
    print(f"# TTS Eval Report — {r.get('model', '?')}\n")
    print(f"_Generated {now}, {r.get('n_samples', '?')} samples_\n")
    print("| Metric | Value | Notes |")
    print("|---|---|---|")
    print(f"| WER | {fmt(r.get('wer'))} | ASR round-trip (lower better) |")
    print(f"| RTFx | {fmt(r.get('rtfx'))} | audio-s / infer-s (higher better) |")
    print(f"| TTFA (mean) | {fmt(r.get('mean_ttfa_s'), ' s')} | time-to-first-audio |")
    if "median_ttfa_s" in r:
        print(f"| TTFA (median) | {fmt(r.get('median_ttfa_s'), ' s')} | |")
    print(f"| UTMOS | {fmt(r.get('utmos'))} | predicted naturalness (not human MOS) |")
    print(f"| Audio total | {fmt(r.get('total_audio_s'), ' s')} | |")
    print(f"| Inference total | {fmt(r.get('total_inference_s'), ' s')} | |")
    print("\n## Honesty notes\n")
    print("- UTMOS is a **learned predictor**, not human MOS.")
    print("- WER depends on the ASR model used; state which (e.g. Whisper-base).")
    print("- RTFx / TTFA are hardware-dependent; record CPU, threads, and batch size.")
    print("- This harness re-synthesizes for WER/UTMOS; timings come from a separate pass.")


if __name__ == "__main__":
    main()
