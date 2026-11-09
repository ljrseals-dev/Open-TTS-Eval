#!/usr/bin/env python3
"""
Convert LibriSpeech (or similar) split into the harness JSONL format.

Usage:
  python scripts/prep_librispeech.py \
    --root /path/to/LibriSpeech/dev-clean \
    --out datasets/ls_dev.jsonl \
    --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path


def parse_transcript(txt_path: str) -> list[tuple[str, str]]:
    lines = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            uid, text = parts[0], parts[1].lower()
            lines.append((uid, text))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="LibriSpeech split dir (e.g. dev-clean)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    out = []
    root = Path(args.root)
    for txt in sorted(root.rglob("*.trans.txt")):
        folder = txt.parent
        for uid, text in parse_transcript(str(txt)):
            # Prefer flac, fall back to wav
            flac = folder / f"{uid}.flac"
            wav = folder / f"{uid}.wav"
            audio = flac if flac.exists() else (wav if wav.exists() else None)
            if audio is None:
                continue
            out.append(
                {
                    "id": uid,
                    "text": text,
                    "reference": text,
                    "audio": str(audio),
                }
            )
            if len(out) >= args.limit:
                break
        if len(out) >= args.limit:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} samples -> {args.out}")


if __name__ == "__main__":
    main()
