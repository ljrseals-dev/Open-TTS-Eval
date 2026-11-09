#!/usr/bin/env python3
"""
Convert a JSONL ({audio, text}) into Piper's LJSpeech-style layout.

Usage:
  python scripts/prep_piper_train.py \
    --in data/train.jsonl \
    --out data/piper_train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "metadata.csv"

    count = 0
    with open(args.inp, encoding="utf-8") as f, meta.open("w", encoding="utf-8") as m:
        for i, line in enumerate(f):
            r = json.loads(line)
            src = Path(r["audio"])
            if not src.exists():
                print(f"Warning: missing audio {src}, skipping")
                continue
            ext = src.suffix or ".wav"
            dst_name = f"utt{i:06d}{ext}"
            dst = wavs / dst_name
            shutil.copy(src, dst)
            # Piper expects: filename|text  (no extension in the id)
            m.write(f"utt{i:06d}|{r['text']}\n")
            count += 1

    print(f"Wrote {count} utterances -> {out_dir}/wavs + metadata.csv")


if __name__ == "__main__":
    main()
