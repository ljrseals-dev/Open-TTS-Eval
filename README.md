# Open TTS Eval Harness

Honest, model-agnostic evaluation harness covering a practical subset of the
metrics used by the
[Hugging Face Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard)
(WER, RTFx, TTFA, UTMOS). Speaker similarity (SECS) is not yet implemented.

| Metric | Meaning | Direction |
|--------|---------|-----------|
| **WER** | Word Error Rate (ASR round-trip) | lower better |
| **RTFx** | Real-time factor = audio_s / inference_s | higher better |
| **TTFA** | Time-to-first-audio (streaming-aware) | lower better |
| **UTMOS** | Predicted naturalness (learned MOS predictor) | higher better |

## Design principles

1. **Model-agnostic** via a thin `TTSAdapter` interface. Drop in Piper, any HF TTS pipeline, or your own fine-tune.
2. **Honest timings** – TTFA uses the actual first chunk when the backend supports streaming.
3. **No fake scores** – UTMOS raises if the predictor is unavailable; WER uses a real ASR (Whisper).
4. **Reproducible** – fixed dataset format, pinned metric models, JSON + Markdown reports.
5. **Single synthesis pass** – audio is generated once and cached; WER/UTMOS reuse the cached `SynthResult` objects (no triple inference).
6. **Metrics are gated** – only metrics listed in `config.yaml` are computed and reported.
7. **Silence-aware WER** – near-silent audio is treated as empty hypothesis (no Whisper hallucination credit).
8. **Per-sample sidecar** – `samples.jsonl` records per-utterance WER / UTMOS for diagnosis.
9. **`--dry-run`** – synthesizes only the first two samples for a fast smoke test.

## Quick start

```bash
cd tts_eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download a Piper voice (example)
# See https://github.com/rhasspy/piper or the OHF-Voice releases
# Place the .onnx (and optional .json) under voices/

# Edit config.yaml to point at your voice and dataset
python run_eval.py --config config.yaml
python scripts/make_report.py results.json > REPORT.md
```

### Sample dataset

A tiny 10-utterance JSONL is included (`datasets/sample_ljspeech.jsonl`).
For real numbers, prepare LibriSpeech or LJSpeech:

```bash
python scripts/prep_librispeech.py \
  --root /path/to/LibriSpeech/dev-clean \
  --out datasets/ls_dev.jsonl \
  --limit 50
```

Then set `dataset: "datasets/ls_dev.jsonl"` in `config.yaml`.

## Adapters

- `adapters/piper.py` – Piper ONNX (streaming TTFA)
- `adapters/example_hf.py` – any Hugging Face `text-to-speech` pipeline
- `adapters/base.py` – implement `synthesize(text) -> SynthResult` for anything else

## Fine-tuning path (Piper)

1. Prepare data in LJSpeech layout with `scripts/prep_piper_train.py`.
2. Use the official Piper / OHF-Voice training stack with a medium-quality checkpoint (`--ckpt_path`).
3. Export to ONNX and point `config.yaml` at the new voice.
4. Re-run the harness and compare WER / UTMOS honestly.

See the Piper training documentation for epoch guidance (roughly 650–850 epochs for a modest fine-tune from a checkpoint). Domain-matched data is what actually moves WER; more of the same LJSpeech-style data rarely does.

## What this is *not*

This harness cannot turn a Karplus-Strong talk-box or magnetic-clavichord physics renderer into a TTS model. Those systems do not produce ASR-transcribable speech and therefore have undefined / catastrophic WER on speech evaluation sets. Use them for what they are good at (deterministic physical modeling, symbolic music pipelines).

## License

MIT for the harness code itself. Respect the licenses of Piper voices, Whisper, UTMOS, and any models you evaluate.
