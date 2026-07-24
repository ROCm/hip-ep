#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Whisper speech-to-text demo on the AMD GPU (MorphiZen EP).

A self-contained, presentation-friendly driver: it downloads a Whisper variant
from AMD's Hugging Face hub, transcribes a 16 kHz wav on the GPU, and prints the
transcription plus a per-phase latency / real-time-factor table. Built on the
same greedy-decode harness the tests use (test/python/whisper/whisper_infer.py),
so what you see here is exactly what the EP runs.

    # one command — downloads the model + the bundled JFK clip on first run:
    python scripts/whisper_demo.py

    python scripts/whisper_demo.py --variant tiny             # fastest, smallest
    python scripts/whisper_demo.py --audio meeting_16k.wav    # your own clip
    python scripts/whisper_demo.py --compare-cpu              # GPU vs ORT CPU
    python scripts/whisper_demo.py --list                     # show variants

Audio must be **16 kHz mono** and <= 30 s. Convert anything else first:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 clip_16k.wav

Prerequisites (one-time — see docs/whisper_demo.md / docs/whisper_quick_start.md):
  * `python build.py` has produced the EP, and THEROCK_DIST + install/{therock,dist}/bin
    are on PATH (so the EP can link the compiled model DLL).
  * `hf auth login` (the AMD Whisper repos are gated).
"""

import argparse
import pathlib
import sys
import time

import numpy as np

# scripts/ sits at the repo root; the harness lives in test/python/whisper/ and
# the shared helpers in test/python/. Put both on sys.path so this runs from any
# working directory.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))
sys.path.insert(0, str(REPO_ROOT / "test" / "python" / "whisper"))

import whisper_infer  # noqa: E402
from conftest import (  # noqa: E402
    WHISPER_VARIANTS,
    setup_jfk_sample,
    setup_whisper_variant,
)

JFK_SAMPLE = REPO_ROOT / "test" / "python" / "data" / "whisper" / "jfk.wav"

# Default to large-v3-turbo for the live demo: near large-v3 accuracy, much
# faster decode (4 decoder layers), and a modest download — the best balance for
# a meeting. Switch to --variant tiny for the quickest possible smoke run.
DEFAULT_VARIANT = "large-v3-turbo"


def _banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def _step(text: str) -> None:
    print(f"  -> {text}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe a 16 kHz wav with Whisper on the AMD GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=sorted(WHISPER_VARIANTS),
        help="Whisper variant (auto-downloads from amd/whisper-<variant>-onnx-fp16)",
    )
    ap.add_argument(
        "--audio",
        default=str(JFK_SAMPLE),
        help="path to a 16 kHz mono wav (<= 30 s); defaults to the bundled JFK clip",
    )
    ap.add_argument(
        "--compare-cpu",
        action="store_true",
        help="also run the ORT CPU EP and print both transcriptions",
    )
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="use the fp32 model (large-v3 only); default is fp16",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="max decoded tokens (cap 448; raise for long clips)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list the supported variants and exit",
    )
    args = ap.parse_args()

    if args.list:
        print("Supported Whisper variants (source: amd/whisper-<name>-onnx-fp16):")
        for name in sorted(WHISPER_VARIANTS):
            tag = "  (default)" if name == DEFAULT_VARIANT else ""
            print(f"  - {name}{tag}")
        return 0

    prec = "fp32" if args.fp32 else "fp16"
    dtype = np.float16 if prec == "fp16" else np.float32
    if prec == "fp32" and args.variant != "large-v3":
        print(
            f"[demo] ERROR: fp32 is only published for large-v3 "
            f"(requested {args.variant}). Drop --fp32 to use fp16."
        )
        return 1

    _banner(f"Whisper demo - {args.variant} ({prec}) on the AMD GPU")

    # --- Audio -------------------------------------------------------------
    audio_path = pathlib.Path(args.audio)
    if not audio_path.exists() and audio_path.resolve() == JFK_SAMPLE.resolve():
        _step("fetching the bundled demo clip (jfk.wav) ...")
        if not setup_jfk_sample(JFK_SAMPLE.parent):
            print("[demo] ERROR: could not download jfk.wav (no network?)")
            return 1
    if not audio_path.exists():
        print(f"[demo] ERROR: audio file not found: {audio_path}")
        return 1
    audio_s = whisper_infer.audio_duration_s(audio_path)
    _step(f"audio: {audio_path}  ({audio_s:.1f} s, expects 16 kHz mono)")

    # --- Model -------------------------------------------------------------
    _step(
        f"preparing model (downloads from amd/whisper-{args.variant}-onnx-{prec} "
        "on first run) ..."
    )
    try:
        model_dir, variant = setup_whisper_variant(args.variant, precision=prec)
    except (FileNotFoundError, KeyError) as e:
        print(f"[demo] ERROR preparing model: {e}")
        return 1
    _step(f"model ready: {model_dir}")

    audio_fp = whisper_infer.load_audio_features(audio_path, variant=variant)

    # --- CPU reference (optional) -----------------------------------------
    if args.compare_cpu:
        _step("running ORT CPU reference (fp32 dynamic graph) ...")
        cpu_factory = whisper_infer.make_cpu_session_factory(model_dir)
        cpu_tokens = whisper_infer.greedy_decode_cpu(
            cpu_factory,
            audio_fp,
            max_length=args.max_length,
            dtype=dtype,
            variant=variant,
        )
        cpu_text = whisper_infer.decode_text(
            cpu_tokens, tokenizer_id=variant.hf_model_id, eot=variant.eot
        )

    # --- GPU run (warmup + timed) -----------------------------------------
    _step("compiling + warming up the GPU session (one-time) ...")
    base_factory = whisper_infer.make_morphizen_session_factory(REPO_ROOT, model_dir)
    _cache = {}

    def gpu_factory(name):
        if name not in _cache:
            _cache[name] = base_factory(name)
        return _cache[name]

    # Warmup primes the very sessions the timed run reuses (model-DLL compile +
    # kernel autotune are excluded from the reported metrics).
    whisper_infer.greedy_decode_morphizen(
        gpu_factory, audio_fp, max_length=args.max_length, dtype=dtype, variant=variant
    )
    _step("transcribing on the GPU (timed) ...")
    timings = {}
    t0 = time.perf_counter()
    gpu_tokens = whisper_infer.greedy_decode_morphizen(
        gpu_factory,
        audio_fp,
        max_length=args.max_length,
        timings=timings,
        dtype=dtype,
        variant=variant,
    )
    wall_s = time.perf_counter() - t0
    gpu_text = whisper_infer.decode_text(
        gpu_tokens, tokenizer_id=variant.hf_model_id, eot=variant.eot
    )

    # --- Results -----------------------------------------------------------
    _banner("Transcription")
    if args.compare_cpu:
        print(f"  CPU : {cpu_text!r}")
        print(f"  GPU : {gpu_text!r}")
        print(f"\n  match: {cpu_text.strip() == gpu_text.strip()}")
    else:
        print(f"  {gpu_text!r}")

    _print_metrics(timings, audio_s, wall_s, label=f"AMD GPU ({prec})")
    return 0


def _print_metrics(t, audio_s, wall_s, label):
    """Per-phase compute latency + real-time factor for the GPU run.

    enc/prefill/decode are model-compute only (exclude feature extraction +
    detokenize); wall is the full timed greedy loop. RTF < 1.0 == faster than
    real time; tok/s is the steady-state per-token decode rate.
    """
    enc = t.get("enc_ms", 0.0)
    prefill = t.get("prefill_ms", 0.0)
    decode = t.get("decode_ms", 0.0)
    n = t.get("n_decode_steps", 0)
    total = enc + prefill + decode
    tps = n / (decode / 1e3) if decode > 0 else 0.0
    rtf = wall_s / audio_s if audio_s > 0 else 0.0
    _banner(f"Performance - {label}")
    print(f"  audio length      : {audio_s:9.2f} s")
    print(f"  encoder           : {enc:9.1f} ms")
    print(f"  prefill (TTFT)    : {enc + prefill:9.1f} ms")
    print(f"  decode loop       : {decode:9.1f} ms   ({n} tokens)")
    print(f"  {'-' * 38}")
    print(f"  total compute     : {total:9.1f} ms")
    print(f"  decode throughput : {tps:9.1f} tok/s")
    if rtf > 0:
        print(
            f"  real-time factor  : {rtf:9.3f}   ({1 / rtf:.1f}x faster than real time)"
            if rtf < 1
            else f"  real-time factor  : {rtf:9.3f}"
        )
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
