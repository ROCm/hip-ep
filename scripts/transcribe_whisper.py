#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Transcribe a 16 kHz wav with Whisper-large-v3 on the MorphiZen EP (GPU).

A standalone driver around the shared greedy-decode harness in
``test/python/whisper/whisper_infer.py`` — no pytest, no ``python -c``. Runs identically
in PowerShell and Git Bash:

    python scripts/transcribe_whisper.py path/to/audio_16k.wav             # GPU fp16 (default)
    python scripts/transcribe_whisper.py path/to/audio_16k.wav --compare   # GPU+CPU
    python scripts/transcribe_whisper.py path/to/audio_16k.wav --cpu       # CPU only
    python scripts/transcribe_whisper.py path/to/audio_16k.wav --fp32      # GPU fp32

Audio must be **16 kHz mono**. Resample first if needed, e.g.:
    ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav
The feature extractor pads/truncates to Whisper's 30 s window; clips longer than
30 s need chunking (not handled here).

Prerequisites (same as the tests — see docs/whisper_quick_start.md):
  * ``python build.py`` has produced install/dist/bin/onnxruntime_morphizen_ep.dll
  * THEROCK_DIST + install/{therock,dist}/bin are on PATH (so the EP can link the
    model DLL). Without them the EP raises rather than silently falling back.
  * The Whisper model must already be BUILT LOCALLY first:
        python build.py --build-whisper-models
    This script only PREPARES (surgery + fix_shapes) and transcribes — it does
    NOT download or build the model. If the raw bundle is absent it exits with a
    build hint rather than a raw traceback.
"""

import argparse
import pathlib
import sys

import numpy as np

# Resolve the repo root from this script's location (scripts/ is at repo root),
# so it works regardless of the current working directory. The shared whisper
# harness lives in test/python/whisper/; conftest helpers in test/python/. Put
# both on the path.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))
sys.path.insert(0, str(REPO_ROOT / "test" / "python" / "whisper"))

import whisper_infer  # noqa: E402
from conftest import (  # noqa: E402
    setup_whisper_fp16_model_dir,
    setup_whisper_model_dir,
)

MODEL_DIR = REPO_ROOT / "models" / "whisper-large-v3-onnx"
MODEL_DIR_FP16 = REPO_ROOT / "models" / "whisper-large-v3-onnx-fp16"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe a 16 kHz wav with Whisper-large-v3 on the "
        "MorphiZen EP (GPU)."
    )
    ap.add_argument("audio", help="path to a 16 kHz mono wav (<= 30 s)")
    # Backend selection. Default (no flag) = GPU only. The two flags are mutually
    # exclusive: --cpu runs ONLY the CPU EP; --compare runs BOTH and prints a
    # CPU: and a GPU: line for an apples-to-apples check.
    backend = ap.add_mutually_exclusive_group()
    backend.add_argument(
        "--cpu",
        action="store_true",
        help="run ONLY the ORT CPU fp32 EP (no GPU)",
    )
    backend.add_argument(
        "--compare",
        action="store_true",
        help="run BOTH GPU and CPU and print both for an apples-to-apples check",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="max decoded tokens (default 200; raise for long clips, cap 448)",
    )
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="use the fp32 model instead of the default fp16. The DEFAULT is the "
        "fp16 model (built locally via the OGA DML builder; body fp16, lm_head "
        "fp32 so greedy is argmax-lossless) — it is faster than fp32 on GPU. Both "
        "precisions must be built first via 'python build.py "
        "--build-whisper-models' (see docs/whisper_quick_start.md).",
    )
    ap.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print per-phase latency (encoder / prefill / decode) + RTF + tok/s "
        "(default on; pass --no-metrics for transcription only)",
    )
    ap.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run one discarded GPU pass before the timed one so the metrics are "
        "steady-state, excluding the one-time model-DLL load + kernel autotune "
        "(default on; pass --no-warmup to skip it and see the cold-start cost)",
    )
    args = ap.parse_args()

    audio_path = pathlib.Path(args.audio)
    if not audio_path.exists():
        print(f"[transcribe] ERROR: audio file not found: {audio_path}")
        return 1

    # Precision selection: fp16 (DEFAULT) or fp32 (--fp32). Both are built locally
    # via the OGA DML builder (python build.py --build-whisper-models). dtype
    # drives the KV/audio cast; lm_head stays fp32 in both so greedy argmax is
    # lossless. fp16 is the default because it is faster on GPU and bit-faithful.
    use_fp16 = not args.fp32
    prec = "fp16" if use_fp16 else "fp32"
    dtype = np.float16 if use_fp16 else np.float32
    model_dir = MODEL_DIR_FP16 if use_fp16 else MODEL_DIR

    # The model must already be built (python build.py --build-whisper-models);
    # the setup helpers are consume-only and raise FileNotFoundError if absent.
    print(f"[transcribe] preparing {prec} model at {model_dir}")
    try:
        if use_fp16:
            setup_whisper_fp16_model_dir(model_dir)
        else:
            setup_whisper_model_dir(model_dir)
    except FileNotFoundError as e:
        print(f"[transcribe] ERROR: {e}")
        print("[transcribe] Build the model first:")
        print("    python build.py --build-whisper-models")
        return 1

    print(f"[transcribe] loading audio features from {audio_path}")
    audio_fp = whisper_infer.load_audio_features(audio_path)

    run_cpu = args.cpu or args.compare
    run_gpu = not args.cpu  # default + --compare run GPU; --cpu skips it

    audio_s = whisper_infer.audio_duration_s(audio_path)

    if run_cpu:
        cpu_factory = whisper_infer.make_cpu_session_factory(model_dir)
        cpu_timings = {} if args.metrics else None
        cpu_tokens = whisper_infer.greedy_decode_cpu(
            cpu_factory,
            audio_fp,
            max_length=args.max_length,
            timings=cpu_timings,
            dtype=dtype,
        )
        print(f"CPU: {whisper_infer.decode_text(cpu_tokens)!r}")
        if args.metrics:
            _print_metrics(cpu_timings, audio_s, label=f"CPU EP ({prec})")

    if run_gpu:
        # Cache GPU sessions so warmup + the timed run reuse the SAME sessions.
        # Each MorphiZen session compiles its ONNX -> a GPU model.dll at creation
        # (the [ConvertOnnxToHipPass] / [SHARED_CONSTANTS] chatter); without
        # caching, warmup and the timed run would build separate sessions and
        # compile every graph twice. Caching also makes warmup meaningful — it
        # primes the very sessions the timed run uses.
        base_factory = whisper_infer.make_morphizen_session_factory(
            REPO_ROOT, model_dir
        )
        _session_cache = {}

        def gpu_factory(name):
            if name not in _session_cache:
                _session_cache[name] = base_factory(name)
            return _session_cache[name]

        if args.warmup:
            print("[transcribe] warmup pass (discarded) ...")
            whisper_infer.greedy_decode_morphizen(
                gpu_factory, audio_fp, max_length=args.max_length, dtype=dtype
            )
        timings = {} if args.metrics else None
        gpu_tokens = whisper_infer.greedy_decode_morphizen(
            gpu_factory,
            audio_fp,
            max_length=args.max_length,
            timings=timings,
            dtype=dtype,
        )
        print(f"GPU: {whisper_infer.decode_text(gpu_tokens)!r}")
        if args.metrics:
            _print_metrics(timings, audio_s, label=f"MorphiZen EP ({prec}, GPU)")

    return 0


def _print_metrics(t, audio_s, label="MorphiZen EP (fp32, GPU)"):
    """Render the per-phase latency / RTF / throughput table for one backend.

    enc_ms      : encoder forward (mel -> hidden + cross-KV), runs once.
    prefill_ms  : S=4 decoder prefill = time-to-first-token compute.
    decode_ms   : the whole S=1 decode loop (n steps).
    total_ms    : enc + prefill + decode (model compute only; excludes feature
                  extraction + detokenize).
    RTF         : total_ms / audio_duration; < 1.0 == faster than real time.
    decode tok/s: n_decode_steps / decode_s (steady-state per-token rate).

    ``label`` names the backend in the header (so --compare's two tables are
    distinguishable).

    Caveats (intentionally NOT printed to keep the output terse):
      * Numbers are compute-only -- they exclude feature extraction and
        detokenize.
      * GPU: warmup is on by default (--warmup), so these are steady-state and
        exclude the one-time model-DLL load + kernel autotune; pass --no-warmup
        to see the cold-start cost. Decode uses the IOBinding GPU-resident
        KV-cache path (self-KV aliased past<->present, cross-KV bound once, only
        tiny per-step inputs + one logits D2H), so decode tok/s is a real kernel
        figure, not Python KV-marshaling overhead.
      * CPU: runs the DYNAMIC fp32 graph with growing-past KV (the only valid CPU
        reference) and is NOT warmed, so it includes ORT's first-run graph setup.
    """
    enc = t["enc_ms"]
    prefill = t["prefill_ms"]
    decode = t["decode_ms"]
    n = t["n_decode_steps"]
    total = enc + prefill + decode
    per_tok = decode / n if n else 0.0
    tps = n / (decode / 1e3) if decode > 0 else 0.0
    ttft = enc + prefill  # encoder + first-token prefill
    rtf = (total / 1e3) / audio_s if audio_s > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"{label} -- audio {audio_s:.2f}s, {n} decode tokens")
    print("=" * 60)
    print(f"  encoder           : {enc:9.1f} ms")
    print(f"  prefill (S=4)     : {prefill:9.1f} ms")
    print(f"  decode loop       : {decode:9.1f} ms  ({per_tok:.1f} ms/tok)")
    print(f"  {'-' * 40}")
    print(f"  total compute     : {total:9.1f} ms")
    print(f"  TTFT (enc+prefill): {ttft:9.1f} ms")
    print(f"  decode throughput : {tps:9.2f} tok/s")
    print(
        f"  RTF               : {rtf:9.4f}  ({1 / rtf:.1f}x real time)"
        if rtf > 0
        else "  RTF               :       n/a"
    )
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
