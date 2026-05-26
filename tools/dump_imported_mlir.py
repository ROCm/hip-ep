#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Dump the MLIR bytecode that the MorphiZen EP would feed to hip-compiler.

The EP path lowers a real ``onnxruntime.InferenceSession`` to MLIR before
handing it to hip-compiler's pass pipeline. When a model fails inside that
pipeline (typical symptom: SSA dominance error, op-not-bufferized, ...) the
fastest way to bisect is to re-run hip-mlir-opt against the *exact* MLIR
the EP would have produced — but the bytecode normally lives under
``%TEMP%\\morphizen_dumps\\<cache_key>\\mlir_bytecode_dump.mlir`` keyed on
an opaque hash, which is awkward to script around.

This wrapper forces the bytecode to a user-named path by exporting
``HIPDNN_EP_BYTECODE_DUMP_PATH`` and ``HIPDNN_EP_ALLOW_CPU_FALLBACK=1``
(so the EP doesn't throw on a known compile-failure model — the only
case we care about here is "I want the dump even though compile fails").
It then creates a session and triggers ``GetCapability`` by initializing
it — no inference is run, no inputs are needed.

Usage::

    python tools/dump_imported_mlir.py \\
        --ep-dll install/dist/bin/onnxruntime_morphizen_ep.dll \\
        --ep-config install/dist/bin/morphizen_config.json \\
        --model path/to/embedding.onnx \\
        --out dump.mlirbc

Then convert to readable MLIR with::

    hip-mlir-opt dump.mlirbc -o dump.mlir

And bisect failing passes individually::

    hip-mlir-opt dump.mlir \\
        --hip-add-context-arg --convert-onnx-to-hip \\
        --mlir-print-ir-after-failure \\
        --mlir-print-stacktrace-on-diagnostic

See ``docs/dev/bisection.md`` for the end-to-end workflow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump MLIR bytecode the MorphiZen EP imports for an ONNX model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ep-dll",
        required=True,
        type=Path,
        help="Path to onnxruntime_morphizen_ep.dll.",
    )
    parser.add_argument(
        "--ep-config",
        required=True,
        type=Path,
        help="Path to morphizen_config.json (passed as the EP's "
        "`config_file` provider option).",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="ONNX model to import.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output bytecode file (.mlirbc by convention).",
    )
    parser.add_argument(
        "--ep-name",
        default="MorphiZenExecutionProvider",
        help="EP name to register under (default: MorphiZenExecutionProvider).",
    )
    args = parser.parse_args()

    for path, label in [
        (args.ep_dll, "--ep-dll"),
        (args.ep_config, "--ep-config"),
        (args.model, "--model"),
    ]:
        if not path.exists():
            print(f"error: {label} {path} does not exist", file=sys.stderr)
            return 2

    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    # `HIPDNN_EP_BYTECODE_DUMP_PATH`: forces level-1 pass to dump
    # bytecode to this exact file (see
    # backend-mlir-compiler/level-1-pass/src/pass_main.cpp). Distinct
    # from `HIPDNN_EP_IR_DUMP_PATH` (which is the hip-compiler-level
    # "IR after each pass" text dump used during pass bisection).
    # `HIPDNN_EP_ALLOW_CPU_FALLBACK=1`: lets the EP gracefully return on
    # compile failure instead of throwing — we already have the dump by
    # the time compile failure happens, and we don't want to mask the
    # dump output by raising a Python exception that may obscure the
    # "dump succeeded" signal.
    os.environ["HIPDNN_EP_BYTECODE_DUMP_PATH"] = str(args.out)
    os.environ["HIPDNN_EP_ALLOW_CPU_FALLBACK"] = "1"

    # Ensure the EP DLL's directory is on PATH so the Windows loader
    # finds co-located dependencies (hip-compiler.dll, ROCm runtime DLLs
    # placed next to the EP, etc.).
    ep_dir = str(args.ep_dll.resolve().parent)
    os.environ["PATH"] = ep_dir + os.pathsep + os.environ.get("PATH", "")

    import onnxruntime as ort

    print(f"[dump] ORT version : {ort.__version__}")
    print(f"[dump] EP DLL      : {args.ep_dll}")
    print(f"[dump] EP config   : {args.ep_config}")
    print(f"[dump] Model       : {args.model}")
    print(f"[dump] Out         : {args.out}")

    capi = ort.capi._pybind_state
    capi.register_execution_provider_library(args.ep_name, str(args.ep_dll.resolve()))

    devices = [d for d in capi.get_ep_devices() if d.ep_name == args.ep_name]
    if not devices:
        print(
            f"error: no EP device named {args.ep_name!r} after register",
            file=sys.stderr,
        )
        return 3

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_provider_for_devices(
        devices,
        {"config_file": str(args.ep_config.resolve())},
    )

    try:
        ort.InferenceSession(str(args.model), sess_options=opts)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[dump] InferenceSession init raised: {type(exc).__name__}: "
            f"{exc}\n[dump] (continuing — bytecode dump fires before "
            f"compile, so the output file may still be valid)",
            file=sys.stderr,
        )

    if not args.out.exists() or args.out.stat().st_size == 0:
        print(
            f"error: no MLIR bytecode dumped to {args.out}. "
            f"Check that the EP DLL is the one with HIPDNN_EP_IR_DUMP_PATH "
            f"support (rebuild if older).",
            file=sys.stderr,
        )
        return 4

    print(
        f"[dump] OK  ({args.out.stat().st_size} bytes)\n"
        f"[dump] Next: hip-mlir-opt {args.out} -o {args.out.with_suffix('.mlir')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
