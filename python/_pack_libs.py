#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Build helper: stage native artifacts into the wheel's onnxruntime/capi.

Invoked by python/CMakeLists.txt's `wheel` target (NOT shipped at runtime).
Copies the prebuilt native libraries and, on Windows, the MSVC/WinSDK CRT import
libraries the JIT linker (lld-link) resolves against. The CRT libs are located
by scanning the `LIB` environment variable, which the VS dev environment / MSVC
toolset populates with the MSVC and WinSDK lib directories.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# CRT/WinSDK import libraries the per-model lld-link step needs (the same set
# the project's runtime lib setup stages for lld-link).
CRT_LIBS = [
    "msvcrt.lib",
    "vcruntime.lib",
    "oldnames.lib",
    "libcpmt.lib",
    "libcmt.lib",
    "ucrt.lib",
    "kernel32.lib",
    "user32.lib",
]


def _find_in_lib_env(name: str):
    for d in os.environ.get("LIB", "").split(os.pathsep):
        if not d:
            continue
        cand = Path(d) / name
        if cand.is_file():
            return cand
    return None


def _copy_crt_libs(dest: Path) -> int:
    missing = []
    for name in CRT_LIBS:
        src = _find_in_lib_env(name)
        if src is None:
            missing.append(name)
            continue
        shutil.copy2(src, dest / name)
        print(f"  packaged CRT lib: {name} <- {src}")
    if missing:
        print(
            "  WARNING: CRT import libs not found on %LIB%: "
            + ", ".join(missing)
            + "\n  Run the wheel build from a VS dev environment (LIB set), "
            "or the JIT linker will fail at inference time."
        )
    return len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dll",
        action="append",
        required=True,
        metavar="PATH",
        help="Path to a native library to bundle (repeatable). The EP plugin "
        "library and its JIT compiler plugin (hip-compiler) are both required.",
    )
    ap.add_argument(
        "--dest", required=True, help="Destination dir (the wheel's onnxruntime/capi)."
    )
    ap.add_argument(
        "--extra-lib",
        action="append",
        default=[],
        metavar="PATH",
        help="Additional import library to bundle (repeatable), e.g. "
        "hip_custom_kernels.lib.",
    )
    ap.add_argument(
        "--with-crt",
        action="store_true",
        help="Also copy MSVC/WinSDK CRT import libs (Windows).",
    )
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for raw in args.dll:
        lib = Path(raw)
        if not lib.is_file():
            print(f"ERROR: library not found: {lib}", file=sys.stderr)
            return 1
        shutil.copy2(lib, dest / lib.name)
        print(f"  packaged library: {lib.name} <- {lib}")

    for raw in args.extra_lib:
        lib = Path(raw)
        if lib.is_file():
            shutil.copy2(lib, dest / lib.name)
            print(f"  packaged import lib: {lib.name} <- {lib}")
        else:
            print(f"  WARNING: extra import lib not found: {lib}")

    if args.with_crt:
        _copy_crt_libs(dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
