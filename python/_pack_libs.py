#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

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


def _copy_rocm_runtime(dist: Path, arch: str, dest: Path) -> int:
    # GEMMs route through the Composable Kernel / reference kernels compiled
    # into custom_kernels, so no vendor BLAS runtime library or Tensile data is
    # bundled. amdhip64 is loaded from the ROCm dist at runtime.
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dll",
        action="append",
        required=True,
        metavar="PATH",
        help="Path to a native library to bundle (repeatable). The EP plugin "
        "library is required; the JIT compiler is linked into it.",
    )
    ap.add_argument(
        "--dest",
        required=True,
        help="Destination dir (the wheel's onnxruntime_ep_amdgpu).",
    )
    ap.add_argument(
        "--rocm-dist",
        required=True,
        metavar="PATH",
        help="TheRock ROCm SDK the EP was built against (THEROCK_DIST). Its "
        "runtime libraries are bundled so the wheel needs no ROCm install.",
    )
    ap.add_argument(
        "--rocm-arch",
        required=True,
        metavar="GFX",
        help="Device ISA whose hipBLASLt Tensile data to bundle, e.g. gfx1151. "
        "A generic compile target (gfx11-generic) is mapped to a concrete "
        "ISA present in the dist. A multi-arch distribution carries every "
        "arch; the wheel ships one.",
    )
    ap.add_argument(
        "--data-file",
        action="append",
        default=[],
        metavar="PATH",
        help="Non-library file the EP loads from its own directory at runtime "
        "(repeatable), e.g. HipFusionPatterns.pdl.mlir. Missing files are a "
        "hard error: the EP cannot compile a model without them.",
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

    for raw in args.data_file:
        data = Path(raw)
        if not data.is_file():
            print(f"ERROR: data file not found: {data}", file=sys.stderr)
            return 1
        shutil.copy2(data, dest / data.name)
        print(f"  packaged data file: {data.name} <- {data}")

    for raw in args.extra_lib:
        lib = Path(raw)
        if lib.is_file():
            shutil.copy2(lib, dest / lib.name)
            print(f"  packaged import lib: {lib.name} <- {lib}")
        else:
            print(f"  WARNING: extra import lib not found: {lib}")

    rc = _copy_rocm_runtime(Path(args.rocm_dist), args.rocm_arch, dest)
    if rc:
        return rc

    if args.with_crt:
        _copy_crt_libs(dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
