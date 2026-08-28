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

ROCM_DLL_GROUPS = [
    ["amdhip64*.dll"],
    ["amd_comgr*.dll"],
    ["hipblaslt.dll", "libhipblaslt.dll"],
    ["hiprtc*.dll"],
]

HIPBLASLT_DATA = ("hipblaslt", "library")


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
    bin_dir = dist / "bin"
    if not bin_dir.is_dir():
        print(f"ERROR: --rocm-dist has no bin/: {dist}", file=sys.stderr)
        return 1

    for group in ROCM_DLL_GROUPS:
        hits = sorted({p for pat in group for p in bin_dir.glob(pat) if p.is_file()})
        if not hits:
            print(
                f"ERROR: no ROCm runtime library matching {' / '.join(group)} "
                f"in {bin_dir}",
                file=sys.stderr,
            )
            return 1
        for src in hits:
            shutil.copy2(src, dest / src.name)
            print(f"  packaged ROCm dll: {src.name} <- {src}")

    src_data = bin_dir.joinpath(*HIPBLASLT_DATA, arch)
    if not src_data.is_dir():
        print(
            f"ERROR: hipBLASLt Tensile data for {arch} not found: {src_data}",
            file=sys.stderr,
        )
        return 1
    dst_data = dest.joinpath(*HIPBLASLT_DATA, arch)
    shutil.copytree(src_data, dst_data)
    count = sum(1 for p in dst_data.rglob("*") if p.is_file())
    print(
        f"  packaged hipBLASLt Tensile data: {'/'.join(HIPBLASLT_DATA)}/{arch} "
        f"({count} files) <- {src_data}"
    )
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
        "--dest", required=True, help="Destination dir (the wheel's onnxruntime/capi)."
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
        help="GPU arch whose hipBLASLt Tensile data to bundle, e.g. gfx1151. A "
        "multi-arch distribution carries every arch; the wheel ships one.",
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
        "--optional-dll",
        action="append",
        default=[],
        metavar="PATH",
        help="Externally-built runtime library to bundle if present (repeatable), "
        "e.g. the amdgpu-ep/hip-backend DLLs built in a separate repo. Missing "
        "paths warn instead of failing, so a plain EP-only wheel build still works.",
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

    for raw in args.optional_dll:
        lib = Path(raw)
        if lib.is_file():
            shutil.copy2(lib, dest / lib.name)
            print(f"  packaged optional dll: {lib.name} <- {lib}")
        else:
            print(f"  WARNING: optional dll not found (skipping): {lib}")

    rc = _copy_rocm_runtime(Path(args.rocm_dist), args.rocm_arch, dest)
    if rc:
        return rc

    if args.with_crt:
        _copy_crt_libs(dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
