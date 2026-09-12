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
    ["hipblaslt.dll", "libhipblaslt.dll"],
]

HIPBLASLT_DATA = ("hipblaslt", "library")

# AMDGPU Generic Processors table from LLVM AMDGPUUsage; must stay in sync with
# genericTargetFor in LlvmIrJit.cpp.
_GENERIC_MEMBERS = {
    "gfx9-generic": ("gfx900", "gfx902", "gfx904", "gfx906", "gfx909", "gfx90c"),
    "gfx9-4-generic": ("gfx942", "gfx950"),
    "gfx10-1-generic": ("gfx1010", "gfx1011", "gfx1012", "gfx1013"),
    "gfx10-3-generic": (
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1033",
        "gfx1034",
        "gfx1035",
        "gfx1036",
    ),
    "gfx11-generic": (
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
    ),
    "gfx12-generic": ("gfx1200", "gfx1201"),
}


def _resolve_tensile_arch(library, requested):
    if (library / requested).is_dir():
        return requested
    members = _GENERIC_MEMBERS.get(requested)
    if not members:
        return None
    present = [a for a in members if (library / a).is_dir()]
    if not present:
        return None
    # gfx1151 is what the pinned dist and the CI GPU are; a dist carrying the
    # whole family would otherwise resolve to the lowest member.
    if "gfx1151" in present:
        return "gfx1151"
    return present[0]


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

    library = bin_dir.joinpath(*HIPBLASLT_DATA)
    tensile_arch = _resolve_tensile_arch(library, arch)
    if tensile_arch is None:
        available = []
        if library.is_dir():
            available = sorted(p.name for p in library.iterdir() if p.is_dir())
        hint = f" (available: {', '.join(available)})" if available else ""
        print(
            f"ERROR: hipBLASLt Tensile data for {arch} not found: "
            f"{library / arch}{hint}",
            file=sys.stderr,
        )
        return 1
    src_data = library / tensile_arch
    if tensile_arch != arch:
        print(
            f"  hipBLASLt Tensile arch {arch} -> {tensile_arch} "
            f"(device ISA present in dist)"
        )
    dst_data = dest.joinpath(*HIPBLASLT_DATA, tensile_arch)
    shutil.copytree(src_data, dst_data)
    count = sum(1 for p in dst_data.rglob("*") if p.is_file())
    print(
        f"  packaged hipBLASLt Tensile data: "
        f"{'/'.join(HIPBLASLT_DATA)}/{tensile_arch} "
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
