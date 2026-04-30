#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Build automation for onnx-hipdnn-ep.

Downloads dependencies and builds the project:
  1. Downloads and extracts TheRock ROCm SDK
  2. Downloads and extracts prebuilt LLVM/MLIR/FlatBuffers/Protobuf
  3. Downloads and extracts ONNX Runtime pre-built binaries
  4. Configures, builds, and installs via CMake

All artifacts are placed under install/ in the project root.

Prerequisite: activate the conda environment first:
    conda env create -f environment.yml   # one-time
    conda activate hipdnn-ep

Usage:
    python build.py                # Full setup + build
    python build.py --skip-build   # Setup only (download deps)
    python build.py --clean        # Remove install/ and start fresh
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTALL = ROOT / "install"
CACHE = INSTALL / "_cache"
THEROCK = INSTALL / "therock"
DEPS = INSTALL / "deps"
ORT = INSTALL / "onnxruntime"
BUILD = INSTALL / "build"
DIST = INSTALL / "dist"
OGA_SOURCE = INSTALL / "oga-source"
OGA_BUILD = INSTALL / "oga-build"

THEROCK_URL = (
    "https://repo.amd.com/rocm/tarball/therock-dist-windows-gfx1150-7.11.0.tar.gz"
)

# Prebuilt deps — keep in sync with scripts/setup-prebuilt.sh
_PREBUILT_BASE = "https://github.com/wcy123/llvm-mlir-prebuilt/releases/download"
_PREBUILTS = [
    ("llvm-22.1.0-release", "llvm-22.1.0-release-windows-x64.zip"),
    ("protobuf-34.0-release", "protobuf-34.0-release-windows-x64.zip"),
    ("flatbuffers-25.12.19-release", "flatbuffers-25.12.19-release-windows-x64.zip"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg):
    print(f"[build] {msg}")


def download(url, dest):
    """Download *url* to *dest* with a progress indicator. Skips if cached."""
    if dest.exists():
        log(f"  Cached: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"  Downloading {dest.name} ...")

    def _progress(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(100.0, done * 100.0 / total)
            print(
                f"\r  {done / 1048576:.1f} / {total / 1048576:.1f} MB ({pct:.0f}%)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_progress)
        print()
        tmp.rename(dest)
    except Exception as exc:
        print()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {url}") from exc


def _tar_extract(archive, dest, *, strip=0):
    """Extract a tar archive, optionally stripping leading path components."""
    log(f"  Extracting {archive.name} -> {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts:
                continue
            if strip:
                parts = p.parts
                if len(parts) <= strip:
                    continue
                member.name = str(Path(*parts[strip:]))
            try:
                if sys.version_info >= (3, 12):
                    tar.extract(member, dest, filter="data")
                else:
                    tar.extract(member, dest)
            except (OSError, PermissionError):
                pass


def _zip_extract(archive, dest, *, strip=0):
    """Extract a zip archive, optionally stripping leading path components."""
    log(f"  Extracting {archive.name} -> {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if strip:
                parts = p.parts
                if len(parts) <= strip:
                    continue
                out = dest / Path(*parts[strip:])
            else:
                out = dest / p
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _read_ci_env(*keys):
    """Read env var values from .github/workflows/windows-build.yml (single source of truth)."""
    ci_yaml = ROOT / ".github" / "workflows" / "windows-build.yml"
    text = ci_yaml.read_text()
    result = {}
    for key in keys:
        m = re.search(rf"^\s+{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
        if not m:
            raise RuntimeError(f"{key} not found in {ci_yaml}")
        result[key] = m.group(1).strip().strip('"').strip("'")
    return result


# ---------------------------------------------------------------------------
# TheRock ROCm SDK
# ---------------------------------------------------------------------------


def fetch_therock():
    log("Setting up TheRock ROCm SDK ...")
    sentinel = THEROCK / ".ok"
    if sentinel.exists():
        log("  Already installed.")
        return
    archive = CACHE / "therock.tar.gz"
    download(THEROCK_URL, archive)
    if THEROCK.exists():
        shutil.rmtree(THEROCK)
    _tar_extract(archive, THEROCK, strip=1)
    sentinel.touch()
    log("  TheRock SDK ready.")


# ---------------------------------------------------------------------------
# Prebuilt LLVM / MLIR / Protobuf / FlatBuffers
# Tags and assets mirror scripts/setup-prebuilt.sh — keep in sync.
# ---------------------------------------------------------------------------


def fetch_prebuilt_deps():
    log("Setting up prebuilt dependencies (LLVM, Protobuf, FlatBuffers) ...")
    for tag, asset in _PREBUILTS:
        sentinel = DEPS / f".{tag}.ok"
        if sentinel.exists():
            log(f"  {tag}: already installed.")
            continue
        url = f"{_PREBUILT_BASE}/{tag}/{asset}"
        archive = CACHE / asset
        download(url, archive)
        _zip_extract(archive, DEPS)
        sentinel.touch()
        log(f"  {tag}: done.")


# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------

ORT_VERSION = "1.24.4"  # must match pip onnxruntime-directml for Python API compat


def fetch_onnxruntime():
    log("Setting up ONNX Runtime ...")
    sentinel = ORT / ".ok"
    if sentinel.exists():
        _generate_ort_cmake_config()
        log("  Already installed.")
        return

    version = ORT_VERSION
    log(f"  Target version: {version}")

    target = f"onnxruntime-win-x64-{version}.zip"
    url = (
        f"https://github.com/microsoft/onnxruntime/releases/download/"
        f"v{version}/{target}"
    )

    archive = CACHE / target
    download(url, archive)
    if ORT.exists():
        shutil.rmtree(ORT)
    _zip_extract(archive, ORT, strip=1)
    _generate_ort_cmake_config()
    sentinel.touch()
    log(f"  ONNX Runtime {version} ready.")


def _generate_ort_cmake_config():
    """Create a CMake config so find_package(onnxruntime CONFIG) works."""
    cmake_dir = ORT / "lib" / "cmake" / "onnxruntime"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    config = cmake_dir / "onnxruntimeConfig.cmake"
    config.write_text("""\
# Auto-generated by build.py for pre-built ONNX Runtime binaries.
get_filename_component(_ort_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include"
    IMPORTED_IMPLIB "${_ort_root}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${_ort_root}/lib/onnxruntime.dll"
  )
endif()

set(onnxruntime_FOUND TRUE)
""")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _ensure_submodules():
    log("  Checking git submodules ...")
    r = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    uninitialized = any(
        line.strip().startswith("-") for line in r.stdout.splitlines() if line.strip()
    )
    if uninitialized:
        log("  Initializing submodules ...")
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            check=True,
            cwd=str(ROOT),
        )
    else:
        log("  Submodules OK.")


def _ensure_dia_sdk_junction():
    """Create C:\\msvsn2022 junction if needed (LLVM prebuilt hardcoded path)."""
    junction = Path("C:/msvsn2022")
    if junction.exists():
        return
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.exists():
        log("  WARNING: vswhere not found — cannot create DIA SDK junction.")
        return
    r = subprocess.run(
        [str(vswhere), "-latest", "-property", "installationPath"],
        capture_output=True,
        text=True,
    )
    vs_path = r.stdout.strip()
    if not vs_path:
        log("  WARNING: No Visual Studio installation found.")
        return
    log(f"  Creating DIA SDK junction: C:\\msvsn2022 -> {vs_path}")
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", "C:\\msvsn2022", vs_path],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        log("  WARNING: mklink failed (may need admin). See docs/quick_start.md.")


def _detect_gpu_arch():
    exe = THEROCK / "lib" / "llvm" / "bin" / "amdgpu-arch.exe"
    if not exe.exists():
        return None
    try:
        r = subprocess.run(
            [str(exe)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = r.stdout.strip().splitlines()
        return lines[0].strip() if lines else None
    except Exception:
        return None


def _ensure_msvc_env():
    """Detect VS2022 and inject its x64 environment into this process."""
    if shutil.which("cl"):
        return
    log("  cl.exe not on PATH — locating Visual Studio ...")
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.exists():
        log("  ERROR: vswhere.exe not found. Install Visual Studio 2022.")
        sys.exit(1)
    r = subprocess.run(
        [str(vswhere), "-latest", "-property", "installationPath"],
        capture_output=True,
        text=True,
    )
    vs_path = r.stdout.strip()
    if not vs_path:
        log("  ERROR: No Visual Studio installation found.")
        sys.exit(1)
    vcvarsall = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.exists():
        log(f"  ERROR: vcvarsall.bat not found at {vcvarsall}")
        sys.exit(1)
    log(f"  Sourcing MSVC environment from {vs_path} ...")
    # Run vcvarsall and dump the resulting environment
    r = subprocess.run(
        ["cmd", "/C", "call", str(vcvarsall), "x64", "&&", "set"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log("  ERROR: vcvarsall.bat failed.")
        sys.exit(1)
    for line in r.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key] = value
    if not shutil.which("cl"):
        log("  ERROR: cl.exe still not found after sourcing vcvarsall.bat.")
        sys.exit(1)
    # Restart sccache so the daemon inherits the MSVC environment (cl.exe
    # depends on DLLs that must be on PATH for CreateProcess to succeed).
    if shutil.which("sccache"):
        subprocess.run(["sccache", "--stop-server"], capture_output=True)
    log("  MSVC environment ready.")


def configure_and_build():
    log("Building onnx-hipdnn-ep ...")

    _ensure_msvc_env()
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            log(f"  ERROR: {tool} not found. Run: conda activate hipdnn-ep")
            sys.exit(1)

    _ensure_submodules()
    _ensure_dia_sdk_junction()

    BUILD.mkdir(parents=True, exist_ok=True)

    # -- cmake configure args --
    prefix_paths = [str(DEPS)]
    if (ORT / ".ok").exists():
        prefix_paths.append(str(ORT))

    cmake_args = [
        "cmake",
        "-S",
        str(ROOT),
        "-B",
        str(BUILD),
        "-G",
        "Ninja",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
        f"-DCMAKE_PREFIX_PATH={';'.join(prefix_paths)}",
        f"-DCMAKE_INSTALL_PREFIX={DIST}",
        "-DBUILD_HIP_TOOLS=ON",
        "-DBUILD_EP=ON",
    ]

    if shutil.which("sccache"):
        cmake_args += [
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ]

    cmake_args.append(f"-DPython3_EXECUTABLE={sys.executable}")

    # Real runtime (TheRock + detected GPU) vs mock
    therock_ready = (THEROCK / ".ok").exists()
    if therock_ready:
        gpu_arch = _detect_gpu_arch()
        if gpu_arch:
            log(f"  GPU architecture: {gpu_arch}")
            cmake_args += [
                f"-DTHEROCK_DIST={THEROCK}",
                "-DHIP_PLATFORM=amd",
                f"-DHIP_ARCHITECTURES={gpu_arch}",
                "-DBUILD_MOCK_RUNTIME=OFF",
            ]
        else:
            log("  No GPU detected — using mock runtime.")
            cmake_args.append("-DBUILD_MOCK_RUNTIME=ON")
    else:
        log("  TheRock not installed — using mock runtime.")
        cmake_args.append("-DBUILD_MOCK_RUNTIME=ON")

    # -- configure --
    log("  Configuring ...")
    subprocess.run(cmake_args, check=True)

    # -- build --
    log("  Compiling ...")
    subprocess.run(
        ["cmake", "--build", str(BUILD), "--config", "RelWithDebInfo", "--parallel"],
        check=True,
    )

    # -- install --
    log("  Installing ...")
    subprocess.run(
        ["cmake", "--install", str(BUILD), "--config", "RelWithDebInfo"],
        check=True,
    )

    log(f"  Done. Installed to: {DIST}")


# ---------------------------------------------------------------------------
# OGA (onnxruntime-genai) fork
# ---------------------------------------------------------------------------


def _ensure_dml_header():
    """Fetch dml_provider_factory.h from ORT DirectML NuGet if missing."""
    header = ORT / "include" / "dml_provider_factory.h"
    if header.exists():
        return
    log("  Fetching dml_provider_factory.h from ORT DirectML NuGet ...")
    url = (
        "https://www.nuget.org/api/v2/package/"
        f"Microsoft.ML.OnnxRuntime.DirectML/{ORT_VERSION}"
    )
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith("dml_provider_factory.h"):
                header.write_bytes(z.read(name))
                return
    raise RuntimeError("dml_provider_factory.h not found in NuGet package")


def build_oga():
    """Build the onnxruntime-genai fork with MorphiZen EP device support."""
    ci = _read_ci_env("OGA_REPO", "OGA_REF")
    oga_repo = f"https://github.com/{ci['OGA_REPO']}.git"
    oga_ref = ci["OGA_REF"]
    log(f"Building OGA fork ({ci['OGA_REPO']} @ {oga_ref[:10]}) ...")

    if not (ORT / ".ok").exists():
        log(
            "  ERROR: ONNX Runtime not installed. Run build.py without --skip-build first."
        )
        sys.exit(1)

    _ensure_msvc_env()

    # -- clone --
    sentinel = OGA_SOURCE / ".ok"
    if not sentinel.exists():
        if OGA_SOURCE.exists():
            shutil.rmtree(OGA_SOURCE)
        log("  Cloning OGA fork ...")
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "feat/oga_hipdnn_experiment",
                oga_repo,
                str(OGA_SOURCE),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", oga_ref],
            check=True,
            cwd=str(OGA_SOURCE),
        )
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            check=True,
            cwd=str(OGA_SOURCE),
        )
        sentinel.touch()
    else:
        log("  OGA source already cloned.")

    _ensure_dml_header()

    # -- build --
    log("  Building OGA ...")
    build_cmd = [
        sys.executable,
        str(OGA_SOURCE / "build.py"),
        "--config",
        "RelWithDebInfo",
        "--cmake_generator",
        "Ninja",
        "--use_dml",
        "--ort_home",
        str(ORT),
        "--skip_tests",
        "--skip_examples",
        "--parallel",
        "--build_dir",
        str(OGA_BUILD),
        "--cmake_extra_defines",
        "CMAKE_CXX_FLAGS=/EHsc",
    ]
    subprocess.run(build_cmd, check=True)

    # -- install binaries --
    bin_dir = DIST / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    oga_bin = OGA_BUILD / "RelWithDebInfo"
    oga_files = {
        "onnxruntime-genai.dll": oga_bin / "onnxruntime-genai.dll",
        "model_benchmark.exe": oga_bin / "benchmark" / "c" / "model_benchmark.exe",
    }
    for name, src in oga_files.items():
        if src.exists():
            shutil.copy2(src, bin_dir / name)
            log(f"  Installed {name}")

    # -- install wheel --
    wheel_dir = oga_bin / "wheel"
    wheels = list(wheel_dir.glob("onnxruntime_genai_directml-*.whl"))
    if wheels:
        log(f"  Installing wheel: {wheels[0].name}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheels[0]),
            ],
            check=True,
        )
    else:
        log("  WARNING: No OGA wheel found.")

    log("  OGA build complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build automation for onnx-hipdnn-ep",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove install/ directory and start fresh",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only download dependencies, do not build",
    )
    parser.add_argument(
        "--build-oga",
        action="store_true",
        help="Build and install onnxruntime-genai fork with MorphiZen EP device support",
    )
    args = parser.parse_args()

    if args.clean:
        if INSTALL.exists():
            log(f"Removing {INSTALL} ...")
            shutil.rmtree(INSTALL)
        log("Clean complete.")
        return

    fetch_therock()
    fetch_prebuilt_deps()
    fetch_onnxruntime()

    if args.skip_build:
        log("")
        log("Setup complete (build skipped).")
    else:
        configure_and_build()
        log("")
        log("Setup + build complete!")

    if args.build_oga:
        build_oga()

    log(f"  TheRock SDK:    {THEROCK}")
    log(f"  Dependencies:   {DEPS}")
    log(f"  ONNX Runtime:   {ORT}")
    if not args.skip_build:
        log(f"  Build output:   {BUILD}")
        log(f"  Install dir:    {DIST}")
    if args.build_oga:
        log(f"  OGA source:     {OGA_SOURCE}")
        log(f"  OGA build:      {OGA_BUILD}")


if __name__ == "__main__":
    main()
