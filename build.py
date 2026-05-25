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
    python build.py --build-oga    # Also build the onnxruntime-genai fork
    python build.py --build-vulkan # Also build the Vulkan baseline (llama.cpp)
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
VULKAN_SDK = INSTALL / "vulkan-sdk"
LLAMACPP_SRC = INSTALL / "llama.cpp"
LLAMACPP_BUILD = INSTALL / "llama.cpp-build"
LLAMACPP_DIST = INSTALL / "llama-vulkan"

THEROCK_URL = (
    "https://repo.amd.com/rocm/tarball/therock-dist-windows-gfx1151-7.11.0.tar.gz"
)

# Pinned llama.cpp commit + Vulkan SDK version for reproducible Vulkan baselines.
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMACPP_REF = "683c5acb90478a9e7e20eb65a1bfee334635216d"
VULKAN_VERSION = "1.4.341.1"
VULKAN_INSTALLER_NAME = f"vulkansdk-windows-X64-{VULKAN_VERSION}.exe"
# ?Human=true disables LunarG's download-token throttling for direct fetches.
VULKAN_INSTALLER_URL = (
    f"https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/windows/"
    f"{VULKAN_INSTALLER_NAME}?Human=true"
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
    archive = CACHE / Path(THEROCK_URL).name
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

ORT_VERSION = "1.25.1"  # must match pip onnxruntime-directml for Python API compat


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
        # Recent TheRock builds emit an informational "HIP Library Path: ..."
        # line on stdout before the arch line; pick the first line that
        # actually looks like a gfx target.
        for line in r.stdout.strip().splitlines():
            tok = line.strip()
            if tok.startswith("gfx"):
                return tok
        # Fallback: legacy single-line output.
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
    # Force the OGA cmake to use the same Python interpreter that's running
    # build.py (the conda env's python). cmake otherwise auto-picks the first
    # python on PATH (e.g. miniforge base, system Python), producing a .pyd
    # whose ABI tag (cp312/cp313) doesn't match the install target's cp314.
    py_for_cmake = sys.executable.replace("\\", "/")
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
        # pybind11 v2.13.6 (vendored by OGA) uses the removed
        # distutils.sysconfig under Python 3.12+; PYBIND11_FINDPYTHON=ON
        # routes through modern CMake FindPython3 instead.
        "--cmake_extra_defines",
        "PYBIND11_FINDPYTHON=ON",
        # PYBIND11_FINDPYTHON routes through CMake FindPython, which keys off
        # `Python_EXECUTABLE` (no `3` suffix) — passing `Python3_EXECUTABLE`
        # alone gets ignored with a "Manually-specified variables were not used"
        # warning, and cmake then auto-detects the wrong python on PATH.
        "--cmake_extra_defines",
        f"Python_EXECUTABLE={py_for_cmake}",
    ]
    # OGA's terminal `PyPackageBuild` ninja target invokes `cmd /C "... && -m pip
    # wheel --no-deps ."` — note the missing `python` before `-m pip`. The C/C++
    # artifacts (DLL, EXE, .pyd) are already built before this step fails, so
    # tolerate non-zero exit and finish the wheel ourselves below.
    rc = subprocess.run(build_cmd).returncode
    if rc != 0:
        log(
            f"  OGA build.py exited {rc} (PyPackageBuild step likely failed; "
            "binaries should still be present)."
        )

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
        else:
            log(f"  ERROR: missing OGA artifact {src}")
            sys.exit(1)

    # Required by model_benchmark.exe: the freshly-built ORT must shadow any
    # System32 onnxruntime.dll (often v1.17 = API 17) so the EP-built-against
    # ORT 1.24 (API 24) can load.
    for ort_dll in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
        src = ORT / "lib" / ort_dll
        if src.exists():
            shutil.copy2(src, bin_dir / ort_dll)

    # -- build wheel (workaround for OGA PyPackageBuild bug) --
    # OGA stages the wheel layout under <build>/wheel/ before invoking pip.
    # If the broken target left no .whl, run pip wheel manually from there.
    wheel_dir = oga_bin / "wheel"
    wheels = (
        list(wheel_dir.glob("onnxruntime_genai_directml-*.whl"))
        if wheel_dir.exists()
        else []
    )
    if not wheels and wheel_dir.exists() and (wheel_dir / "setup.py").exists():
        log("  PyPackageBuild produced no wheel — running pip wheel manually.")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                str(wheel_dir),
            ],
            check=True,
            cwd=str(wheel_dir),
        )
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
        log("  ERROR: No OGA wheel produced; cannot install onnxruntime_genai.")
        sys.exit(1)

    log("  OGA build complete.")


# ---------------------------------------------------------------------------
# Vulkan baseline (llama.cpp built against the LunarG Vulkan SDK)
# ---------------------------------------------------------------------------


def _download_with_browser_ua(url, dest):
    """Wrap download() with a Mozilla UA opener. LunarG's CDN returns 403/404
    to the default Python urllib User-Agent."""
    prev_opener = urllib.request._opener
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    try:
        download(url, dest)
    finally:
        urllib.request.install_opener(prev_opener)


def fetch_vulkan_sdk():
    log("Setting up Vulkan SDK ...")
    sentinel = VULKAN_SDK / ".ok"
    if sentinel.exists():
        log("  Already installed.")
        return

    installer = CACHE / VULKAN_INSTALLER_NAME
    _download_with_browser_ua(VULKAN_INSTALLER_URL, installer)

    if VULKAN_SDK.exists():
        shutil.rmtree(VULKAN_SDK)
    VULKAN_SDK.mkdir(parents=True, exist_ok=True)

    # Qt Installer Framework based; --root makes it install user-writable
    # (no admin/UAC) and the silent flags suppress the GUI. Default component
    # set covers what llama.cpp needs (Headers, Loader, glslc, validation).
    log(f"  Running installer (silent) -> {VULKAN_SDK}")
    r = subprocess.run(
        [
            str(installer),
            "--root",
            str(VULKAN_SDK),
            "--accept-licenses",
            "--default-answer",
            "--confirm-command",
            "install",
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Vulkan SDK installer exited with code {r.returncode}. "
            "If a UAC prompt was dismissed, re-run from an elevated shell."
        )

    must_exist = [
        VULKAN_SDK / "Include" / "vulkan" / "vulkan.h",
        VULKAN_SDK / "Lib" / "vulkan-1.lib",
        VULKAN_SDK / "Bin" / "glslc.exe",
    ]
    missing = [p for p in must_exist if not p.exists()]
    if missing:
        raise RuntimeError(
            "Vulkan SDK install looks incomplete. Missing:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    sentinel.touch()
    log(f"  Vulkan SDK {VULKAN_VERSION} ready.")


def fetch_llamacpp():
    log("Setting up llama.cpp source ...")
    sentinel = LLAMACPP_SRC / ".ok"
    if sentinel.exists():
        # Verify the pinned commit is checked out — if HEAD moved we don't
        # want to silently build the wrong thing.
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(LLAMACPP_SRC),
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head == LLAMACPP_REF:
            log(f"  Already at pinned commit {LLAMACPP_REF[:8]}.")
            return
        log(f"  HEAD is {head[:8]}, expected {LLAMACPP_REF[:8]} — re-checking out.")

    if not LLAMACPP_SRC.exists():
        LLAMACPP_SRC.parent.mkdir(parents=True, exist_ok=True)
        log(f"  Cloning {LLAMACPP_REPO} ...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", LLAMACPP_REPO, str(LLAMACPP_SRC)],
            check=True,
        )

    log(f"  Fetching pinned commit {LLAMACPP_REF[:8]} ...")
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", LLAMACPP_REF],
        cwd=str(LLAMACPP_SRC),
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", LLAMACPP_REF],
        cwd=str(LLAMACPP_SRC),
        check=True,
    )

    sentinel.touch()
    log("  llama.cpp ready.")


def build_vulkan():
    """Fetch the Vulkan SDK + llama.cpp source, then build llama.cpp with
    GGML_VULKAN=ON. Installs binaries into install/llama-vulkan/bin."""
    log("Building Vulkan baseline (llama.cpp) ...")

    fetch_vulkan_sdk()
    fetch_llamacpp()

    _ensure_msvc_env()
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            log(f"  ERROR: {tool} not found. Run: conda activate hipdnn-ep")
            sys.exit(1)

    # Make the SDK visible to CMake's FindVulkan via the standard env var.
    os.environ["VULKAN_SDK"] = str(VULKAN_SDK)
    os.environ["PATH"] = str(VULKAN_SDK / "Bin") + os.pathsep + os.environ["PATH"]

    LLAMACPP_BUILD.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        "-S",
        str(LLAMACPP_SRC),
        "-B",
        str(LLAMACPP_BUILD),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={LLAMACPP_DIST}",
        "-DGGML_VULKAN=ON",
        "-DGGML_NATIVE=ON",
        "-DGGML_CUDA=OFF",
        "-DGGML_HIP=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_CURL=OFF",
        f"-DVulkan_INCLUDE_DIR={VULKAN_SDK / 'Include'}",
        f"-DVulkan_LIBRARY={VULKAN_SDK / 'Lib' / 'vulkan-1.lib'}",
        f"-DVulkan_GLSLC_EXECUTABLE={VULKAN_SDK / 'Bin' / 'glslc.exe'}",
    ]

    if shutil.which("sccache"):
        cmake_args += [
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ]

    log("  Configuring ...")
    subprocess.run(cmake_args, check=True)

    log("  Compiling ...")
    subprocess.run(
        ["cmake", "--build", str(LLAMACPP_BUILD), "--config", "Release"],
        check=True,
    )

    log("  Installing ...")
    subprocess.run(
        ["cmake", "--install", str(LLAMACPP_BUILD), "--config", "Release"],
        check=True,
    )
    log(f"  Vulkan baseline ready -> {LLAMACPP_DIST}")


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
    parser.add_argument(
        "--build-vulkan",
        action="store_true",
        help="Build the Vulkan baseline (llama.cpp + LunarG Vulkan SDK)",
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

    if args.build_vulkan:
        build_vulkan()

    log(f"  TheRock SDK:    {THEROCK}")
    log(f"  Dependencies:   {DEPS}")
    log(f"  ONNX Runtime:   {ORT}")
    if not args.skip_build:
        log(f"  Build output:   {BUILD}")
        log(f"  Install dir:    {DIST}")
    if args.build_oga:
        log(f"  OGA source:     {OGA_SOURCE}")
        log(f"  OGA build:      {OGA_BUILD}")
    if args.build_vulkan:
        log(f"  Vulkan SDK:     {VULKAN_SDK}")
        log(f"  llama.cpp src:  {LLAMACPP_SRC} (commit {LLAMACPP_REF[:8]})")
        log(f"  Vulkan binaries:{LLAMACPP_DIST}")


if __name__ == "__main__":
    main()
