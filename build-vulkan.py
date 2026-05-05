#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Build automation for the Vulkan llama.cpp baseline.

Sets up everything needed to collect Vulkan numbers from llama.cpp:
  1. Downloads and silently installs the Vulkan SDK into install/vulkan-sdk
  2. Clones llama.cpp at a pinned commit into install/llama.cpp
  3. Configures (Ninja, GGML_VULKAN=ON, Release) and builds it
  4. Installs binaries into install/llama-vulkan/bin

All artifacts live under install/ — nothing is touched outside the repo.

Pinned versions match the local reference setup at
  C:\\local\\llama.cpp + C:\\VulkanSDK\\1.4.341.1
so remote and local numbers are directly comparable.

Prerequisite: activate the conda environment first (provides cmake + ninja):
    conda activate hipdnn-ep

Usage:
    python build-vulkan.py                # full setup + build
    python build-vulkan.py --skip-build   # just fetch SDK + sources
    python build-vulkan.py --clean        # wipe install/llama* + install/vulkan-sdk
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTALL = ROOT / "install"
CACHE = INSTALL / "_cache"
VULKAN_SDK = INSTALL / "vulkan-sdk"
LLAMACPP_SRC = INSTALL / "llama.cpp"
LLAMACPP_BUILD = INSTALL / "llama.cpp-build"
LLAMACPP_DIST = INSTALL / "llama-vulkan"

# Pinned to match local reference at C:\local\llama.cpp (683c5ac).
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMACPP_REF = "683c5acb90478a9e7e20eb65a1bfee334635216d"

# Pinned to match local SDK at C:\VulkanSDK\1.4.341.1.
VULKAN_VERSION = "1.4.341.1"
VULKAN_INSTALLER_NAME = f"vulkansdk-windows-X64-{VULKAN_VERSION}.exe"
# ?Human=true disables LunarG's download-token throttling for direct fetches.
VULKAN_INSTALLER_URL = (
    f"https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/windows/"
    f"{VULKAN_INSTALLER_NAME}?Human=true"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg):
    print(f"[vulkan-build] {msg}")


def download(url, dest):
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

    # LunarG's CDN returns 404/403 to the default Python-urllib User-Agent.
    # Install a browser UA on the global opener for the duration of this call.
    prev_opener = urllib.request._opener
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_progress)
        print()
        tmp.rename(dest)
    except Exception as exc:
        print()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {url}") from exc
    finally:
        urllib.request.install_opener(prev_opener)


# ---------------------------------------------------------------------------
# Vulkan SDK (silent install)
# ---------------------------------------------------------------------------


def fetch_vulkan_sdk():
    log("Setting up Vulkan SDK ...")
    sentinel = VULKAN_SDK / ".ok"
    if sentinel.exists():
        log("  Already installed.")
        return

    installer = CACHE / VULKAN_INSTALLER_NAME
    download(VULKAN_INSTALLER_URL, installer)

    if VULKAN_SDK.exists():
        shutil.rmtree(VULKAN_SDK)
    VULKAN_SDK.mkdir(parents=True, exist_ok=True)

    # The Lunarg installer is Qt Installer Framework based; --root makes it
    # install user-writable (no admin/UAC), and the silent flags suppress the
    # GUI. Default component set is enough for llama.cpp (Headers, Loader,
    # glslc, validation layers).
    log(f"  Running installer (silent) -> {VULKAN_SDK}")
    cmd = [
        str(installer),
        "--root",
        str(VULKAN_SDK),
        "--accept-licenses",
        "--default-answer",
        "--confirm-command",
        "install",
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(
            f"Vulkan SDK installer exited with code {r.returncode}. "
            "If a UAC prompt was dismissed, re-run from an elevated shell."
        )

    # Sanity check key files we need for building llama.cpp.
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


# ---------------------------------------------------------------------------
# llama.cpp source
# ---------------------------------------------------------------------------


def fetch_llamacpp():
    log("Setting up llama.cpp source ...")
    sentinel = LLAMACPP_SRC / ".ok"
    if sentinel.exists():
        # Verify the pinned commit is checked out — if the user moved HEAD
        # manually we don't want to silently build the wrong thing.
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


# ---------------------------------------------------------------------------
# MSVC environment (mirrors build.py)
# ---------------------------------------------------------------------------


def _ensure_msvc_env():
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
    if shutil.which("sccache"):
        subprocess.run(["sccache", "--stop-server"], capture_output=True)
    log("  MSVC environment ready.")


# ---------------------------------------------------------------------------
# Configure + build
# ---------------------------------------------------------------------------


def configure_and_build():
    log("Building llama.cpp (Vulkan) ...")

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
    log(f"  Done. Installed to: {LLAMACPP_DIST}")


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def clean():
    for path in (VULKAN_SDK, LLAMACPP_SRC, LLAMACPP_BUILD, LLAMACPP_DIST):
        if path.exists():
            log(f"  Removing {path} ...")
            shutil.rmtree(path, ignore_errors=True)
    # Cached installer/archives are kept so re-runs don't re-download GBs.


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Only fetch the Vulkan SDK and llama.cpp source — skip compile.",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Remove install/vulkan-sdk and install/llama.cpp* and exit.",
    )
    args = ap.parse_args()

    if args.clean:
        clean()
        return

    INSTALL.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    fetch_vulkan_sdk()
    fetch_llamacpp()

    if args.skip_build:
        log("Skip-build requested — done.")
        return

    configure_and_build()

    log("")
    log("Setup + build complete!")
    log(f"  Vulkan SDK:    {VULKAN_SDK}")
    log(f"  llama.cpp src: {LLAMACPP_SRC}  (commit {LLAMACPP_REF[:8]})")
    log(f"  Build output:  {LLAMACPP_BUILD}")
    log(f"  Install dir:   {LLAMACPP_DIST}")
    log("")
    log("Try it:")
    log(f"  set PATH={VULKAN_SDK / 'Bin'};%PATH%")
    log(f"  {LLAMACPP_DIST / 'bin' / 'llama-bench.exe'} --help")


if __name__ == "__main__":
    main()
