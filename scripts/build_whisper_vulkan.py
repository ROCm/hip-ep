#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Standalone builder for the Whisper Vulkan baseline (whisper.cpp).

This is intentionally a SEPARATE script from build.py so the core build stays
lean — the Whisper Vulkan baseline is a benchmarking reference point, not part
of the MorphiZen EP build.

It is fully self-contained: the small Vulkan-SDK-fetch / download / MSVC-env
helpers it needs are inlined below (the LunarG download-URL gotcha + browser-UA
handling, pinned SDK version, repo-local install/_cache/). It used to import
these from build.py, but build.py was rewritten to a different layout/CLI that
no longer exposes them, so coupling here was removed.

What this does (idempotent, .ok sentinels per stage, repo-local install/_cache/):
  1. fetch_vulkan_sdk()         — inlined LunarG silent install
  2. fetch whisper.cpp source   — pinned to release tag v1.8.5
  3. build with GGML_VULKAN=ON  — same ggml-vulkan backend as the llama baseline
  4. download Whisper-large-v3 GGUF (f16) from HF ggerganov/whisper.cpp

Run:
    conda activate hip-ep
    python scripts/build_whisper_vulkan.py            # build + download model
    python scripts/build_whisper_vulkan.py --run      # build + run jfk.wav bench

Outputs:
    install/vulkan-sdk/          — LunarG SDK (repo-local)
    install/whisper.cpp/         — pinned whisper.cpp source (tag v1.8.5)
    C:\\wcpp_vk_build\\           — Ninja build dir (SHORT path; see WHISPER_BUILD;
                                   override with the WHISPER_VK_BUILD env var)
    install/whisper-vulkan/bin/  — whisper-cli.exe, whisper-bench.exe, ggml-vulkan.dll
    install/whisper-vulkan/models/ggml-large-v3.bin  — f16 GGUF (~3.1 GB)
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Self-contained build helpers.
#
# This script used to import these from build.py, but build.py was rewritten to
# a different layout/CLI and no longer exposes them. Rather than couple this
# optional Vulkan-baseline script to build.py's internals, the small pieces it
# needs (a logger, a downloader, the Vulkan SDK fetch, MSVC env detection) are
# inlined here. Outputs still live under the repo-local `install/` tree, shared
# with any cached Vulkan SDK download.
# ---------------------------------------------------------------------------

INSTALL = REPO_ROOT / "install"
CACHE = INSTALL / "_cache"
VULKAN_SDK = INSTALL / "vulkan-sdk"

# Pinned Vulkan SDK version for reproducible Vulkan baselines.
VULKAN_VERSION = "1.4.341.1"
VULKAN_INSTALLER_NAME = f"vulkansdk-windows-X64-{VULKAN_VERSION}.exe"
VULKAN_INSTALLER_URL = (
    f"https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/windows/"
    f"{VULKAN_INSTALLER_NAME}?Human=true"
)


def log(msg):
    print(f"[whisper-vulkan] {msg}")


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
    # set covers what whisper.cpp needs (Headers, Loader, glslc, validation).
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
# Pinned versions for reproducibility
# ---------------------------------------------------------------------------

WHISPER_REPO = "https://github.com/ggml-org/whisper.cpp.git"
# Pin to the v1.8.5 release tag (commit 832416f) — the latest stable release as
# of 2026-05. whisper.cpp uses the SAME ggml-vulkan backend as llama.cpp, so the
# Vulkan SDK / cmake recipe mirrors build.py::build_vulkan().
WHISPER_REF = "v1.8.5"
# The COMMIT the v1.8.5 tag points to (NOT the annotated-tag-object SHA, which
# git ls-remote --tags reports as 832416f4 — that is the tag wrapper, not the
# commit). `git rev-parse HEAD` after checkout yields the commit, so the
# idempotency sentinel check must compare against the commit SHA.
WHISPER_REF_COMMIT = "f24588a272ae8e23280d9c220536437164e6ed28"

WHISPER_SRC = INSTALL / "whisper.cpp"
WHISPER_DIST = INSTALL / "whisper-vulkan"
WHISPER_MODELS = WHISPER_DIST / "models"

# CRITICAL — keep the BUILD dir on a SHORT path.
# whisper.cpp builds vulkan-shaders-gen as a nested ExternalProject, whose
# compiler-detection try-compiles write a `vc140.pdb` ~5 levels deeper than the
# build root. Under the normal `install/whisper.cpp-build/...` location inside a
# `.claude/worktrees/...` checkout the PDB path hits 274 chars > MSVC's MAX_PATH
# (260) -> "fatal error C1041: cannot open program database". /Z7, /FS, and a
# toolchain file do NOT help because the failure is in cmake's own compiler
# detection before our flags apply and the subproject doesn't inherit them.
# The only robust fix is a short build root. Overridable via env if the default
# collides on a given machine.
WHISPER_BUILD = Path(os.environ.get("WHISPER_VK_BUILD", r"C:\wcpp_vk_build"))

# Whisper-large-v3 GGUF (f16) from the canonical whisper.cpp HF repo. whisper.cpp
# ships f16 GGUF as its standard distribution — that is the precision to label in
# any cross-backend table (it is NOT the fp32 ONNX our EP/CPU run).
GGUF_NAME = "ggml-large-v3.bin"
GGUF_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{GGUF_NAME}"

JFK_WAV = REPO_ROOT / "test" / "python" / "data" / "whisper" / "jfk.wav"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def fetch_whispercpp():
    log("Setting up whisper.cpp source ...")
    sentinel = WHISPER_SRC / ".ok"
    if sentinel.exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WHISPER_SRC),
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head == WHISPER_REF_COMMIT:
            log(f"  Already at pinned {WHISPER_REF} ({WHISPER_REF_COMMIT[:8]}).")
            return
        log(f"  HEAD is {head[:8]}, expected {WHISPER_REF_COMMIT[:8]} — re-checkout.")

    if not WHISPER_SRC.exists():
        WHISPER_SRC.parent.mkdir(parents=True, exist_ok=True)
        log(f"  Cloning {WHISPER_REPO} ...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", WHISPER_REPO, str(WHISPER_SRC)],
            check=True,
        )

    log(f"  Checking out pinned tag {WHISPER_REF} ({WHISPER_REF_COMMIT[:8]}) ...")
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", "tag", WHISPER_REF],
        cwd=str(WHISPER_SRC),
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", WHISPER_REF_COMMIT],
        cwd=str(WHISPER_SRC),
        check=True,
    )

    sentinel.touch()
    log("  whisper.cpp ready.")


def build_whispercpp():
    """Build whisper.cpp with GGML_VULKAN=ON — same backend / cmake recipe as
    build.py::build_vulkan() for llama.cpp."""
    log("Building whisper.cpp (Vulkan) ...")
    sentinel = WHISPER_DIST / ".build.ok"
    if sentinel.exists():
        log("  Already built.")
        return

    fetch_vulkan_sdk()
    fetch_whispercpp()

    _ensure_msvc_env()
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            log(f"  ERROR: {tool} not found. Run: conda activate hip-ep")
            sys.exit(1)

    # Make the SDK visible to CMake's FindVulkan via the standard env var.
    os.environ["VULKAN_SDK"] = str(VULKAN_SDK)
    os.environ["PATH"] = str(VULKAN_SDK / "Bin") + os.pathsep + os.environ["PATH"]

    # The PRIMARY fix for the vc140.pdb C1041 is the short WHISPER_BUILD path
    # (see the WHISPER_BUILD comment above) — the failure is a MAX_PATH overflow
    # on the PDB path inside the nested vulkan-shaders-gen ExternalProject, NOT
    # concurrency. The toolchain below is secondary hygiene: /Z7 embeds debug
    # info in the .obj (no separate PDB at all) and /FS serialises any remaining
    # PDB writes. ExternalProject_Add propagates CMAKE_TOOLCHAIN_FILE to child
    # projects and a toolchain's *_FLAGS_INIT apply during compiler detection,
    # so this reaches the shaders-gen subproject's try-compiles too.
    WHISPER_BUILD.mkdir(parents=True, exist_ok=True)
    toolchain = WHISPER_BUILD / "msvc_pdb_fix_toolchain.cmake"
    toolchain.write_text(
        'set(CMAKE_C_FLAGS_INIT "/FS /Z7")\nset(CMAKE_CXX_FLAGS_INIT "/FS /Z7")\n'
    )

    cmake_args = [
        "cmake",
        "-S",
        str(WHISPER_SRC),
        "-B",
        str(WHISPER_BUILD),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={WHISPER_DIST}",
        "-DGGML_VULKAN=ON",
        "-DGGML_NATIVE=ON",
        "-DGGML_CUDA=OFF",
        "-DGGML_HIP=OFF",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=ON",  # need whisper-cli + whisper-bench
        f"-DVulkan_INCLUDE_DIR={VULKAN_SDK / 'Include'}",
        f"-DVulkan_LIBRARY={VULKAN_SDK / 'Lib' / 'vulkan-1.lib'}",
        f"-DVulkan_GLSLC_EXECUTABLE={VULKAN_SDK / 'Bin' / 'glslc.exe'}",
        # See the msvc_pdb_fix_toolchain note above — forces /FS /Z7 into every
        # try-compile and the nested vulkan-shaders-gen subproject to avoid the
        # C1041 vc140.pdb contention.
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
    ]

    # NOTE: intentionally NOT enabling the sccache compiler launcher here.
    # whisper.cpp builds vulkan-shaders-gen as a nested ExternalProject that
    # forwards only its own CMAKE_ARGS (not our CMAKE_C_FLAGS / launcher), so
    # sccache + MSVC's default /Zi triggers the vc140.pdb contention
    # (C1041) inside that subproject's compiler-ABI try-compile. The build is
    # small and one-shot, so skipping sccache is the robust choice.

    log("  Configuring ...")
    subprocess.run(cmake_args, check=True)

    log("  Compiling ...")
    subprocess.run(
        ["cmake", "--build", str(WHISPER_BUILD), "--config", "Release"],
        check=True,
    )

    log("  Installing ...")
    subprocess.run(
        ["cmake", "--install", str(WHISPER_BUILD), "--config", "Release"],
        check=True,
    )

    # cmake --install lays binaries under bin/; the DLL may land next to the
    # examples in the build tree on some generators. Make sure the runtime DLLs
    # sit beside the exes so whisper-cli launches without PATH fiddling.
    bindir = WHISPER_DIST / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for dll in WHISPER_BUILD.rglob("*.dll"):
        dest = bindir / dll.name
        if not dest.exists():
            shutil.copy2(dll, dest)
    for exe in WHISPER_BUILD.rglob("whisper-*.exe"):
        dest = bindir / exe.name
        if not dest.exists():
            shutil.copy2(exe, dest)

    sentinel.touch()
    log(f"  whisper.cpp Vulkan build ready -> {WHISPER_DIST}")


def fetch_gguf():
    log("Setting up Whisper-large-v3 GGUF (f16) ...")
    WHISPER_MODELS.mkdir(parents=True, exist_ok=True)
    dest = WHISPER_MODELS / GGUF_NAME
    if dest.exists():
        log(f"  Cached: {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    # Cache the download under the shared install/_cache/ then hardlink/copy in.
    cached = CACHE / GGUF_NAME
    # HF resolve URLs are happy with the default UA, but reuse the browser-UA
    # helper for robustness against any CDN UA filtering.
    _download_with_browser_ua(GGUF_URL, cached)
    shutil.copy2(cached, dest)
    log(f"  GGUF ready -> {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def run_bench():
    """Run whisper-cli on jfk.wav: prints transcription + encode/decode timing,
    and the Vulkan device init line (GPU confirmation)."""
    cli = WHISPER_DIST / "bin" / "whisper-cli.exe"
    if not cli.exists():
        # whisper.cpp renamed `main` -> `whisper-cli`; fall back just in case.
        alt = WHISPER_DIST / "bin" / "main.exe"
        cli = alt if alt.exists() else cli
    model = WHISPER_MODELS / GGUF_NAME
    if not cli.exists():
        log(f"  ERROR: {cli} not found — build first.")
        sys.exit(1)
    if not model.exists():
        log(f"  ERROR: {model} not found — download GGUF first.")
        sys.exit(1)

    log(f"  Running: {cli.name} -m {model.name} -f {JFK_WAV.name}")
    # whisper.cpp prints encode/decode ms in its timing footer; capture all of it.
    cmd = [
        str(cli),
        "-m",
        str(model),
        "-f",
        str(JFK_WAV),
        "-l",
        "en",
        "-bs",
        "1",  # greedy beam=1 to mirror our EP greedy decode
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("===== whisper-cli STDOUT =====")
    print(r.stdout)
    print("===== whisper-cli STDERR =====")
    print(r.stderr)
    if r.returncode != 0:
        log(f"  whisper-cli exited {r.returncode}")
        sys.exit(r.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Build + run the Whisper Vulkan baseline (whisper.cpp)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="After building + downloading, run jfk.wav through whisper-cli",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip the ~3 GB GGUF download (build only)",
    )
    args = parser.parse_args()

    build_whispercpp()
    if not args.skip_model:
        fetch_gguf()

    if args.run:
        run_bench()

    log("")
    log("Whisper Vulkan baseline ready:")
    log(f"  Vulkan SDK:       {VULKAN_SDK}")
    log(f"  whisper.cpp src:  {WHISPER_SRC} ({WHISPER_REF} / {WHISPER_REF_COMMIT[:8]})")
    log(f"  Binaries:         {WHISPER_DIST / 'bin'}")
    log(f"  GGUF model:       {WHISPER_MODELS / GGUF_NAME}")


if __name__ == "__main__":
    main()
