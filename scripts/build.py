#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Native cross-platform build driver for onnx-hipdnn-ep.

Structured after ONNX Runtime's tools/ci_build/build.py (logging, run_subprocess,
update_submodules, --config/--cmake_generator/--cmake_extra_defines/
--skip_submodule_sync conventions) but deliberately plain: it ensures
submodules, sets up the platform compiler environment, picks the GPU arch, runs
the cmake configure/build/install for this project's artifacts -- the MorphiZen
EP shared library (libonnxruntime_morphizen_ep.so / .dll) plus the HIP MLIR
tools -- and (on Linux) bundles the runtime .so into a self-contained install/.

All C++ dependency *acquisition* is delegated to cmake/deps.cmake: LLVM/MLIR/LLD
(from-source fallback), protobuf/flatbuffers (from source), ONNX Runtime
(official release download), and the TheRock ROCm SDK (auto-download). This
script does NOT build or cache those itself -- caching and any
build-into-a-prefix optimization are a CI concern (see
.github/workflows/linux-build.yml), and CI injects its cached prefixes via
--cmake_prefix_path. It also does NOT build onnxruntime-genai (OGA).

The legacy Windows-only root build.py is left untouched.

Layout (siblings of the repo, matching docs/quick_start.md and CI):
    <workspace>/<repo>/          project source (this repo)
    <workspace>/build/<repo>/    cmake build tree (+ _therock, _deps)
    <workspace>/install/         install prefix (self-contained on Linux)

Usage:
    python scripts/build.py                       # real build, auto-detect arch
    python scripts/build.py --mock                # mock runtime (no GPU/HIP/TheRock)
    python scripts/build.py --hip_arch gfx1151    # explicit GPU arch
    python scripts/build.py --cmake_prefix_path "/x/llvm-install;/x/ort-install"
    python scripts/build.py --clean               # remove build/ + install/
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent
PROJECT_NAME = REPO.name
IS_WINDOWS = os.name == "nt"

logging.basicConfig(format="[build] %(message)s", level=logging.INFO)
log = logging.getLogger("hipdnn-ep.build")


# ---------------------------------------------------------------------------
# Errors + subprocess helper (ONNX Runtime style)
# ---------------------------------------------------------------------------


class BuildError(Exception):
    pass


def run_subprocess(args, cwd=None, env=None, capture_stdout=False):
    """Run a command (sequence of str/Path), echoing it. Raises on failure."""
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")
    args = [str(a) for a in args]
    my_env = os.environ.copy()
    if env:
        my_env.update(env)
    log.info(" ".join(shlex.quote(a) for a in args))
    return subprocess.run(
        args, cwd=cwd, env=my_env, check=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        text=True if capture_stdout else None,
    )


def step(msg):
    bar = "=" * 68
    print(f"\n{bar}\n> {msg}\n{bar}", flush=True)


def have_tool(name):
    return shutil.which(name) is not None


def _check_python_version():
    if sys.version_info[:2] < (3, 8):
        raise BuildError(f"Python 3.8+ required; found {sys.version.split()[0]}")


# ---------------------------------------------------------------------------
# Submodules + platform toolchain
# ---------------------------------------------------------------------------


def update_submodules():
    log.info("Checking git submodules ...")
    r = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    uninitialized = any(
        ln.strip().startswith("-") for ln in r.stdout.splitlines() if ln.strip()
    )
    if not uninitialized:
        log.info("  submodules OK.")
        return
    log.info("  initializing submodules ...")
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=str(REPO))
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO))


def default_generator():
    # Like ONNX Runtime: on Windows default to the Visual Studio generator so
    # CMake locates MSVC itself (no vcvarsall sourcing); Ninja elsewhere.
    return "Visual Studio 17 2022" if IS_WINDOWS else "Ninja"


def check_toolchain(generator):
    for tool in ("cmake", "git"):
        if not have_tool(tool):
            raise BuildError(f"required tool not found on PATH: {tool}")
    if generator == "Ninja" and not have_tool("ninja"):
        raise BuildError("Ninja generator selected but 'ninja' is not on PATH")
    if IS_WINDOWS:
        # The Visual Studio generator finds MSVC on its own. For Ninja we rely on
        # the caller having loaded the MSVC environment (run from an "x64 Native
        # Tools Command Prompt for VS"), exactly as ONNX Runtime's build expects.
        if generator == "Ninja" and not have_tool("cl"):
            log.warning(
                "cl.exe not on PATH; for the Ninja generator on Windows run from "
                "an 'x64 Native Tools Command Prompt for VS' (or pass "
                "--cmake_generator 'Visual Studio 17 2022')."
            )
    elif not (have_tool("c++") or have_tool("g++") or have_tool("clang++")):
        raise BuildError("no C++ compiler (c++/g++/clang++) on PATH")


# ---------------------------------------------------------------------------
# GPU architecture detection (Linux: amdgpu kernel sysfs)
# ---------------------------------------------------------------------------


def detect_hip_arch():
    """Read gfx_target_version from /sys/class/kfd topology (no ROCm needed).

    gfx_target_version encodes the arch as major*10000 + minor*100 + step
    (e.g. 110501 -> gfx1151, 90010 -> gfx90a). The CPU node reports 0.
    Returns a gfx string or None.
    """
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    if not nodes.is_dir():
        return None
    for props in sorted(nodes.glob("*/properties")):
        try:
            text = props.read_text()
        except OSError:
            continue
        ver = None
        for ln in text.splitlines():
            if ln.startswith("gfx_target_version "):
                ver = int(ln.split()[1])
                break
        if not ver:
            continue
        major, minor, step = ver // 10000, (ver // 100) % 100, ver % 100
        return f"gfx{major}{minor:x}{step:x}"
    return None


# ---------------------------------------------------------------------------
# Configure / build / install (deps resolved by cmake/deps.cmake)
# ---------------------------------------------------------------------------


def generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock):
    step(f"Configure (HIP_ARCHITECTURES={hip_arch or 'mock'})")
    cmd = ["cmake", "-S", str(REPO), "-B", str(build_dir), "-G", args.cmake_generator]
    # The Visual Studio (multi-config) generator needs the target platform via -A
    # (Ninja/Makefiles take the build type via CMAKE_BUILD_TYPE only).
    if args.cmake_generator.startswith("Visual Studio"):
        cmd += ["-A", "x64"]
    cmd += [
        f"-DCMAKE_BUILD_TYPE={args.config}",
        f"-DCMAKE_INSTALL_PREFIX={args.install_dir}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DBUILD_EP=ON",
        "-DBUILD_HIP_TOOLS=ON",
    ]
    if prefix_paths:
        # CMAKE_PREFIX_PATH always uses ';' as the list separator.
        cmd.append("-DCMAKE_PREFIX_PATH=" + ";".join(prefix_paths))
    if mock:
        cmd.append("-DBUILD_MOCK_RUNTIME=ON")
    else:
        cmd.append("-DBUILD_MOCK_RUNTIME=OFF")
        cmd.append(f"-DHIP_ARCHITECTURES={hip_arch}")
        if args.therock_dist:
            cmd.append(f"-DTHEROCK_DIST={args.therock_dist}")
    if IS_WINDOWS:
        cmd.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded")
    if have_tool("sccache"):
        cmd += [
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ]
    cmd.append(f"-DPython3_EXECUTABLE={sys.executable}")
    # Arbitrary -D escape hatch (ORT-style --cmake_extra_defines KEY=VALUE).
    for define in args.cmake_extra_defines:
        cmd.append(f"-D{define}")
    run_subprocess(cmd)


def build_targets(args, build_dir):
    step("Build")
    run_subprocess([
        "cmake", "--build", str(build_dir),
        "--config", args.config, "--parallel", str(args.parallel),
    ])
    step("Install")
    run_subprocess(["cmake", "--install", str(build_dir), "--config", args.config])


def run_tests(args, build_dir):
    """Run the LIT suite (MLIR pass verification; no GPU needed)."""
    step("Test (check-hip-mlir-lit)")
    run_subprocess([
        "cmake", "--build", str(build_dir),
        "--config", args.config, "--target", "check-hip-mlir-lit",
    ])


# ---------------------------------------------------------------------------
# Arguments + main
# ---------------------------------------------------------------------------


def parse_arguments():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="Release",
                   choices=["Release", "RelWithDebInfo", "Debug"],
                   help="build configuration (default: Release)")
    p.add_argument("--build_dir", default=str(WORKSPACE / "build" / PROJECT_NAME))
    p.add_argument("--install_dir", default=str(WORKSPACE / "install"))
    p.add_argument("--cmake_generator", default=None,
                   help="cmake generator (default: 'Visual Studio 17 2022' on Windows, Ninja elsewhere)")
    p.add_argument("--cmake_extra_defines", action="append", default=[], metavar="KEY=VALUE",
                   help="extra -D<KEY=VALUE> forwarded to cmake configure (repeatable)")
    p.add_argument("--cmake_prefix_path", default="",
                   help="extra prefixes (';'-separated) forwarded to CMAKE_PREFIX_PATH, "
                        "e.g. CI-built/cached llvm-install;ort-install")
    p.add_argument("--therock_dist", default="",
                   help="path to a TheRock ROCm SDK (else cmake/deps.cmake auto-downloads)")
    p.add_argument("--hip_arch", default="",
                   help="GPU arch (e.g. gfx1151); auto-detected on Linux if unset")
    p.add_argument("--mock", action="store_true",
                   help="mock runtime (no GPU/HIP/TheRock)")
    p.add_argument("--skip_submodule_sync", action="store_true",
                   help="do not sync/update git submodules")
    p.add_argument("--clean", action="store_true",
                   help="remove build/ and install/ then exit")
    p.add_argument("--skip_build", action="store_true",
                   help="configure only; do not build/install")
    p.add_argument("--skip_tests", action="store_true",
                   help="do not run the LIT tests after install")
    p.add_argument("--allow_running_as_root", action="store_true")
    p.add_argument("-j", "--parallel", type=int, default=os.cpu_count() or 4)
    return p.parse_args()


def main():
    log.debug("argv: %s", " ".join(shlex.quote(a) for a in sys.argv[1:]))
    _check_python_version()
    args = parse_arguments()

    build_dir = Path(args.build_dir)
    install_dir = Path(args.install_dir)
    prefix_paths = [
        s for s in args.cmake_prefix_path.replace(os.pathsep, ";").split(";") if s
    ]

    if args.clean:
        for d in (build_dir, install_dir):
            if d.exists():
                log.info(f"removing {d}")
                shutil.rmtree(d)
        log.info("clean complete.")
        return

    if not IS_WINDOWS and os.geteuid() == 0 and not args.allow_running_as_root:
        raise BuildError(
            "Running as root is not allowed. Pass --allow_running_as_root to override."
        )

    log.info(f"REPO={REPO}")
    log.info(f"WORKSPACE={WORKSPACE}")
    log.info(f"platform={'windows' if IS_WINDOWS else 'linux'}  jobs={args.parallel}")

    args.cmake_generator = args.cmake_generator or default_generator()
    log.info(f"cmake generator: {args.cmake_generator}")

    if not args.skip_submodule_sync:
        update_submodules()
    check_toolchain(args.cmake_generator)

    mock = args.mock
    hip_arch = args.hip_arch.strip()
    if not mock and not hip_arch:
        hip_arch = detect_hip_arch() if not IS_WINDOWS else ""
        if hip_arch:
            log.info(f"auto-detected HIP_ARCHITECTURES={hip_arch}")
        elif IS_WINDOWS:
            log.info("HIP_ARCHITECTURES unset; cmake/deps.cmake will auto-detect on Windows")
        else:
            log.info("no AMD GPU detected and no --hip_arch; falling back to --mock")
            mock = True

    generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock)
    if args.skip_build:
        log.info("configure done; skipping build (--skip_build).")
        return
    build_targets(args, build_dir)

    if not args.skip_tests:
        run_tests(args, build_dir)

    step("DONE")
    log.info(f"install tree: {install_dir}")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        log.error(str(exc))
        sys.exit(1)
