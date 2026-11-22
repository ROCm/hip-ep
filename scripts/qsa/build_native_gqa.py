#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Build the current checkout in an isolated native-QSA installation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_baseline import ROOT, WORK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toolchain-root", type=Path, default=ROOT.parent / "hip-ep-turboquant"
    )
    parser.add_argument(
        "--ort-sdk",
        type=Path,
        default=ROOT
        / "_local/ring-1259/deps/onnxruntime-sdk/onnxruntime-win-x64-1.31.0",
    )
    parser.add_argument(
        "--lit", type=Path, default=ROOT / "_local/ring-1259/venv/Scripts/lit.exe"
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=WORK,
        help="Parent of the isolated native-gqa build/install directories",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip build.py tests when refreshing an already-tested installation",
    )
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--cmake-define", action="append", default=[])
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the local dependency recipe without building",
    )
    args = parser.parse_args()
    assert os.name == "nt", (
        "This helper reuses the Windows host toolchain; use build.py directly elsewhere"
    )
    folder = args.work_root.resolve() / "native-gqa"
    folder.mkdir(parents=True, exist_ok=True)
    old = args.toolchain_root.resolve()
    sdk = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
    command = [
        sys.executable,
        str(ROOT / "build.py"),
        "--hip_arch",
        "gfx1151",
        "--config",
        "Release",
        "--parallel",
        str(args.parallel),
        "--skip_wheel",
        "--skip_submodule_sync",
        "--cmake_generator",
        "Ninja",
        "--build_dir",
        str(folder / "build"),
        "--install_dir",
        str(folder / "install"),
        "--therock_dist",
        str(sdk),
    ]
    if args.skip_tests:
        command.append("--skip_tests")
    if args.configure_only:
        command.append("--skip_build")
    definitions = {
        "CMAKE_C_COMPILER": "cl",
        "CMAKE_CXX_COMPILER": "cl",
        "BUILD_PYTHON_WHEEL": "OFF",
        "ENABLE_ROCMLIRTRITON": "OFF",
        "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
        "LIT_EXECUTABLE": args.lit.as_posix(),
        "MLIR_DIR": (old / "build/full-msvc/lib/cmake/mlir").as_posix(),
        "LLVM_DIR": (
            old / "build/full-msvc/_deps/llvm-project-build/lib/cmake/llvm"
        ).as_posix(),
        "LLD_DIR": (old / "build/full-msvc/lib/cmake/lld").as_posix(),
    }
    for name, path in {
        "ONNXRUNTIME": args.ort_sdk,
        "PROTOBUF": old / "deps/protobuf-v34",
        "FLATBUFFERS": old / "deps/flatbuffers-v25",
        "ABSL": old / "deps/abseil-20250512",
        "CPPTRACE": old / "build/full-msvc/_deps/cpptrace-src",
        "GLOG": old / "build/full-msvc/_deps/glog-src",
        "MICROSOFT.GSL": old / "build/full-msvc/_deps/microsoft.gsl-src",
        "HASH-LIBRARY": old / "build/full-msvc/_deps/hash-library-src",
    }.items():
        assert path.exists(), path
        definitions[f"FETCHCONTENT_SOURCE_DIR_{name}"] = path.resolve().as_posix()
    assert args.lit.exists(), args.lit
    for key, value in definitions.items():
        command += ["--cmake_extra_defines", key + "=" + value]
    for definition in args.cmake_define:
        command += ["--cmake_extra_defines", definition]
    if args.plan:
        print(json.dumps(command, indent=2))
        return
    env = dict(os.environ)
    vs = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools")
    tools = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake"
    llvm = old / "build/full-msvc/_deps/llvm-project-build"
    env["PATH"] = (
        os.pathsep.join(
            map(
                str,
                [
                    ROOT / "_local/ring-1259/venv/Scripts",
                    old / "venv/Scripts",
                    tools / "CMake/bin",
                    tools / "Ninja",
                    llvm / "bin",
                    sdk / "bin",
                ],
            )
        )
        + os.pathsep
        + env["PATH"]
    )
    env["THEROCK_DIST"] = str(sdk)
    (folder / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    batch = folder / "build.cmd"
    batch.write_text(
        '@echo off\ncall "' + str(vs / "VC/Auxiliary/Build/vcvars64.bat") + '" >nul\n'
        "if errorlevel 1 exit /b %errorlevel%\n"
        + 'set "INCLUDE='
        + str(old / "deps/rocmlirtriton-resumable/external/llvm-project/lld/include")
        + ';%INCLUDE%"\n'
        + subprocess.list2cmdline(command)
        + "\nexit /b %errorlevel%\n"
    )
    subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
