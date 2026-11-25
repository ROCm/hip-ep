#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Build the staged QSA C ABI as a separate gfx1151 library (CPU-only build)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_baseline import ROOT, WORK, sha


def main():
    sdk = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
    core = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
    vs = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools")
    cmake = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    ninja = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
    source = ROOT / "lib/Runtime/Kernels/qsa"
    build = WORK / "native-build"
    install = WORK / "native"
    build.mkdir(exist_ok=True)
    install.mkdir(exist_ok=True)
    commands = [
        [str(sdk / "bin/hipcc.exe"), "--version"],
        [
            str(cmake),
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=cl",
            "-DCMAKE_CXX_COMPILER=cl",
            "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
            "-DHIP_ARCHITECTURES=gfx1151",
            "-DHIP_PATH=" + str(sdk),
            "-DTHEROCK_DIST=" + str(sdk),
        ],
        [str(cmake), "--build", str(build), "--verbose"],
    ]
    batch = WORK / "build-qsa.cmd"
    batch.write_text(
        '@echo off\ncall "' + str(vs / "VC/Auxiliary/Build/vcvars64.bat") + '" >nul\n'
        "if errorlevel 1 exit /b %errorlevel%\n"
        'set "PATH='
        + str(sdk / "bin")
        + ";"
        + str(core / "bin")
        + ';%PATH%"\n'
        + "\n".join(
            subprocess.list2cmdline(c) + "\nif errorlevel 1 exit /b %errorlevel%"
            for c in commands
        )
        + "\n"
    )
    env = dict(os.environ)
    env["THEROCK_DIST"] = str(sdk)
    subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=ROOT, env=env, check=True)
    library = install / "qsa_kernels_gfx1151.dll"
    shutil.copyfile(build / library.name, library)
    files = [
        ROOT / "lib/Runtime/Kernels/hip/qsa/qsa_port.hip",
        ROOT / "lib/Runtime/Kernels/include/hip_qsa.h",
        source / "CMakeLists.txt",
        ROOT / "lib/Runtime/Kernels/cmake/hip_utils.cmake",
    ]
    receipt = {
        "architecture": "gfx1151",
        "wave_size": 32,
        "cooperative_launch_required": False,
        "commands": commands,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "library": str(library),
        "library_sha256": sha(library),
        "training_updates": 0,
    }
    (install / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
