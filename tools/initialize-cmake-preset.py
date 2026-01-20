#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import pathlib
import json
import os

THIS_DIR = pathlib.Path(__file__).parent.parent.resolve()
DEFAULT_BUILD_DIR = (
    THIS_DIR / ".." / ".." / "build" / os.environ.get("BUILD_TYPE", "debug")
).resolve()
DEFAULT_PREFIX_DIR = (
    THIS_DIR / ".." / ".." / "local" / os.environ.get("BUILD_TYPE", "debug")
).resolve()
VAI_RT_BUILD_DIR = pathlib.Path(
    os.environ.get("VAI_RT_BUILD_DIR", DEFAULT_BUILD_DIR.as_posix())
)
VAI_RT_PREFIX = pathlib.Path(
    os.environ.get("VAI_RT_PREFIX", DEFAULT_PREFIX_DIR.as_posix())
)

config = {
    "version": 6,  # cmake v3.29 only support up to version 6
    "cmakeMinimumRequired": {"major": 3, "minor": 23, "patch": 0},
    "configurePresets": [
        {
            "name": "morphizen-mlir debug",
            "description": "Default configuration for morphizen mlir",
            "generator": "Visual Studio 17 2022",
            "binaryDir": (VAI_RT_BUILD_DIR / "morphizen-mlir").as_posix(),
            "cacheVariables": {
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
                "CMAKE_MSVC_RUNTIME_LIBRARY": "MultiThreaded$<$<CONFIG:Debug>:Debug>",
                "Protobuf_DEBUG": "ON",
                "BUILD_SHARED_LIBS": "OFF",
                "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
                "CMAKE_CONFIGURATION_TYPES": "Debug;Release;RelWithDebInfo",
                "CMAKE_INSTALL_PREFIX": VAI_RT_PREFIX.as_posix(),
                "CMAKE_PREFIX_PATH": VAI_RT_PREFIX.as_posix(),
            },
        }
    ],
}
json.dump(config, open(THIS_DIR / "CMakePresets.json", "w"), indent=4)
