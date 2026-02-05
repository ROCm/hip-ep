#!/usr/bin/env python3
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

"""
Generate CMakePresets.json for onnx-hipdnn-ep project.
"""

import os
import json

def main():
    # Get paths from environment or use defaults
    build_dir = os.environ.get('VAI_RT_BUILD_DIR', 'build')
    prefix = os.environ.get('VAI_RT_PREFIX', 'local')
    
    presets = {
        "version": 3,
        "cmakeMinimumRequired": {
            "major": 3,
            "minor": 29,
            "patch": 0
        },
        "configurePresets": [
            {
                "name": "default",
                "displayName": "Default Config",
                "description": "Default build using Ninja generator",
                "generator": "Ninja",
                "binaryDir": "${sourceDir}/build",
                "cacheVariables": {
                    "CMAKE_BUILD_TYPE": "Debug",
                    "CMAKE_INSTALL_PREFIX": prefix,
                    "CMAKE_PREFIX_PATH": prefix,
                    "BUILD_SHARED_LIBS": "OFF"
                }
            }
        ],
        "buildPresets": [
            {
                "name": "default",
                "configurePreset": "default"
            }
        ]
    }
    
    # Write presets file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    presets_file = os.path.join(project_root, "CMakePresets.json")
    
    with open(presets_file, 'w') as f:
        json.dump(presets, f, indent=2)
    
    print(f"Generated {presets_file}")

if __name__ == "__main__":
    main()
