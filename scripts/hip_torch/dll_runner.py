#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
DLL runner: ctypes-based in-process GPU DLL execution.

Re-exports from the existing hip_dll_runner module for backward
compatibility while providing the canonical import path.
"""

import sys
from pathlib import Path

# Add parent scripts/ dir so we can import the existing module
_scripts_dir = str(Path(__file__).parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from hip_dll_runner import HipDllRunner  # noqa: E402, F401
