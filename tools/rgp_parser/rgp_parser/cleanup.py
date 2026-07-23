#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""cleanup.py: stop the RGP capture 'listeners'.

The RGP capture path leaves the Radeon Developer Service / Panel running (the
'listeners' that talk to the driver's DevDriver router). This kills them. It does
not touch the environment/PATH and does not reboot.
"""

from __future__ import annotations

import subprocess
import sys

PROCESSES = [
    "RadeonDeveloperPanelCLI.exe",
    "RadeonDeveloperPanel.exe",
    "RadeonDeveloperServiceCLI.exe",
    "RadeonDeveloperService.exe",
]


def main(argv=None) -> int:
    if sys.platform != "win32":
        print("nothing to do: RGP listeners run only on Windows.", file=sys.stderr)
        return 0
    for name in PROCESSES:
        r = subprocess.run(
            ["taskkill", "/F", "/IM", name], capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"killed {name}")
    return 0
