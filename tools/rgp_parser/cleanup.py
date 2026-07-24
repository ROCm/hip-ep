#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Entry point: stop the RGP capture listeners.

python cleanup.py   # kill Radeon Developer Service/Panel
"""

from rgp_parser.cleanup import main

if __name__ == "__main__":
    raise SystemExit(main())
