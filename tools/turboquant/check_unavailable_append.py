#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check compiled append exports and invalid-input rejection without a GPU."""

import argparse
import ctypes
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--dll-dir", action="append", default=[], type=Path)
    parser.add_argument(
        "--available",
        action="store_true",
        help="Expect Ring append to be compiled for this slice",
    )
    args = parser.parse_args()
    # Keep handles alive until the library has been exercised.
    dll_dirs = (
        [os.add_dll_directory(str(p.resolve())) for p in args.dll_dir]
        if os.name == "nt"
        else []
    )
    # CDLL loads with RTLD_NOW on POSIX: unresolved Ring exports fail here too.
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.hip_tq4_prefill_workspace.argtypes = [ctypes.c_int]
    lib.hip_tq4_prefill_workspace.restype = ctypes.c_size_t
    lib.hip_tq4_prefill.argtypes = (
        [ctypes.c_void_p] * 7
        + [ctypes.c_int] * 3
        + [ctypes.c_void_p] * 4
        + [ctypes.c_size_t]
    )
    lib.hip_tq4_prefill.restype = ctypes.c_int
    for count in (0, 1, 128, 262144, 262145):
        size = lib.hip_tq4_prefill_workspace(count)
        rows = min(count, 128) * 24
        expected = (
            rows * (256 * 2 + (2 * 256 + 2) * 4)
            if args.available and 1 <= count <= 262144
            else 0
        )
        if size != expected:
            raise AssertionError(f"Append workspace {size}, expected {expected}")
    status = lib.hip_tq4_prefill(*([None] * 7), 2, 1, 3, *([None] * 4), 0)
    expected_status = -1 if args.available else 801
    if status != expected_status:  # hipErrorNotSupported
        raise AssertionError(f"Expected rejection {expected_status}, got {status}")
    for directory in dll_dirs:
        directory.close()
    print("PASS: library loads and append admission rejects without GPU access")


if __name__ == "__main__":
    main()
