#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Load a no-Ring kernel library and check append rejection without a GPU."""

import argparse
import ctypes
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--dll-dir", action="append", default=[], type=Path)
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
    for count in (0, 1, 128, 131072, 131073):
        size = lib.hip_tq4_prefill_workspace(count)
        if size != 0:
            raise AssertionError(f"Unavailable append returned workspace {size}")
    status = lib.hip_tq4_prefill(*([None] * 7), 2, 1, 3, *([None] * 4), 0)
    if status != 801:  # hipErrorNotSupported
        raise AssertionError(f"Expected hipErrorNotSupported, got {status}")
    for directory in dll_dirs:
        directory.close()
    print("PASS: library loads and unavailable append rejects without GPU access")


if __name__ == "__main__":
    main()
