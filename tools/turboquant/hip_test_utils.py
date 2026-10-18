#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Direct HIP correctness tests with guarded allocations and an FP64 oracle."""

from pathlib import Path
import ctypes as c
import sys
import os
import numpy as np

# Preserve explicit exports used by the earlier standalone diagnostic scripts.
import json as json
from reference import (
    attention as attention,
    encode as encode,
    unpack_rows as unpack_rows,
)

ROOT = Path(os.environ.get("TQ_WORK_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
hip = c.CDLL(os.environ.get("TQ_HIP_RUNTIME", "libamdhip64.so"))
lib = c.CDLL(os.environ.get("TQ_TEST_LIBRARY", str(ROOT / "build/libtq4-selected.so")))
ptr = c.c_void_p
hip.hipMalloc.argtypes = [c.POINTER(ptr), c.c_size_t]
hip.hipFree.argtypes = [ptr]
hip.hipMemcpy.argtypes = [ptr, ptr, c.c_size_t, c.c_int]
hip.hipStreamCreate.argtypes = [c.POINTER(ptr)]
hip.hipStreamSynchronize.argtypes = [ptr]
hip.hipStreamDestroy.argtypes = [ptr]
lib.hip_tq4_encode_append.argtypes = [ptr] * 5 + [c.c_int] * 3 + [ptr] * 3
lib.hip_tq4_decode.argtypes = (
    [ptr] * 5 + [c.c_int] * 2 + [c.c_float] + [ptr] * 4 + [c.c_size_t]
)
lib.hip_tq4_decode_workspace.argtypes = [c.c_int]
lib.hip_tq4_decode_workspace.restype = c.c_size_t


def check(rc):
    assert rc == 0, rc


class Buffer:
    def __init__(self, bytes):
        self.size = bytes
        self.base = ptr()
        check(hip.hipMalloc(c.byref(self.base), bytes + 512))
        self.ptr = ptr(self.base.value + 256)
        self.poison = np.full(bytes + 512, 0xA5, np.uint8)
        check(
            hip.hipMemcpy(
                self.base, ptr(self.poison.ctypes.data), self.poison.nbytes, 1
            )
        )

    def put(self, x):
        x = np.ascontiguousarray(x)
        assert x.nbytes == self.size
        check(hip.hipMemcpy(self.ptr, ptr(x.ctypes.data), x.nbytes, 1))

    def get(self, dtype, shape):
        x = np.empty(shape, dtype)
        assert x.nbytes == self.size
        check(hip.hipMemcpy(ptr(x.ctypes.data), self.ptr, x.nbytes, 2))
        return x

    def guards(self):
        x = np.empty(self.size + 512, np.uint8)
        check(hip.hipMemcpy(ptr(x.ctypes.data), self.base, x.nbytes, 2))
        np.testing.assert_array_equal(x[:256], 0xA5)
        np.testing.assert_array_equal(x[-256:], 0xA5)

    def close(self):
        self.guards()
        check(hip.hipFree(self.base))


def device(x):
    b = Buffer(x.nbytes)
    b.put(x)
    return b
