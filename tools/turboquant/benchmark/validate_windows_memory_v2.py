#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Characterize WDDM update delay/retention using process and adapter counters."""

from pathlib import Path
import ctypes as c
import json
import os
import sys
import time
from windows_gpu_memory import GpuMemory

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
handle = os.add_dll_directory(str(CORE / "bin"))
hip = c.CDLL(str(CORE / "bin/amdhip64_7.dll"))
hip.hipMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t]
hip.hipFree.argtypes = [c.c_void_p]
hip.hipHostMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t, c.c_uint]
hip.hipHostFree.argtypes = [c.c_void_p]
report = {"status": "running", "pid": os.getpid(), "samples": []}
try:
    q = c.c_void_p()
    assert hip.hipMalloc(c.byref(q), 4096) == 0
    assert hip.hipFree(q) == 0
    time.sleep(1)
    process = GpuMemory()
    adapter = GpuMemory(adapter=True)

    def sample(phase):
        report["samples"].append(
            {"phase": phase, "process": process.collect(), "adapter": adapter.collect()}
        )

    sample("initial")
    for i in range(3):
        p = c.c_void_p()
        assert hip.hipHostMalloc(c.byref(p), 64 * 2**20, 3) == 0
        report.setdefault("pointers", []).append(p.value)
        sample(f"{i}:allocated-immediate")
        time.sleep(1)
        sample(f"{i}:allocated-1s")
        assert hip.hipHostFree(p) == 0
        assert hip.hipDeviceSynchronize() == 0
        sample(f"{i}:freed-immediate")
        for j in range(3):
            time.sleep(1)
            sample(f"{i}:freed-{j + 1}s")
    report["status"] = "completed"
except BaseException as exc:
    report.update(status="failed", error=repr(exc))
(ROOT / "results/profile/wddm-validation-v2.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
