#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check that WDDM already counts mapped HIP buffers and releases them."""

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
report = {"status": "running", "pid": os.getpid(), "checks": []}
try:
    q = c.c_void_p()
    assert hip.hipMalloc(c.byref(q), 4096) == 0
    assert hip.hipFree(q) == 0
    time.sleep(0.1)
    memory = GpuMemory()
    baseline = memory.collect()
    for i in range(3):
        check = memory.validate_mapped(hip)
        time.sleep(0.1)
        after = memory.collect()
        check["post_free_delta"] = after["committed"] - baseline["committed"]
        assert abs(check["post_free_delta"]) <= 2**20, check
        report["checks"].append(check)
    report.update(status="passed", baseline=baseline)
except BaseException as exc:
    report.update(status="failed", error=repr(exc))
(ROOT / "results/profile/wddm-validation.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
