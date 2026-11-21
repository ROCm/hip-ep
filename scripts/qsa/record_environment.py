#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Record physical device/runtime and immutable input receipts for this port."""
# isort: skip_file

import ctypes
import importlib.metadata
import json
import platform
import re
import subprocess

from runtime import CORE, INSTALL, SDK, WORK
import psutil
import torch

from prepare_baseline import ROOT, SELECTOR, SELECTOR_SHA, sha


def device_attributes(hip):
    header = (SDK / "include/hip/hip_runtime_api.h").read_text()
    enum = (
        header.split("typedef enum hipDeviceAttribute_t", 1)[1]
        .split("{", 1)[1]
        .split("}", 1)[0]
    )
    enum = re.sub(r"/\*.*?\*/|//[^\n]*", "", enum, flags=re.DOTALL)
    values, current = {}, -1
    for entry in enum.split(","):
        match = re.search(r"(hipDeviceAttribute\w+)\s*(?:=\s*([^\n]+))?", entry)
        if match is None:
            continue
        name, expression = match.groups()
        if expression:
            expression = expression.strip()
            current = values[expression] if expression in values else int(expression, 0)
        else:
            current += 1
        values[name] = current
    result = {}
    for suffix in [
        "CooperativeLaunch",
        "CooperativeMultiDeviceLaunch",
        "WarpSize",
        "MultiprocessorCount",
        "Integrated",
    ]:
        key = "hipDeviceAttribute" + suffix
        value = ctypes.c_int()
        status = hip.hipDeviceGetAttribute(ctypes.byref(value), values[key], 0)
        result[suffix] = {"enum": values[key], "status": status, "value": value.value}
    return result


def main():
    assert sha(SELECTOR) == SELECTOR_SHA
    WORK.mkdir(parents=True, exist_ok=True)
    hip = ctypes.CDLL(str(CORE / "bin/amdhip64_7.dll"))
    runtime, driver = ctypes.c_int(), ctypes.c_int()
    assert hip.hipRuntimeGetVersion(ctypes.byref(runtime)) == 0
    assert hip.hipDriverGetVersion(ctypes.byref(driver)) == 0
    hardware = subprocess.check_output(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            (
                "@{gpu=Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID;"
                "cpu=Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors;"
                "memory=Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed} | ConvertTo-Json -Depth 4"
            ),
        ],
        text=True,
    )
    receipt = {
        "platform": platform.platform(),
        "machine": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "hip_runtime": runtime.value,
        "hip_driver": driver.value,
        "hardware": json.loads(hardware),
        "device": str(torch.cuda.get_device_properties(0)),
        "device_attributes": device_attributes(hip),
        "os_managed_memory_bytes": psutil.virtual_memory().total,
        "os_available_memory_bytes_at_receipt": psutil.virtual_memory().available,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["onnx", "onnxruntime", "transformers", "safetensors"]
        },
        "hip_ep_checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "baseline_install": str(INSTALL),
        "selector_sha256": SELECTOR_SHA,
        "training_updates": 0,
        "installed_libraries": {
            str(p): sha(p)
            for p in [
                INSTALL / "bin/hipgpu.dll",
                INSTALL / "bin/custom_kernels_gfx1151.dll",
                CORE / "bin/amdhip64_7.dll",
            ]
        },
    }
    baseline_build = ROOT / "_local/ring-1259/results/build-existing.json"
    build_receipt = json.loads(baseline_build.read_text())
    for relative, digest in build_receipt["files"].items():
        assert sha(INSTALL / relative) == digest
    receipt["baseline_build_receipt"] = build_receipt
    receipt["baseline_build_receipt_sha256"] = sha(baseline_build)
    (WORK / "environment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    preparation = json.loads((WORK / "preparation.json").read_text())
    preparation.update(
        stage="original_source_verified",
        transfer="SHA256-verified bounded HTTPS ranges after stalled Hub transports",
        files={
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted((WORK / "base-model").iterdir())
            if p.is_file()
        },
    )
    (WORK / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
