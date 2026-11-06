#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run the Ring ONNX regressions; fail on skips or missing Ring dispatch."""

import argparse
import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def verify_results(report, log):
    cases = list(ET.parse(report).iter("testcase"))
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("skipped", "failure", "error")
    ):
        raise RuntimeError(
            "Ring CI requires a nonempty test suite with no skips or failures"
        )
    queries = [int(q) for q in re.findall(r"Ring GQA q=(\d+) past=", log)]
    if 1 not in queries or not any(q > 1 for q in queries):
        raise RuntimeError("Missing Ring decode or prefill dispatch in runtime log")
    return len(cases)


def worker(package, output):
    # Keep the DLL-directory handle alive for the entire pytest invocation.
    bin_dir = package / "bin"
    dll_directory = os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["HIPDNN_EP_STRICT"] = "1"
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    os.environ["HIPDNN_EP_USE_RING_ATTENTION"] = "1"
    ort_library = ctypes.CDLL(str(bin_dir / "onnxruntime.dll"))
    kernels = ctypes.CDLL(str(bin_dir / "custom_kernels_gfx11-generic.dll"))
    supported = kernels.hip_gqa_ring_device_supported
    supported.argtypes = []
    supported.restype = ctypes.c_int
    if supported() != 1:
        raise RuntimeError(
            "Ring-enabled generic kernels and a gfx1151 device are required"
        )
    import pytest

    result = pytest.main(
        [
            str(Path(__file__).parent / "tests" / "test_gqa.py"),
            "-k",
            "gqa_ring",
            "-v",
            "-s",
            "--no-cache",
            "--ep-name",
            "hipgpu",
            "--ep-dll",
            str(bin_dir / "hipgpu.dll"),
            "--ep-option",
            f"config_file={package / 'morphizen_config.json'}",
            "--output-dir",
            str(output / "numeric"),
            "--junitxml",
            str(output / "junit.xml"),
        ]
    )
    # Retain loaded runtime handles through test completion.
    _ = ort_library, kernels, dll_directory
    return int(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    package, output = args.package.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(package, output)
    # A child process captures native runtime diagnostics as well as Python output.
    log_path = output / "ring.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--package",
                str(package),
                "--output",
                str(output),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=False,
        )
    log = log_path.read_text(encoding="utf-8", errors="replace")
    print(log[-12000:])
    if result.returncode:
        return result.returncode
    count = verify_results(output / "junit.xml", log)
    print(f"PASS: {count} Ring tests, no skips; prefill and decode dispatch confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
