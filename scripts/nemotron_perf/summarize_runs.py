#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Summarize timings and whole-device samples from completed runs."""

import argparse
import json
from pathlib import Path
import re


def summary(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (root / "exit_status").read_text().strip() != "0":
        raise ValueError(f"Run did not succeed: {root}")
    log = (root / "run.log").read_text()

    def metric(section, label):
        text = log.split(section + ":", 1)[1]
        return float(re.search(re.escape(label) + r"\s*([0-9.]+)", text)[1])

    samples = [
        json.loads(line)
        for line in (root / "gpu-device-samples.jsonl").read_text().splitlines()
    ]
    valid = [
        row
        for row in samples
        if all(
            isinstance(row.get(key), int)
            for key in ("mem_info_vram_used", "mem_info_gtt_used")
        )
    ]
    peak = max(
        valid, key=lambda row: row["mem_info_vram_used"] + row["mem_info_gtt_used"]
    )
    initial = valid[0]["mem_info_vram_used"] + valid[0]["mem_info_gtt_used"]
    total = peak["mem_info_vram_used"] + peak["mem_info_gtt_used"]
    shape = re.search(r"prompt tokens: (\d+), tokens to generate: (\d+)", log)

    def enabled(key, variable):
        return manifest.get(key, variable + "=1" in manifest["command"])

    return {
        "run": str(root),
        "sparse_experts": manifest["sparse_experts"],
        "head_f16_matvec": enabled(
            "head_f16_matvec", "HIPDNN_EP_EXPERIMENTAL_HEAD_F16_MATVEC"
        ),
        "la_f16_state": enabled("la_f16_state", "HIPDNN_EP_EXPERIMENTAL_LA_F16_STATE"),
        "la_compact_heads": enabled(
            "la_compact_heads", "HIPDNN_EP_EXPERIMENTAL_LA_COMPACT_HEADS"
        ),
        "la_fixed_geometry": enabled(
            "la_fixed_geometry", "HIPDNN_EP_EXPERIMENTAL_LA_FIXED_GEOMETRY"
        ),
        "swiglu_f32": enabled("swiglu_f32", "HIPDNN_EP_EXPERIMENTAL_SWIGLU_F32"),
        "router_sequential_f32": enabled(
            "router_sequential_f32", "HIPDNN_EP_EXPERIMENTAL_SEQUENTIAL_ROUTER_F32"
        ),
        "blas_tuning_override_sha256": manifest.get("blas_tuning_override", {}).get(
            "sha256"
        ),
        "nbits_autotune_mode": manifest.get("nbits_autotune_mode", "lookup"),
        "diagnostic": manifest["diagnostic"],
        "prompt_tokens": int(shape[1]),
        "new_tokens": int(shape[2]),
        "ttft_seconds": metric("Prompt processing (time to first token)", "avg (us):")
        / 1e6,
        "decode_tps": 1e6 / metric("Token generation", "avg (us):"),
        "sampled_device_peak_gib": total / 2**30,
        "sampled_device_initial_gib": initial / 2**30,
        "sampled_device_peak_above_initial_gib": (total - initial) / 2**30,
        "peak_temperature_c": max(row.get("temp1_input", 0) for row in samples) / 1000,
        "peak_power_w": max(row.get("power1_average", 0) for row in samples) / 1e6,
        "memory_scope": "simultaneous whole-device VRAM+GTT sample; not exact model allocation",
        "compiler_sha256": manifest["compiler_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summary(root) for root in args.runs], indent=2))


if __name__ == "__main__":
    main()
