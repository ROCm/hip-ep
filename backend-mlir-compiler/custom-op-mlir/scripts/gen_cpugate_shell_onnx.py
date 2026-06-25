#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Emit the CPUGate shell ONNX used by debug CPU fallback (Quark-style)."""

from __future__ import annotations

import sys
from pathlib import Path

from onnx import TensorProto, helper
import onnx


def build_model() -> onnx.ModelProto:
    i = helper.make_tensor_value_info("i", TensorProto.UINT8, [1])
    o = helper.make_tensor_value_info("o", TensorProto.UINT8, [1])
    n = helper.make_node(
        "CpuGate",
        ["i"],
        ["o"],
        name="theGate",
        domain="com.amd.morphizen.cpu",
    )
    g = helper.make_graph([n], "MorphiZenCPUGate", [i], [o])
    return helper.make_model(
        g,
        opset_imports=[helper.make_opsetid("com.amd.morphizen.cpu", 1)],
        producer_name="MorphiZen",
        ir_version=9,
    )


def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "onnx" / "cpugate_shell.onnx"
    )
    m = build_model()
    onnx.checker.check_model(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
