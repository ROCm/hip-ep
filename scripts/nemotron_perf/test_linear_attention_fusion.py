#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check state-cast matching and keep all non-matching IR unchanged."""

import argparse
import os
import subprocess

SOURCE = """func.func @state(%ctx: !hip.context, %other: !hip.context,
    %q: tensor<1x1x8192xf32>, %k: tensor<1x1x8192xf32>,
    %v: tensor<1x1x4096xf32>, %g: tensor<1x1x64xf32>,
    %past: tensor<1x64x128x64xf16>) -> (tensor<1x1x4096xf32>, tensor<1x64x128x64xf16>) {
  %e32 = tensor.empty() : tensor<1x64x128x64xf32>
  %past32 = hip.cast(%ctx) ins(%past : tensor<1x64x128x64xf16>)
      outs(%e32 : tensor<1x64x128x64xf32>) {to = 1 : i64} : tensor<1x64x128x64xf32>
  %out = tensor.empty() : tensor<1x1x4096xf32>
  %la:2 = hip.linear_attention(%ctx)
      ins(%q, %k, %v : tensor<1x1x8192xf32>, tensor<1x1x8192xf32>, tensor<1x1x4096xf32>)
      past_state(%past32 : tensor<1x64x128x64xf32>)
      decay(%g : tensor<1x1x64xf32>)
      outs(%out, %e32 : tensor<1x1x4096xf32>, tensor<1x64x128x64xf32>)
      {q_num_heads = 64 : i64, kv_num_heads = 64 : i64, update_rule = "gated"}
      : tensor<1x1x4096xf32>, tensor<1x64x128x64xf32>
  %e16 = tensor.empty() : tensor<1x64x128x64xf16>
  %present = hip.cast(%ctx) ins(%la#1 : tensor<1x64x128x64xf32>)
      outs(%e16 : tensor<1x64x128x64xf16>) {to = 10 : i64} : tensor<1x64x128x64xf16>
  return %la#0, %present : tensor<1x1x4096xf32>, tensor<1x64x128x64xf16>
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    args = parser.parse_args()

    def run(source, enabled=True):
        env = dict(os.environ, HIPDNN_EP_EXPERIMENTAL_LA_F16_STATE=str(int(enabled)))
        result = subprocess.run(
            [args.compiler, "--hip-fusion-transform", "--canonicalize"],
            input=source,
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    result = run(SOURCE)
    assert result.count("hip.fp16_state_boundary") == 1, result
    assert "hip.cast" not in result, result
    assert result == run(result), "rewrite is not idempotent"
    # The real pipeline queries both cast results before canonicalization.
    dynamic = SOURCE.replace("tensor<1x64x128x64x", "tensor<?x64x128x64x")
    dynamic = dynamic.replace(
        "%e32 = tensor.empty()",
        """%c0 = arith.constant 0 : index
  %batch = tensor.dim %past, %c0 : tensor<?x64x128x64xf16>
  %e32 = tensor.empty(%batch)""",
    )
    dynamic = dynamic.replace(
        "%out = tensor.empty()",
        """%batch32 = tensor.dim %past32, %c0 : tensor<?x64x128x64xf32>
  %s32 = tensor.empty(%batch32) : tensor<?x64x128x64xf32>
  %out = tensor.empty()""",
    )
    dynamic = dynamic.replace("outs(%out, %e32", "outs(%out, %s32")
    dynamic = dynamic.replace(
        "%e16 = tensor.empty()",
        """%present_batch = tensor.dim %la#1, %c0 : tensor<?x64x128x64xf32>
  %e16 = tensor.empty(%present_batch)""",
    )
    dynamic_result = run(dynamic)
    assert dynamic_result.count("hip.fp16_state_boundary") == 1, dynamic_result
    assert "hip.cast" not in dynamic_result, dynamic_result
    assert dynamic_result == run(dynamic_result)
    assert "hip.fp16_state_boundary" not in run(SOURCE, False)
    for negative in [
        SOURCE.replace("%past32 = hip.cast(%ctx)", "%past32 = hip.cast(%other)"),
        SOURCE.replace("%present = hip.cast(%ctx)", "%present = hip.cast(%other)"),
        SOURCE.replace("xf16>", "xbf16>"),
    ]:
        assert "hip.fp16_state_boundary" not in run(negative)

    for extra in ("%past32", "%la#1"):
        negative = SOURCE.replace(
            "-> (tensor<1x1x4096xf32>, tensor<1x64x128x64xf16>)",
            "-> (tensor<1x1x4096xf32>, tensor<1x64x128x64xf16>, tensor<1x64x128x64xf32>)",
        )
        negative = negative.replace(
            "return %la#0, %present : tensor<1x1x4096xf32>, tensor<1x64x128x64xf16>",
            f"return %la#0, %present, {extra} : tensor<1x1x4096xf32>, tensor<1x64x128x64xf16>, tensor<1x64x128x64xf32>",
        )
        assert "hip.fp16_state_boundary" not in run(negative)
    print("PASS: state fusion, default-off, context/dtype/user guards, idempotence")


if __name__ == "__main__":
    main()
