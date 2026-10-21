#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Exercise the experimental fusion with small generated HIP graphs."""

import argparse
import os
import subprocess


def graph():
    parts = [
        """func.func @experts(%ctx: !hip.context, %x: tensor<1x1x32xf16>,
        %scores: tensor<1x1x2xf32>, %unknown: tensor<1x1x1xi64>,
        %w: tensor<32x1x16xui8>, %s: tensor<32x1xf16>,
        %z: tensor<32x1xui8>) -> tensor<1x1x32xf16> {
      %k = hip.constant {value = dense<1> : tensor<1xi64>} : tensor<1xi64>
      %axis = hip.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
      %zero = hip.constant {value = dense<0.0> : tensor<f16>} : tensor<f16>
      %empty = tensor.empty() : tensor<1x1x32xf16>
      %gempty = tensor.empty() : tensor<1x1x1xf32>
      %iempty = tensor.empty() : tensor<1x1x1xi64>
      %bempty = tensor.empty() : tensor<1x1x1xi1>
      %hempty = tensor.empty() : tensor<1x1x1xf16>
      %top:2 = hip.top_k(%ctx) ins(%scores, %k : tensor<1x1x2xf32>, tensor<1xi64>)
          outs(%gempty, %iempty : tensor<1x1x1xf32>, tensor<1x1x1xi64>)
          {axis = 2 : i64} : tensor<1x1x1xf32>, tensor<1x1x1xi64>
    """
    ]
    for e in range(2):
        parts.append(f"""
      %id{e} = hip.constant {{value = dense<{e}> : tensor<i64>}} : tensor<i64>
      %eq{e} = hip.equal(%ctx) ins(%top#1, %id{e} : tensor<1x1x1xi64>, tensor<i64>)
          outs(%bempty : tensor<1x1x1xi1>) : tensor<1x1x1xi1>
      %mask{e} = hip.cast(%ctx) ins(%eq{e} : tensor<1x1x1xi1>)
          outs(%gempty : tensor<1x1x1xf32>) {{to = 1 : i64}} : tensor<1x1x1xf32>
      %masked{e} = hip.mul(%ctx) ins(%top#0, %mask{e} : tensor<1x1x1xf32>, tensor<1x1x1xf32>)
          outs(%gempty : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
      %gate{e} = hip.reduce_sum(%ctx) ins(%masked{e}, %axis : tensor<1x1x1xf32>, tensor<1xi64>)
          outs(%gempty : tensor<1x1x1xf32>) : tensor<1x1x1xf32>
      %half{e} = hip.cast(%ctx) ins(%gate{e} : tensor<1x1x1xf32>)
          outs(%hempty : tensor<1x1x1xf16>) {{to = 10 : i64}} : tensor<1x1x1xf16>
      %up{e} = hip.matmul_nbits(%ctx) ins(%x, %w, %s : tensor<1x1x32xf16>, tensor<32x1x16xui8>, tensor<32x1xf16>)
          zero_points(%z : tensor<32x1xui8>) outs(%empty : tensor<1x1x32xf16>)
          {{K = 32 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 1 : i64, accuracy_level = 0 : i64}} : tensor<1x1x32xf16>
      %relu{e} = hip.max(%ctx) ins(%up{e}, %zero : tensor<1x1x32xf16>, tensor<f16>)
          outs(%empty : tensor<1x1x32xf16>) : tensor<1x1x32xf16>
      %square{e} = hip.mul(%ctx) ins(%relu{e}, %relu{e} : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
          outs(%empty : tensor<1x1x32xf16>) -> tensor<1x1x32xf16>
      %down{e} = hip.matmul_nbits(%ctx) ins(%square{e}, %w, %s : tensor<1x1x32xf16>, tensor<32x1x16xui8>, tensor<32x1xf16>)
          zero_points(%z : tensor<32x1xui8>) outs(%empty : tensor<1x1x32xf16>)
          {{K = 32 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 1 : i64, accuracy_level = 0 : i64}} : tensor<1x1x32xf16>
      %result{e} = hip.mul(%ctx) ins(%down{e}, %half{e} : tensor<1x1x32xf16>, tensor<1x1x1xf16>)
          outs(%empty : tensor<1x1x32xf16>) -> tensor<1x1x32xf16>
        """)
    parts.append("""
      %sum = hip.add(%ctx) ins(%result0, %result1 : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
          outs(%empty : tensor<1x1x32xf16>) -> tensor<1x1x32xf16>
      return %sum : tensor<1x1x32xf16>
    }""")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    args = parser.parse_args()

    def compile_ir(source, enabled=True):
        env = dict(os.environ, HIPDNN_EP_EXPERIMENTAL_SPARSE_EXPERTS=str(int(enabled)))
        result = subprocess.run(
            [args.compiler, "--hip-fusion-transform", "--canonicalize", "--cse"],
            input=source,
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    source = graph()
    output = compile_ir(source)
    assert output.count("hip.routed_experts") == 1, output
    assert "hip.matmul_nbits" not in output, output
    assert compile_ir(output) == output, "Fusion is not idempotent"
    cases = {
        "default-disabled": (source, False),
        "unknown-routing": (source.replace("ins(%top#1,", "ins(%unknown,"), True),
        "reverse-sum-order": (
            source.replace("ins(%result0, %result1", "ins(%result1, %result0"),
            True,
        ),
        "duplicate-expert-id": (
            source.replace("dense<1> : tensor<i64>", "dense<0> : tensor<i64>"),
            True,
        ),
        "wrong-activation": (source.replace("dense<0.0>", "dense<1.0>"), True),
    }
    for name, (ir, enabled) in cases.items():
        assert "hip.routed_experts" not in compile_ir(ir, enabled), name
    print("PASS: full expert chain, idempotence, and " + ", ".join(cases))


if __name__ == "__main__":
    main()
