#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check the rounded residual chain, broadcast mapping, and refusal guards."""

import argparse
import re
import subprocess

SOURCE = """func.func @residual(%ctx: !hip.context, %other: !hip.context,
    %base: tensor<2x3x32xf32>, %x: tensor<1x1x4x1xf32>,
    %y: tensor<2x3x4x8xf32>) -> tensor<2x3x32xf16> {
  %pe = tensor.empty() : tensor<2x3x4x8xf32>
  %p = hip.mul(%ctx) ins(%x, %y : tensor<1x1x4x1xf32>, tensor<2x3x4x8xf32>)
      outs(%pe : tensor<2x3x4x8xf32>) -> tensor<2x3x4x8xf32>
  %flat = tensor.collapse_shape %p [[0], [1], [2, 3]] : tensor<2x3x4x8xf32> into tensor<2x3x32xf32>
  %e = tensor.empty() : tensor<2x3x32xf16>
  %a = hip.cast(%ctx) ins(%base : tensor<2x3x32xf32>) outs(%e : tensor<2x3x32xf16>) {to = 10 : i64} : tensor<2x3x32xf16>
  %b = hip.cast(%ctx) ins(%flat : tensor<2x3x32xf32>) outs(%e : tensor<2x3x32xf16>) {to = 10 : i64} : tensor<2x3x32xf16>
  %add = hip.add(%ctx) ins(%a, %b : tensor<2x3x32xf16>, tensor<2x3x32xf16>) outs(%e : tensor<2x3x32xf16>) -> tensor<2x3x32xf16>
  return %add : tensor<2x3x32xf16>
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    args = parser.parse_args()

    def run(source, *passes):
        result = subprocess.run(
            [args.compiler, *(passes or ("--hip-fusion-transform", "--canonicalize"))],
            input=source,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    def check(source):
        result = run(source)
        assert result.count("hip.rounded_scaled_add") == 1, result
        assert (
            "hip.cast" not in result
            and "hip.mul(" not in result
            and "hip.add(" not in result
        ), result
        assert result == run(result), "not idempotent"
        return result

    check(SOURCE)
    assert "product_on_lhs = true" in check(SOURCE.replace("ins(%a, %b", "ins(%b, %a"))
    direct = SOURCE.replace("2x3x32", "2x3x4x8")
    direct = "\n".join(
        line for line in direct.splitlines() if "%flat =" not in line
    ).replace("ins(%flat :", "ins(%p :")
    check(direct)
    scalar = re.sub(r"tensor<[^>]+xf32>", "tensor<f32>", direct)
    scalar = re.sub(r"tensor<[^>]+xf16>", "tensor<f16>", scalar)
    check(scalar)

    def base_shape(shape):
        result = SOURCE.replace(
            "%base: tensor<2x3x32xf32>", f"%base: tensor<{shape}xf32>"
        )
        result = result.replace(
            "  %a = hip.cast",
            f"  %be = tensor.empty() : tensor<{shape}xf16>\n  %a = hip.cast",
        )
        result = result.replace(
            "ins(%base : tensor<2x3x32xf32>) outs(%e : tensor<2x3x32xf16>) {to = 10 : i64} : tensor<2x3x32xf16>",
            f"ins(%base : tensor<{shape}xf32>) outs(%be : tensor<{shape}xf16>) {{to = 10 : i64}} : tensor<{shape}xf16>",
        )
        return result.replace(
            "ins(%a, %b : tensor<2x3x32xf16>", f"ins(%a, %b : tensor<{shape}xf16>"
        )

    check(base_shape("1x3x32"))
    assert "hip.rounded_scaled_add" not in run(base_shape("2x3x1"))
    different_groups = SOURCE.replace("2x3x32", "6x4x8").replace(
        "[[0], [1], [2, 3]]", "[[0, 1], [2], [3]]"
    )
    assert "hip.rounded_scaled_add" not in run(different_groups)
    dynamic = SOURCE.replace("2x3x", "?x?x")
    dynamic = dynamic.replace(
        "%pe = tensor.empty()",
        """%c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch = tensor.dim %y, %c0 : tensor<?x?x4x8xf32>
  %tokens = tensor.dim %y, %c1 : tensor<?x?x4x8xf32>
  %pe = tensor.empty(%batch, %tokens)""",
    )
    dynamic = dynamic.replace(
        "%e = tensor.empty()",
        """%d = tensor.dim %flat, %c0 : tensor<?x?x32xf32>
  %e = tensor.empty(%d, %tokens)""",
    )
    check(dynamic)
    # The standalone op also lowers without any model-specific information.
    lowered = run(
        """func.func @lower(%c: !hip.context, %b: memref<2x3x4x8xf32>,
        %x: memref<1x1x4x1xf32>, %y: memref<2x3x4x8xf32>, %o: memref<2x3x4x8xf16>) {
      hip.rounded_scaled_add(%c) ins(%b, %x, %y : memref<2x3x4x8xf32>, memref<1x1x4x1xf32>, memref<2x3x4x8xf32>) outs(%o : memref<2x3x4x8xf16>)
      return
    }""",
        "--convert-hip-to-llvm",
    )
    assert "llvm.call @wrap_rounded_scaled_add_f16" in lowered
    for before in ("%a = hip.cast(%ctx)", "%b = hip.cast(%ctx)", "%p = hip.mul(%ctx)"):
        assert "hip.rounded_scaled_add" not in run(
            SOURCE.replace(before, before.replace("%ctx", "%other"))
        )
    assert "hip.rounded_scaled_add" not in run(SOURCE.replace("xf16>", "xbf16>"))
    for value, shape in (
        ("%p", "2x3x4x8xf32"),
        ("%flat", "2x3x32xf32"),
        ("%a", "2x3x32xf16"),
        ("%b", "2x3x32xf16"),
    ):
        negative = SOURCE.replace(
            "  return %add",
            f"  func.call @sink({value}) : (tensor<{shape}>) -> ()\n  return %add",
        )
        negative += f"\nfunc.func private @sink(tensor<{shape}>)"
        assert "hip.rounded_scaled_add" not in run(negative)
    print(
        "PASS: rounded scaled add, both orders, scalar/broadcast/flattened/direct/dynamic shapes, reshape/shared-user/context guards, lowering, idempotence"
    )


if __name__ == "__main__":
    main()
