#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check the opt-in activation rewrite, shape users, and broadcast contract."""

import argparse
import os
import subprocess


def graph(dynamic=False, broadcast=False):
    g = "?x?x4096" if dynamic else "1x3x17"
    u = "?x?x4096" if dynamic else ("2x1x17" if broadcast else g)
    o = "?x?x4096" if dynamic else ("2x3x17" if broadcast else g)
    pre = """%c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index"""
    dims = ""
    if dynamic:
        pre += f"""
  %gd0 = tensor.dim %g, %c0 : tensor<{g}xf16>
  %gd1 = tensor.dim %g, %c1 : tensor<{g}xf16>
  %ud0 = tensor.dim %u, %c0 : tensor<{u}xf16>
  %ud1 = tensor.dim %u, %c1 : tensor<{u}xf16>"""
        dims = "%gd0, %gd1"
    query = (
        (
            f"%s0 = tensor.dim %s, %c0 : tensor<{g}xf32>\n"
            f"  %s1 = tensor.dim %gc, %c1 : tensor<{g}xf32>"
        )
        if dynamic
        else ""
    )
    mquery = (
        (
            f"%m0 = tensor.dim %m, %c0 : tensor<{o}xf32>\n"
            f"  %m1 = tensor.dim %m, %c1 : tensor<{o}xf32>"
        )
        if dynamic
        else ""
    )
    return f"""func.func @activation(%ctx: !hip.context, %other: !hip.context,
    %g: tensor<{g}xf16>, %u: tensor<{u}xf16>) -> tensor<{o}xf16> {{
  {pre}
  %ge = tensor.empty({dims}) : tensor<{g}xf32>
  %ue = tensor.empty({"%ud0, %ud1" if dynamic else ""}) : tensor<{u}xf32>
  %gc = hip.cast(%ctx) ins(%g : tensor<{g}xf16>) outs(%ge : tensor<{g}xf32>)
      {{to = 1 : i64}} : tensor<{g}xf32>
  %uc = hip.cast(%ctx) ins(%u : tensor<{u}xf16>) outs(%ue : tensor<{u}xf32>)
      {{to = 1 : i64}} : tensor<{u}xf32>
  %s = hip.swish(%ctx) ins(%gc : tensor<{g}xf32>) outs(%ge : tensor<{g}xf32>)
      {{alpha = 1.0 : f64}} : tensor<{g}xf32>
  {query}
  %me = tensor.empty({"%s0, %s1" if dynamic else ""}) : tensor<{o}xf32>
  %m = hip.mul(%ctx) ins(%uc, %s : tensor<{u}xf32>, tensor<{g}xf32>)
      outs(%me : tensor<{o}xf32>) -> tensor<{o}xf32>
  {mquery}
  %oe = tensor.empty({"%m0, %m1" if dynamic else ""}) : tensor<{o}xf16>
  %r = hip.cast(%ctx) ins(%m : tensor<{o}xf32>) outs(%oe : tensor<{o}xf16>)
      {{to = 10 : i64}} : tensor<{o}xf16>
  return %r : tensor<{o}xf16>
}}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    args = parser.parse_args()

    def run(source, enabled=True):
        env = dict(os.environ, HIPDNN_EP_EXPERIMENTAL_SWIGLU_F32=str(int(enabled)))
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

    for source in (
        graph(),
        graph(dynamic=True),
        graph(broadcast=True),
        graph().replace("ins(%uc, %s", "ins(%s, %uc"),
    ):
        result = run(source)
        assert result.count("hip.swiglu_f32") == 1, result
        assert "hip.cast" not in result and "hip.swish" not in result, result
        assert "hip.mul" not in result, result
        assert result == run(result), "rewrite is not idempotent"
        assert "hip.swiglu_f32" not in run(source, False)

    source = graph()
    negatives = [
        source.replace("1x3x17", "1x1x1x3x17"),
        source.replace("alpha = 1.0", "alpha = 0.5"),
        source.replace("xf16>", "xbf16>"),
    ]
    for name in (
        "%gc = hip.cast",
        "%uc = hip.cast",
        "%s = hip.swish",
        "%m = hip.mul",
        "%r = hip.cast",
    ):
        negatives.append(source.replace(name + "(%ctx)", name + "(%other)"))
    for value in ("%gc", "%uc", "%s", "%m"):
        extra = source.replace(
            "-> tensor<1x3x17xf16> {", "-> (tensor<1x3x17xf16>, tensor<1x3x17xf32>) {"
        )
        extra = extra.replace(
            "return %r : tensor<1x3x17xf16>",
            f"return %r, {value} : tensor<1x3x17xf16>, tensor<1x3x17xf32>",
        )
        negatives.append(extra)
    for negative in negatives:
        assert "hip.swiglu_f32" not in run(negative), negative
    print(
        "PASS: activation fusion, broadcast, shape users, default-off, guards, idempotence"
    )


if __name__ == "__main__":
    main()
