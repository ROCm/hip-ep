#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Check the explicit-repeat to native-head-broadcast rewrite."""

import argparse
import os
import subprocess


def graph():
    parts = [
        """func.func @heads(%ctx: !hip.context, %other: !hip.context,
      %q: tensor<?x?x1024xf32>, %k: tensor<?x?x1024xf32>,
      %v: tensor<?x?x4096xf32>, %past: tensor<?x64x128x64xf32>,
      %decay: tensor<?x?x64xf32>, %unknown: tensor<5xi64>, %wrong: index)
      -> (tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %shape = hip.constant {value = dense<[1, 1, 1, 8, 1]> : tensor<5xi64>} : tensor<5xi64>
    """
    ]
    for name in ("q", "k"):
        parts.append(f"""
      %{name}b = tensor.dim %{name}, %c0 : tensor<?x?x1024xf32>
      %{name}t = tensor.dim %{name}, %c1 : tensor<?x?x1024xf32>
      %{name}split = tensor.expand_shape %{name} [[0], [1], [2, 3, 4]]
          output_shape [%{name}b, %{name}t, 8, 1, 128]
          : tensor<?x?x1024xf32> into tensor<?x?x8x1x128xf32>
      %{name}empty = tensor.empty(%{name}b, %{name}t) : tensor<?x?x8x8x128xf32>
      %{name}repeat = hip.expand(%ctx) ins(%{name}split, %shape : tensor<?x?x8x1x128xf32>, tensor<5xi64>)
          outs(%{name}empty : tensor<?x?x8x8x128xf32>) : tensor<?x?x8x8x128xf32>
      %{name}flat = tensor.collapse_shape %{name}repeat [[0], [1], [2, 3, 4]]
          : tensor<?x?x8x8x128xf32> into tensor<?x?x8192xf32>
        """)
    parts.append("""
      %ob = tensor.dim %qflat, %c0 : tensor<?x?x8192xf32>
      %ot = tensor.dim %qflat, %c1 : tensor<?x?x8192xf32>
      %out = tensor.empty(%ob, %ot) : tensor<?x?x4096xf32>
      %state = tensor.empty(%qb) : tensor<?x64x128x64xf32>
      %la:2 = hip.linear_attention(%ctx)
          ins(%qflat, %kflat, %v : tensor<?x?x8192xf32>, tensor<?x?x8192xf32>, tensor<?x?x4096xf32>)
          past_state(%past : tensor<?x64x128x64xf32>) decay(%decay : tensor<?x?x64xf32>)
          outs(%out, %state : tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>)
          {q_num_heads = 64 : i64, kv_num_heads = 64 : i64, update_rule = "gated"}
          : tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>
      return %la#0, %la#1 : tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>
    }""")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    args = parser.parse_args()

    def run(source, enabled=True):
        env = dict(
            os.environ, HIPDNN_EP_EXPERIMENTAL_LA_COMPACT_HEADS=str(int(enabled))
        )
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

    source = graph()
    result = run(source)
    assert "q_num_heads = 8 : i64" in result, result
    assert "hip.expand" not in result, result
    assert result == run(result), "rewrite is not idempotent"
    assert "q_num_heads = 64 : i64" in run(source, False)
    cases = {
        "unknown-repeat": source.replace("%qsplit, %shape", "%qsplit, %unknown"),
        "wrong-context": source.replace(
            "%qrepeat = hip.expand(%ctx)", "%qrepeat = hip.expand(%other)"
        ),
        "wrong-batch": source.replace(
            "%qempty = tensor.empty(%qb, %qt)", "%qempty = tensor.empty(%wrong, %qt)"
        ),
        "changed-output-heads": source.replace("q_num_heads = 64", "q_num_heads = 128"),
        "repeat-other-axis": source.replace("[1, 1, 1, 8, 1]", "[1, 1, 8, 1, 1]"),
    }
    for name, negative in cases.items():
        assert "q_num_heads = 8 : i64" not in run(negative), name
    # Other consumers must keep the original repeated tensor.
    extra = source.replace(
        "-> (tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>)",
        "-> (tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>, tensor<?x?x8192xf32>)",
    )
    extra = extra.replace(
        "return %la#0, %la#1 : tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>",
        "return %la#0, %la#1, %qflat : tensor<?x?x4096xf32>, tensor<?x64x128x64xf32>, tensor<?x?x8192xf32>",
    )
    result = run(extra)
    assert "q_num_heads = 8 : i64" in result and result.count("hip.expand") == 1, result
    print(
        "PASS: native head broadcast, shape/context guards, shared users, default-off, idempotence"
    )


if __name__ == "__main__":
    main()
