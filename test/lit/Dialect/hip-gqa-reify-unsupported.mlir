// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-each=0 --test-gqa-reify-failure-atomic %s 2>&1 | FileCheck %s

// CHECK: error: 'hip.gqa' op softcap must be exactly zero
// CHECK: remark: failed GQA reification left IR unchanged
// CHECK-NOT: failed GQA reification mutated IR
func.func @softcap_failure_is_atomic(
    %ctx: !hip.context, %query: tensor<?x?x32xf16>,
    %key: tensor<?x?x16xf16>, %value: tensor<?x?x16xf16>,
    %past_key: tensor<?x2x?x8xf16>, %past_value: tensor<?x2x?x8xf16>,
    %seqlens: tensor<?xi32>, %total: tensor<i32>,
    %out: tensor<?x?x?xf16>, %present_key: tensor<?x2x?x8xf16>,
    %present_value: tensor<?x2x?x8xf16>) -> index {
  %result:3 = hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        tensor<?x?x32xf16>, tensor<?x?x16xf16>, tensor<?x?x16xf16>,
        tensor<?x2x?x8xf16>, tensor<?x2x?x8xf16>, tensor<?xi32>, tensor<i32>)
    outs(%out, %present_key, %present_value :
        tensor<?x?x?xf16>, tensor<?x2x?x8xf16>, tensor<?x2x?x8xf16>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64,
     test.gqa_reify_failure_atomic}
    : tensor<?x?x?xf16>, tensor<?x2x?x8xf16>, tensor<?x2x?x8xf16>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %result#0, %c0 : tensor<?x?x?xf16>
  return %dim : index
}
