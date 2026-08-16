// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_kernel
// CHECK-SAME: %[[INPUT:[^,]+]]: tensor<?x32x?xf16>
// CHECK-SAME: %[[WEIGHT:[^,]+]]: tensor<32x1x?xf16>
// CHECK: %[[OUT_B:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: %[[OUT_L:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: %[[STATE_B:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: %[[K:.*]] = tensor.dim %[[WEIGHT]], %{{.*}}
// CHECK: %[[STATE_LEN:.*]] = arith.addi %[[K]], %{{.*}} : index
// CHECK: return %[[OUT_B]], %[[OUT_L]], %[[STATE_B]], %{{.*}}, %[[STATE_LEN]]
func.func @dynamic_kernel(
    %ctx: !hip.context,
    %input: tensor<?x32x?xf16>,
    %weight: tensor<32x1x?xf16>,
    %output: tensor<?x32x?xf16>,
    %present: tensor<?x32x?xf16>) -> (index, index, index, index, index) {
  %result:2 = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<?x32x?xf16>, tensor<32x1x?xf16>)
      outs(%output, %present : tensor<?x32x?xf16>, tensor<?x32x?xf16>)
      {ndim = 1 : i64}
      : tensor<?x32x?xf16>, tensor<?x32x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %out_b = tensor.dim %result#0, %c0 : tensor<?x32x?xf16>
  %out_l = tensor.dim %result#0, %c2 : tensor<?x32x?xf16>
  %state_b = tensor.dim %result#1, %c0 : tensor<?x32x?xf16>
  %state_c = tensor.dim %result#1, %c1 : tensor<?x32x?xf16>
  %state_len = tensor.dim %result#1, %c2 : tensor<?x32x?xf16>
  return %out_b, %out_l, %state_b, %state_c, %state_len
      : index, index, index, index, index
}
