// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @two_outputs
// CHECK-SAME: %[[INPUT:[^,]+]]: tensor<?x?x64xf16>
// CHECK: %[[O0:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: %[[O1:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: %[[R0:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: %[[R1:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: return %[[O0]], %[[O1]], %[[R0]], %[[R1]]
func.func @two_outputs(
    %ctx: !hip.context,
    %input: tensor<?x?x64xf16>,
    %skip: tensor<?x?x64xf16>,
    %gamma: tensor<64xf16>,
    %output: tensor<?x?x64xf16>,
    %residual: tensor<?x?x64xf16>) -> (index, index, index, index) {
  %result:2 = hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<64xf16>)
      outs(%output, %residual :
           tensor<?x?x64xf16>, tensor<?x?x64xf16>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x?x64xf16>, tensor<?x?x64xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out0 = tensor.dim %result#0, %c0 : tensor<?x?x64xf16>
  %out1 = tensor.dim %result#0, %c1 : tensor<?x?x64xf16>
  %res0 = tensor.dim %result#1, %c0 : tensor<?x?x64xf16>
  %res1 = tensor.dim %result#1, %c1 : tensor<?x?x64xf16>
  return %out0, %out1, %res0, %res1 : index, index, index, index
}

// CHECK-LABEL: func.func @one_output
// CHECK-SAME: %[[INPUT:[^,]+]]: tensor<?x32xf32>
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: return %[[D0]]
func.func @one_output(
    %ctx: !hip.context,
    %input: tensor<?x32xf32>,
    %skip: tensor<?x32xf32>,
    %gamma: tensor<32xf32>,
    %output: tensor<?x32xf32>) -> index {
  %result = hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          tensor<?x32xf32>, tensor<?x32xf32>, tensor<32xf32>)
      outs(%output : tensor<?x32xf32>)
      {epsilon = 1.0e-05 : f32}
      : tensor<?x32xf32>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x32xf32>
  return %d0 : index
}
