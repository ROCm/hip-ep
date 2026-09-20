// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @qadd
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xi8>, %[[RHS:.*]]: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x128x32xi8>
// CHECK-NEXT:    %[[QADD:.*]] = hip.qadd(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x32xi8>, tensor<1x128x32xi8>) outs(%[[INIT]] : tensor<1x128x32xi8>) {lhs_scale = 2.500000e-01 : f32, lhs_zp = -5 : i64, output_scale = 1.250000e-01 : f32, output_zp = 7 : i64, rhs_scale = 5.000000e-01 : f32, rhs_zp = 3 : i64} : tensor<1x128x32xi8>
// CHECK-NEXT:    return %[[QADD]] : tensor<1x128x32xi8>
// CHECK-NEXT:  }
func.func @qadd(%ctx: !hip.context,
                %lhs: tensor<1x128x32xi8>,
                %rhs: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
  %lhs_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %lhs_zp = hip.constant {value = dense<-5> : tensor<i8>} : tensor<i8>
  %rhs_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %rhs_zp = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %out_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %out_zp = hip.constant {value = dense<7> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<1x128x32xf32>
  %dq_lhs = hip.dequantize_linear(%ctx)
      ins(%lhs, %lhs_scale : tensor<1x128x32xi8>, tensor<f32>)
      zero_point(%lhs_zp : tensor<i8>)
      outs(%e0 : tensor<1x128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x32xf32>

  %e1 = tensor.empty() : tensor<1x128x32xf32>
  %dq_rhs = hip.dequantize_linear(%ctx)
      ins(%rhs, %rhs_scale : tensor<1x128x32xi8>, tensor<f32>)
      zero_point(%rhs_zp : tensor<i8>)
      outs(%e1 : tensor<1x128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x32xf32>

  %e2 = tensor.empty() : tensor<1x128x32xf32>
  %sum = hip.add(%ctx)
      ins(%dq_lhs, %dq_rhs : tensor<1x128x32xf32>, tensor<1x128x32xf32>)
      outs(%e2 : tensor<1x128x32xf32>) -> tensor<1x128x32xf32>

  %e3 = tensor.empty() : tensor<1x128x32xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%sum, %out_scale : tensor<1x128x32xf32>, tensor<f32>)
      zero_point(%out_zp : tensor<i8>)
      outs(%e3 : tensor<1x128x32xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<1x128x32xi8>

  return %q : tensor<1x128x32xi8>
}
