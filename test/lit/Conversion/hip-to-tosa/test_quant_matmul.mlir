// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file %s | FileCheck %s

// An i8 a goes straight to an i8 x i8 -> i32 tosa.matmul with zero points 0.

// CHECK-LABEL: func.func @quant_matmul_i8
// CHECK-SAME: (%[[A:.*]]: tensor<4x64xi8>, %[[B:.*]]: tensor<64x16xi8>)
// CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[A3:.*]] = tosa.reshape %[[A]], %{{.*}} : (tensor<4x64xi8>, !tosa.shape<3>) -> tensor<1x4x64xi8>
// CHECK-DAG: %[[B3:.*]] = tosa.reshape %[[B]], %{{.*}} : (tensor<64x16xi8>, !tosa.shape<3>) -> tensor<1x64x16xi8>
// CHECK: %[[M:.*]] = tosa.matmul %[[A3]], %[[B3]], %[[ZP]], %[[ZP]] : {{.*}} -> tensor<1x4x16xi32>
// CHECK: tosa.reshape %[[M]], %{{.*}} : (tensor<1x4x16xi32>, !tosa.shape<2>) -> tensor<4x16xi32>
func.func @quant_matmul_i8(%a: tensor<4x64xi8>, %b: tensor<64x16xi8>)
    -> tensor<4x16xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<4x16xi32>
  %r = hip.quant_matmul(%ctx) ins(%a, %b : tensor<4x64xi8>, tensor<64x16xi8>)
         outs(%e : tensor<4x16xi32>) : tensor<4x16xi32>
  return %r : tensor<4x16xi32>
}

// -----

// A ui8 a becomes a - 128 as i8 before the product.

// CHECK-LABEL: func.func @quant_matmul_ui8
// CHECK-NOT: ui8
// CHECK: tosa.custom %{{.*}} {{.*}}operator_name = "unsigned_cast"
// CHECK: %[[C128:.*]] = "tosa.const"() <{values = dense<128> : tensor<1x1xi32>}>
// CHECK: %[[S:.*]] = tosa.sub %{{.*}}, %[[C128]] : (tensor<4x64xi32>, tensor<1x1xi32>) -> tensor<4x64xi32>
// CHECK: %[[A8:.*]] = tosa.cast %[[S]] : (tensor<4x64xi32>) -> tensor<4x64xi8>
// CHECK: tosa.reshape %[[A8]]
// CHECK: tosa.matmul {{.*}} -> tensor<1x4x16xi32>
func.func @quant_matmul_ui8(%a: tensor<4x64xui8>, %b: tensor<64x16xi8>)
    -> tensor<4x16xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<4x16xi32>
  %r = hip.quant_matmul(%ctx) ins(%a, %b : tensor<4x64xui8>, tensor<64x16xi8>)
         outs(%e : tensor<4x16xi32>) : tensor<4x16xi32>
  return %r : tensor<4x16xi32>
}
