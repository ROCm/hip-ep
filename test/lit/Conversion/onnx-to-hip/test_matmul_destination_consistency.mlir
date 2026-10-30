// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-onnx-to-hip --hip-infer-shapes --canonicalize --cse --hip-split-duplicate-dps-inits --hip-resolve-tensor-dims --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize --cse | FileCheck %s

func.func @main_graph(%ctx: !hip.context, %a: tensor<f32>) -> tensor<f32> {
  return %a : tensor<f32>
}

// The destination and the collapsed result use the same batch extent.
// This tests the compiler shape rule, not the runtime's batch-layout support.
// CHECK-LABEL: func.func @matmul_collapse
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[A:.*]]: memref<?x4x8xf32>, %[[B:.*]]: memref<?x8x16xf32>
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[DA:.*]] = memref.dim %[[A]], %[[ZERO]]
// CHECK: %[[DB:.*]] = memref.dim %[[B]], %[[ZERO]]
// CHECK: %[[ISONE:.*]] = arith.cmpi eq, %[[DA]], %[[ONE]] : index
// CHECK: %[[SIZE:.*]] = arith.select %[[ISONE]], %[[DB]], %[[DA]] : index
// CHECK: %[[OUT:.*]] = memref.alloc(%[[SIZE]]) {{.*}}: memref<?x4x16xf32>
// CHECK: hip.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : memref<?x4x8xf32>, memref<?x8x16xf32>) outs(%[[OUT]] : memref<?x4x16xf32>)
// CHECK: %[[FLAT:.*]] = memref.collapse_shape %[[OUT]]
// CHECK: %[[COUNT:.*]] = affine.apply {{.*}}()[%[[SIZE]]]
// CHECK-NOT: arith.select
// CHECK: return %[[FLAT]], %[[COUNT]]
func.func @matmul_collapse(%ctx: !hip.context, %a: tensor<?x4x8xf32>, %b: tensor<?x8x16xf32>) -> (tensor<?xf32>, index) {
  %c0 = arith.constant 0 : index
  %r = "onnx.MatMul"(%a, %b) : (tensor<?x4x8xf32>, tensor<?x8x16xf32>) -> tensor<?x4x16xf32>
  %flat = tensor.collapse_shape %r [[0, 1, 2]] : tensor<?x4x16xf32> into tensor<?xf32>
  %size = tensor.dim %flat, %c0 : tensor<?xf32>
  return %flat, %size : tensor<?xf32>, index
}
