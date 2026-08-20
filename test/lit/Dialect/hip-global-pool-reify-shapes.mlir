// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_spatial
// CHECK-SAME: %[[INPUT:[^,]+]]: tensor<?x?x?x?xf32>
// CHECK-DAG: %[[N:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK-DAG: %[[C:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: return %[[N]], %[[C]], %{{.*}}, %{{.*}} : index, index, index, index
func.func @dynamic_spatial(
    %ctx: !hip.context,
    %input: tensor<?x?x?x?xf32>,
    %output: tensor<?x?x?x?xf32>) -> (index, index, index, index) {
  %result = hip.global_pool(%ctx)
      ins(%input : tensor<?x?x?x?xf32>)
      outs(%output : tensor<?x?x?x?xf32>)
      {mode = 0 : i64}
      : tensor<?x?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %n = tensor.dim %result, %c0 : tensor<?x?x?x?xf32>
  %c = tensor.dim %result, %c1 : tensor<?x?x?x?xf32>
  %h = tensor.dim %result, %c2 : tensor<?x?x?x?xf32>
  %w = tensor.dim %result, %c3 : tensor<?x?x?x?xf32>
  return %n, %c, %h, %w : index, index, index, index
}
