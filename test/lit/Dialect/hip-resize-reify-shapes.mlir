// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_nc
// CHECK-SAME: %[[INPUT:[^,]+]]: tensor<?x?x16x16xf16>
// CHECK-DAG: %[[N:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK-DAG: %[[C:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK: return %[[N]], %[[C]], %{{.*}}, %{{.*}} : index, index, index, index
func.func @dynamic_nc(
    %ctx: !hip.context,
    %input: tensor<?x?x16x16xf16>,
    %output: tensor<?x?x32x24xf16>) -> (index, index, index, index) {
  %result = hip.resize(%ctx)
      ins(%input : tensor<?x?x16x16xf16>)
      outs(%output : tensor<?x?x32x24xf16>)
      {mode = 1, coord_transform = 0, nearest_mode = 0}
      : tensor<?x?x32x24xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %n = tensor.dim %result, %c0 : tensor<?x?x32x24xf16>
  %c = tensor.dim %result, %c1 : tensor<?x?x32x24xf16>
  %h = tensor.dim %result, %c2 : tensor<?x?x32x24xf16>
  %w = tensor.dim %result, %c3 : tensor<?x?x32x24xf16>
  return %n, %c, %h, %w : index, index, index, index
}
