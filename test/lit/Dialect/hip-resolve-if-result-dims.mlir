// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// `--hip-resolve-if-result-dims` on the post-conversion spelling of decoder
// head reshapes: a size read back from a shape tensor built as
// where(equal(target, 0), shape, target) (Reshape's "0 copies the input dim"),
// and a -1 entry inferred by division. Both must become caller-side host
// arithmetic on the captured dims, with no readback, and the division must be
// guarded because the untaken branch's size is computed too.

// RUN: hip-mlir-opt --hip-resolve-if-result-dims --cse %s | FileCheck %s

module {
  func.func @main_graph(%ctx: !hip.context, %c: i1, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @then_n0 else @else_n1 {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @then_n0 else @else_n1 {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @then_n0(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n64 = arith.index_cast %n : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %n64, %c8, %c64 : tensor<3xi64>
    %target = arith.constant dense<[0, 8, 64]> : tensor<3xi64>
    %zeros = arith.constant dense<0> : tensor<3xi64>
    %eq_init = tensor.empty() : tensor<3xui8>
    %eq = hip.equal(%ctx) ins(%target, %zeros : tensor<3xi64>, tensor<3xi64>)
        outs(%eq_init : tensor<3xui8>) : tensor<3xui8>
    %w_init = tensor.empty() : tensor<3xi64>
    %w = hip.where(%ctx) ins(%eq, %shape, %target
        : tensor<3xui8>, tensor<3xi64>, tensor<3xi64>)
        outs(%w_init : tensor<3xi64>) : tensor<3xi64>
    %first = tensor.extract_slice %w[0] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %scalar = tensor.collapse_shape %first [] : tensor<1xi64> into tensor<i64>
    %rows = hip.readback_scalar(%ctx, %scalar : tensor<i64>) -> i64
    %rows_idx = arith.index_cast %rows : i64 to index
    %y = tensor.reshape %x(%w)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    %out_init = tensor.empty(%rows_idx) : tensor<?x8x64xf32>
    %out = hip.neg(%ctx) ins(%y : tensor<?x8x64xf32>)
        outs(%out_init : tensor<?x8x64xf32>) : tensor<?x8x64xf32>
    return %out : tensor<?x8x64xf32>
  }

  // -1 inference: rows = (n * m) / m with a run-time divisor.
  func.func private @else_n1(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %m = tensor.dim %x, %c1 : tensor<?x?xf32>
    %total = arith.muli %n, %m : index
    %rows = arith.divsi %total, %m : index
    %rows_i64 = arith.index_cast %rows : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %rows_i64, %c8, %c64 : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK-NOT: hip.readback_scalar
// CHECK: %[[N:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[THEN:.*]] = arith.select %{{.*}}, %[[N]], %{{.*}} : index
// CHECK: %[[M:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[TOTAL:.*]] = arith.muli %[[N]], %[[M]] : index
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[M]], %{{.*}} : index
// CHECK: %[[DIVISOR:.*]] = arith.select %[[IS_ZERO]], %{{.*}}, %[[M]] : index
// CHECK: %[[ELSE:.*]] = arith.divsi %[[TOTAL]], %[[DIVISOR]] : index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %{{.*}}, %[[ELSE]] : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>
// CHECK-NOT: hip.if_result_dim
// CHECK: hip.if
// CHECK-LABEL: func.func private @then_n0
// CHECK: hip.readback_scalar
