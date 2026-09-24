// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The second hip-mlir-opt re-parses and re-verifies the outlined kernels. The
// pass runs on func.func but writes new funcs into the parent module, so the
// nested pass manager never re-verifies the module itself; without this pipe a
// kernel that reads a value from main_graph would go unnoticed here and only
// fail later, in whichever module-level pass happens to run next.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s

// A bare `tensor.empty` init is cloned into the kernel, as it always was.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x4x4xf16>, %[[W:.*]]: tensor<16x8x3x3xf16>, %[[B:.*]]: tensor<16xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x16x4x4xf16>
// CHECK: %[[C:.*]] = hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] :
// CHECK-SAME: outs(%[[E]] : tensor<1x16x4x4xf16>)
// CHECK: return %[[C]]

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x16x4x4xf16>
  // CHECK: hip.rocmlir(%{{.*}}) @rocMlir{{[0-9]+}} ins({{.*}} : tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>, tensor<16xf16>) outs(%[[EMPTY]] : tensor<1x16x4x4xf16>)
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A 1-D convolution has no rocMLIR anchor, so convert-onnx-to-hip widens it to
// 2-D by wrapping the operands *and the init* in `tensor.expand_shape`. The
// init's producer is then a reshape that reads a `tensor.empty` of its own, so
// cloning only the reshape would leave the kernel pointing at a value in
// main_graph -- which func.func, being IsolatedFromAbove, rejects. Both ops
// have to travel, and the empty has to be cloned before the reshape that
// reads it.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x1x4xf16>, %[[W:.*]]: tensor<16x8x1x3xf16>, %[[B:.*]]: tensor<16xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x16x4xf16>
// CHECK: %[[X:.*]] = tensor.expand_shape %[[E]]
// CHECK: %[[C:.*]] = hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] :
// CHECK-SAME: outs(%[[X]] : tensor<1x16x1x4xf16>)
// CHECK: return %[[C]]

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4xf16>,
                      %w: tensor<16x8x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4xf16> {
  %in4 = tensor.expand_shape %in [[0], [1], [2, 3]]
      output_shape [1, 8, 1, 4] : tensor<1x8x4xf16> into tensor<1x8x1x4xf16>
  %w4 = tensor.expand_shape %w [[0], [1], [2, 3]]
      output_shape [16, 8, 1, 3] : tensor<16x8x3xf16> into tensor<16x8x1x3xf16>
  %e = tensor.empty() : tensor<1x16x4xf16>
  %e4 = tensor.expand_shape %e [[0], [1], [2, 3]]
      output_shape [1, 16, 1, 4] : tensor<1x16x4xf16> into tensor<1x16x1x4xf16>
  %c = hip.conv(%ctx) ins(%in4, %w4, %b : tensor<1x8x1x4xf16>,
                                          tensor<16x8x1x3xf16>, tensor<16xf16>)
      outs(%e4 : tensor<1x16x1x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 3],
       pads = [0, 1, 0, 1], strides = [1, 1]} : tensor<1x16x1x4xf16>
  %out = tensor.collapse_shape %c [[0], [1], [2, 3]]
      : tensor<1x16x1x4xf16> into tensor<1x16x4xf16>
  return %out : tensor<1x16x4xf16>
}

// -----

// An init that is a block argument has no producer to clone at all, so it goes
// straight to the argument list.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x4x4xf16>, %[[W:.*]]: tensor<16x8x3x3xf16>, %[[B:.*]]: tensor<16xf16>, %[[KBUF:.*]]: tensor<1x16x4x4xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK: %[[C:.*]] = hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] :
// CHECK-SAME: outs(%[[KBUF]] : tensor<1x16x4x4xf16>)

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[BUF:[a-z0-9_]+]]: tensor<1x16x4x4xf16>) ->
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %buf: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
  // CHECK: hip.rocmlir(%{{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}, %[[BUF]] : tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>, tensor<16xf16>, tensor<1x16x4x4xf16>) outs(%[[BUF]] : tensor<1x16x4x4xf16>)
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%buf : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// When the value an init is built from cannot be rematerialized inside the
// kernel -- here a reshape of a caller-supplied buffer rather than a fresh
// `tensor.empty` -- it is passed in as an extra argument instead. The
// destination still reaches the kernel through that computed value, and the
// buffer keeps the identity the caller gave it.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x1x4xf16>, %[[W:.*]]: tensor<16x8x1x3xf16>, %[[B:.*]]: tensor<16xf16>, %[[KBUF:.*]]: tensor<1x16x4xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK-NOT: tensor.empty
// CHECK: %[[X:.*]] = tensor.expand_shape %[[KBUF]]
// CHECK: %[[C:.*]] = hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] :
// CHECK-SAME: outs(%[[X]] : tensor<1x16x1x4xf16>)

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[BUF:[a-z0-9_]+]]: tensor<1x16x4xf16>) ->
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4xf16>,
                      %w: tensor<16x8x3xf16>, %b: tensor<16xf16>,
                      %buf: tensor<1x16x4xf16>) -> tensor<1x16x4xf16> {
  %in4 = tensor.expand_shape %in [[0], [1], [2, 3]]
      output_shape [1, 8, 1, 4] : tensor<1x8x4xf16> into tensor<1x8x1x4xf16>
  %w4 = tensor.expand_shape %w [[0], [1], [2, 3]]
      output_shape [16, 8, 1, 3] : tensor<16x8x3xf16> into tensor<16x8x1x3xf16>
  %buf4 = tensor.expand_shape %buf [[0], [1], [2, 3]]
      output_shape [1, 16, 1, 4] : tensor<1x16x4xf16> into tensor<1x16x1x4xf16>
  %bufid = tensor.collapse_shape %buf4 [[0], [1], [2, 3]]
      : tensor<1x16x1x4xf16> into tensor<1x16x4xf16>
  %e4 = tensor.expand_shape %bufid [[0], [1], [2, 3]]
      output_shape [1, 16, 1, 4] : tensor<1x16x4xf16> into tensor<1x16x1x4xf16>
  // The computed buffer value is handed to the kernel alongside the data
  // operands instead of re-materializing the reshape chain inside the kernel.
  // CHECK: %[[BUF4:.*]] = tensor.expand_shape %[[BUF]]
  // CHECK: %[[BUFID:.*]] = tensor.collapse_shape %[[BUF4]]
  // CHECK: hip.rocmlir(%{{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}, %[[BUFID]] : tensor<1x8x1x4xf16>, tensor<16x8x1x3xf16>, tensor<16xf16>, tensor<1x16x4xf16>)
  %c = hip.conv(%ctx) ins(%in4, %w4, %b : tensor<1x8x1x4xf16>,
                                          tensor<16x8x1x3xf16>, tensor<16xf16>)
      outs(%e4 : tensor<1x16x1x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 3],
       pads = [0, 1, 0, 1], strides = [1, 1]} : tensor<1x16x1x4xf16>
  %out = tensor.collapse_shape %c [[0], [1], [2, 3]]
      : tensor<1x16x1x4xf16> into tensor<1x16x4xf16>
  return %out : tensor<1x16x4xf16>
}
