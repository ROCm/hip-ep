// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The second hip-mlir-opt re-parses and re-verifies the outlined kernels. The
// pass runs on func.func but writes new funcs into the parent module, so the
// nested pass manager never re-verifies the module itself; without this pipe a
// kernel that reads a value from main_graph would go unnoticed here and only
// fail later, in whichever module-level pass happens to run next.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s

// Every outlined kernel has to convert: hip-to-tosa fails the whole kernel on
// one op it marks illegal but cannot rewrite.
// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s \
// RUN:   | hip-mlir-opt --split-input-file --convert-hip-to-tosa > /dev/null

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
// kernel -- here a caller-supplied buffer rather than a fresh `tensor.empty` --
// it is passed in as an extra argument instead. The destination still reaches
// the kernel, and the buffer keeps the identity the caller gave it.

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
  %e4 = tensor.expand_shape %buf [[0], [1], [2, 3]]
      output_shape [1, 16, 1, 4] : tensor<1x16x4xf16> into tensor<1x16x1x4xf16>
  // The buffer is handed to the kernel alongside the data operands.
  // CHECK: hip.rocmlir(%{{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}, %[[BUF]] : tensor<1x8x1x4xf16>, tensor<16x8x1x3xf16>, tensor<16xf16>, tensor<1x16x4xf16>)
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

// A rank-0 operand cannot be a kernel argument: rocMLIR turns each argument
// into a `rock.transforms_to_ptr`, which needs a coordinate to linearize, and
// a scalar has none. The Relu zero is an inline `hip.constant`, so it is
// cloned into the kernel and lowers to a `tosa.const` there.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x4x4xf16>, %[[W:.*]]: tensor<16x8x3x3xf16>, %[[B:.*]]: tensor<16xf16>)
// CHECK: %[[ZERO:.*]] = hip.constant
// CHECK-SAME: value = dense<0.000000e+00> : tensor<f16>
// CHECK: hip.max
// CHECK-SAME: %[[ZERO]]

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %zero = hip.constant {value = dense<0.000000e+00> : tensor<f16>} : tensor<f16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %maxInit = tensor.empty() : tensor<1x16x4x4xf16>
  // The scalar does not appear in the dispatch operand list.
  // CHECK: hip.rocmlir
  // CHECK-SAME: tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>, tensor<16xf16>) outs
  %r = hip.max(%ctx) ins(%c, %zero : tensor<1x16x4x4xf16>, tensor<f16>)
      outs(%maxInit : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}

// -----

// The same applies to any pointwise op, not just the min/max pair Relu and
// Clip lower to. A residual block scales its sum by a splat before adding it
// back, which reaches the pass as `hip.mul` against a rank-0 constant.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x4x4xf16>, %[[W:.*]]: tensor<16x8x3x3xf16>, %[[B:.*]]: tensor<16xf16>)
// CHECK: %[[SCALE:.*]] = hip.constant
// CHECK-SAME: value = dense<1.999510e-01> : tensor<f16>
// CHECK: hip.mul
// CHECK-SAME: %[[SCALE]]

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %scale = hip.constant {value = dense<1.999510e-01> : tensor<f16>} : tensor<f16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %mulInit = tensor.empty() : tensor<1x16x4x4xf16>
  // CHECK: hip.rocmlir
  // CHECK-SAME: tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>, tensor<16xf16>) outs
  %r = hip.mul(%ctx) ins(%c, %scale : tensor<1x16x4x4xf16>, tensor<f16>)
      outs(%mulInit : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}

// -----

// Only inline constants are cloned. A bound the caller supplies at runtime has
// no value to rematerialize, so it stays a kernel argument -- but crosses the
// boundary as tensor<1xT>. A rank-0 argument reaches rock.transforms_to_ptr
// with no coordinate to linearize; tensor<1xT> carries the single index 0.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<16xf16>, %[[KBOUND:.*]]: tensor<1xf16>)
// CHECK: hip.max
// CHECK-SAME: %[[KBOUND]]

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[BOUND:.[a-z0-9_]+]]: tensor<f16>) ->
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %bound: tensor<f16>) -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %maxInit = tensor.empty() : tensor<1x16x4x4xf16>
  // The graph keeps its rank-0 value; only what crosses into the kernel is
  // reshaped.
  // CHECK: %[[R1:.*]] = tensor.reshape %[[BOUND]]
  // CHECK-SAME: (tensor<f16>, tensor<1xindex>) -> tensor<1xf16>
  // CHECK: hip.rocmlir
  // CHECK-SAME: %[[R1]] : tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>, tensor<16xf16>, tensor<1xf16>) outs
  %r = hip.max(%ctx) ins(%c, %bound : tensor<1x16x4x4xf16>, tensor<f16>)
      outs(%maxInit : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}

// -----

// A pointwise chain with no conv, matmul, or gemm is one rocMLIR kernel.
// PR 122 compiles a rock.kernel that contains only pointwise ops.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<1x4x4xf16>, %[[B:.*]]: tensor<1x4x4xf16>, %[[C:.*]]: tensor<1x4x4xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK: hip.mul(%[[CTX]])
// CHECK: hip.add(%[[CTX]])
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4xf16>,
                      %b: tensor<1x4x4xf16>, %c: tensor<1x4x4xf16>)
    -> tensor<1x4x4xf16> {
  %mulEmpty = tensor.empty() : tensor<1x4x4xf16>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<1x4x4xf16>, tensor<1x4x4xf16>)
      outs(%mulEmpty : tensor<1x4x4xf16>) -> tensor<1x4x4xf16>
  %addEmpty = tensor.empty() : tensor<1x4x4xf16>
  // One dispatch, and the chain is not left in main_graph.
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK-NOT: hip.mul
  // CHECK-NOT: hip.add
  // CHECK: return
  %r = hip.add(%ctx) ins(%m, %c : tensor<1x4x4xf16>, tensor<1x4x4xf16>)
      outs(%addEmpty : tensor<1x4x4xf16>) -> tensor<1x4x4xf16>
  return %r : tensor<1x4x4xf16>
}

// -----

// A conv followed by hip.add stays on the anchor pattern: one kernel, not a
// convolution kernel plus a separate pointwise kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.conv
// CHECK: hip.add
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %other: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %addEmpty = tensor.empty() : tensor<1x16x4x4xf16>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK-NOT: hip.add
  // CHECK-NOT: hip.conv
  // CHECK: return
  %r = hip.add(%ctx) ins(%c, %other : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%addEmpty : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}

// -----

// MIGraphX reshaper_names: a transpose between a conv and a pointwise op is
// part of the same rocMLIR kernel, not a separate HIP kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.conv
// CHECK: hip.transpose
// CHECK: hip.add
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %other: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %transEmpty = tensor.empty() : tensor<1x4x4x16xf16>
  %t = hip.transpose(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%transEmpty : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]}
      : tensor<1x4x4x16xf16>
  %addEmpty = tensor.empty() : tensor<1x4x4x16xf16>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK-NOT: hip.transpose
  // CHECK-NOT: hip.add
  // CHECK: return
  %r = hip.add(%ctx) ins(%t, %other : tensor<1x4x4x16xf16>, tensor<1x4x4x16xf16>)
      outs(%addEmpty : tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16>
  return %r : tensor<1x4x4x16xf16>
}

// -----

// A reshaper between two pointwise ops, with no conv or gemm, stays in that
// one kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK: hip.transpose
// CHECK: hip.add
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4xf16>,
                      %b: tensor<1x4xf16>, %c: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %mulEmpty = tensor.empty() : tensor<1x4xf16>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<1x4xf16>, tensor<1x4xf16>)
      outs(%mulEmpty : tensor<1x4xf16>) -> tensor<1x4xf16>
  %transEmpty = tensor.empty() : tensor<4x1xf16>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x4xf16>)
      outs(%transEmpty : tensor<4x1xf16>) {perm = [1, 0]} : tensor<4x1xf16>
  %addEmpty = tensor.empty() : tensor<4x1xf16>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK-NOT: hip.transpose
  // CHECK-NOT: hip.mul
  // CHECK-NOT: hip.add
  // CHECK: return
  %r = hip.add(%ctx) ins(%t, %c : tensor<4x1xf16>, tensor<4x1xf16>)
      outs(%addEmpty : tensor<4x1xf16>) -> tensor<4x1xf16>
  return %r : tensor<4x1xf16>
}

// -----

// A cast of a value that is not the conv can sit in front of the add that
// consumes the conv. The conv stays the parent of the add and the output
// cast. The input cast is not that parent: it is its own pointwise kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.cast
// CHECK-NOT: hip.conv
// CHECK-NOT: hip.add
// CHECK: return

// CHECK-LABEL: func.func @rocMlir
// CHECK-NOT: hip.cast
// CHECK: hip.conv
// CHECK: hip.add
// CHECK: hip.cast
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %skip: tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32> {
  %castEmpty = tensor.empty() : tensor<1x16x4x4xf16>
  %cast = hip.cast(%ctx) ins(%skip : tensor<1x16x4x4xf32>)
      outs(%castEmpty : tensor<1x16x4x4xf16>) {to = 10 : i64}
      : tensor<1x16x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %addEmpty = tensor.empty() : tensor<1x16x4x4xf16>
  %add = hip.add(%ctx) ins(%c, %cast : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%addEmpty : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %outEmpty = tensor.empty() : tensor<1x16x4x4xf32>
  // Two dispatches: the input cast, then the conv with its epilogue.
  // CHECK: hip.rocmlir
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK-NOT: hip.cast
  // CHECK-NOT: hip.conv
  // CHECK-NOT: hip.add
  // CHECK: return
  %r = hip.cast(%ctx) ins(%add : tensor<1x16x4x4xf16>)
      outs(%outEmpty : tensor<1x16x4x4xf32>) {to = 1 : i64}
      : tensor<1x16x4x4xf32>
  return %r : tensor<1x16x4x4xf32>
}

// -----

// A ui8 boolean has no hip-to-tosa lowering, so hip.less stays on the HIP
// path. The chain stops before it and the hip.mul still compiles.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK-NOT: hip.less
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<4xf16>,
                      %b: tensor<4xf16>) -> tensor<4xui8> {
  %mulEmpty = tensor.empty() : tensor<4xf16>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<4xf16>, tensor<4xf16>)
      outs(%mulEmpty : tensor<4xf16>) -> tensor<4xf16>
  %lessEmpty = tensor.empty() : tensor<4xui8>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK: hip.less
  // CHECK: return
  %r = hip.less(%ctx) ins(%m, %b : tensor<4xf16>, tensor<4xf16>)
      outs(%lessEmpty : tensor<4xui8>) : tensor<4xui8>
  return %r : tensor<4xui8>
}

// -----

// tensor.reshape has no lowering inside a rock kernel, so it splits the chain
// into two kernels and stays in main_graph.

// CHECK-LABEL: func.func @rocMlir
// CHECK-NOT: tensor.reshape
// CHECK: return
// CHECK-LABEL: func.func @rocMlir
// CHECK-NOT: tensor.reshape
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x16xf16>,
                      %b: tensor<1x16xf16>, %shape: tensor<2xindex>,
                      %c: tensor<4x4xf16>) -> tensor<4x4xf16> {
  %mulEmpty = tensor.empty() : tensor<1x16xf16>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<1x16xf16>, tensor<1x16xf16>)
      outs(%mulEmpty : tensor<1x16xf16>) -> tensor<1x16xf16>
  %rs = tensor.reshape %m(%shape)
      : (tensor<1x16xf16>, tensor<2xindex>) -> tensor<4x4xf16>
  %addEmpty = tensor.empty() : tensor<4x4xf16>
  // CHECK: hip.rocmlir
  // CHECK: tensor.reshape
  // CHECK: hip.rocmlir
  // CHECK: return
  %r = hip.add(%ctx) ins(%rs, %c : tensor<4x4xf16>, tensor<4x4xf16>)
      outs(%addEmpty : tensor<4x4xf16>) -> tensor<4x4xf16>
  return %r : tensor<4x4xf16>
}

// -----

// A dynamic consumer ends the chain: the static hip.mul is a kernel, and the
// dynamic hip.add after it stays on the HIP path.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK-NOT: hip.add
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<16xf16>,
                      %b: tensor<16xf16>, %d: index,
                      %c: tensor<?xf16>) -> tensor<?xf16> {
  %mulEmpty = tensor.empty() : tensor<16xf16>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<16xf16>, tensor<16xf16>)
      outs(%mulEmpty : tensor<16xf16>) -> tensor<16xf16>
  %addEmpty = tensor.empty(%d) : tensor<?xf16>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK: hip.add
  // CHECK: return
  %r = hip.add(%ctx) ins(%m, %c : tensor<16xf16>, tensor<?xf16>)
      outs(%addEmpty : tensor<?xf16>) -> tensor<?xf16>
  return %r : tensor<?xf16>
}

// -----

// A rank-0 result stays on the HIP path: only kernel arguments are promoted
// to tensor<1xT>, so a cloned scalar op would mix ranks.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.exp
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %x: tensor<f32>) -> tensor<f32> {
  %e = tensor.empty() : tensor<f32>
  %r = hip.exp(%ctx) ins(%x : tensor<f32>) outs(%e : tensor<f32>) : tensor<f32>
  return %r : tensor<f32>
}

// -----

// tosa.intdiv takes only i32 and i64, so an i16 hip.div stays on the HIP path
// while the hip.mul before it still compiles.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK-NOT: hip.div
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<4xi16>,
                      %b: tensor<4xi16>) -> tensor<4xi16> {
  %mulEmpty = tensor.empty() : tensor<4xi16>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<4xi16>, tensor<4xi16>)
      outs(%mulEmpty : tensor<4xi16>) -> tensor<4xi16>
  %divEmpty = tensor.empty() : tensor<4xi16>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK: hip.div
  // CHECK: return
  %r = hip.div(%ctx) ins(%m, %b : tensor<4xi16>, tensor<4xi16>)
      outs(%divEmpty : tensor<4xi16>) : tensor<4xi16>
  return %r : tensor<4xi16>
}

// -----

// hip-to-tosa has no floating-point hip.mod, so it stays on the HIP path.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK-NOT: hip.mod
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<4xf32>,
                      %b: tensor<4xf32>) -> tensor<4xf32> {
  %mulEmpty = tensor.empty() : tensor<4xf32>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%mulEmpty : tensor<4xf32>) -> tensor<4xf32>
  %modEmpty = tensor.empty() : tensor<4xf32>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK: hip.mod
  // CHECK: return
  %r = hip.mod(%ctx) ins(%m, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%modEmpty : tensor<4xf32>) {fmod = 1 : i64} : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// An integer hip.mod with fmod = 1 is outside its contract, which hip-to-tosa
// rejects, so it stays on the HIP path too.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK-NOT: hip.mod
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<4xi32>,
                      %b: tensor<4xi32>) -> tensor<4xi32> {
  %mulEmpty = tensor.empty() : tensor<4xi32>
  %m = hip.mul(%ctx) ins(%a, %b : tensor<4xi32>, tensor<4xi32>)
      outs(%mulEmpty : tensor<4xi32>) -> tensor<4xi32>
  %modEmpty = tensor.empty() : tensor<4xi32>
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.rocmlir
  // CHECK: hip.mod
  // CHECK: return
  %r = hip.mod(%ctx) ins(%m, %b : tensor<4xi32>, tensor<4xi32>)
      outs(%modEmpty : tensor<4xi32>) {fmod = 1 : i64} : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// A hip.mod left on the HIP path between a pointwise op and a conv epilogue
// splits them: the hip.mul is its own kernel and the conv takes the hip.add.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.mul
// CHECK-NOT: hip.mod
// CHECK: return
// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.conv
// CHECK-NOT: hip.mod
// CHECK: hip.add
// CHECK: return

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %x: tensor<1x16x4x4xf16>, %y: tensor<1x16x4x4xf16>)
    -> tensor<1x16x4x4xf16> {
  %mulEmpty = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%x, %y : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%mulEmpty : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %modEmpty = tensor.empty() : tensor<1x16x4x4xf16>
  %mod = hip.mod(%ctx) ins(%m, %y : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%modEmpty : tensor<1x16x4x4xf16>) {fmod = 1 : i64}
      : tensor<1x16x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %addEmpty = tensor.empty() : tensor<1x16x4x4xf16>
  // CHECK: hip.rocmlir
  // CHECK: hip.mod
  // CHECK: hip.rocmlir
  // CHECK-NOT: hip.conv
  // CHECK-NOT: hip.add
  // CHECK: return
  %r = hip.add(%ctx) ins(%c, %mod : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%addEmpty : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}
