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
// reads it. The operand reshapes are absorbed too, so the kernel reads the
// rank-3 tensors as main_graph holds them.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x4xf16>, %[[W:.*]]: tensor<16x8x3xf16>, %[[B:.*]]: tensor<16xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK-DAG: %[[IN4:.*]] = tensor.expand_shape %[[IN]]
// CHECK-DAG: %[[W4:.*]] = tensor.expand_shape %[[W]]
// CHECK-DAG: %[[E:.*]] = tensor.empty() : tensor<1x16x4xf16>
// CHECK-DAG: %[[X:.*]] = tensor.expand_shape %[[E]]
// CHECK: %[[C:.*]] = hip.conv(%[[CTX]]) ins(%[[IN4]], %[[W4]], %[[B]] :
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
// CHECK-SAME: (%[[IN:.*]]: tensor<1x8x4xf16>, %[[W:.*]]: tensor<16x8x3xf16>, %[[B:.*]]: tensor<16xf16>, %[[KBUF:.*]]: tensor<1x16x4xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK-NOT: tensor.empty
// CHECK-DAG: %[[IN4:.*]] = tensor.expand_shape %[[IN]]
// CHECK-DAG: %[[W4:.*]] = tensor.expand_shape %[[W]]
// CHECK-DAG: %[[X:.*]] = tensor.expand_shape %[[KBUF]]
// CHECK: %[[C:.*]] = hip.conv(%[[CTX]]) ins(%[[IN4]], %[[W4]], %[[B]] :
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
  // CHECK: hip.rocmlir(%{{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}, %[[BUF]] : tensor<1x8x4xf16>, tensor<16x8x3xf16>, tensor<16xf16>, tensor<1x16x4xf16>)
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

// An NHWC convolution arrives wrapped in layout transposes, and the graph
// around it transposes back for the residual add. Every transpose becomes part
// of the kernel's input or output layout, so the kernel reads the NHWC
// activation and filter directly and main_graph keeps no hip.transpose. Each
// absorbed view's input takes the argument slot of the value it replaced.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<1x4x4x8xf16>, %[[W:[a-z0-9]+]]: tensor<16x3x3x8xf16>, %{{[a-z0-9]+}}: tensor<16xf16>, %[[R:[a-z0-9]+]]: tensor<1x16x4x4xf16>) -> tensor<1x4x4x16xf16>
// CHECK-DAG: hip.transpose({{.*}}) ins(%[[X]] : tensor<1x4x4x8xf16>)
// CHECK-DAG: hip.transpose({{.*}}) ins(%[[W]] : tensor<16x3x3x8xf16>)
// CHECK: hip.conv
// CHECK: hip.transpose
// CHECK: hip.add
// CHECK-SAME: %[[R]]
// CHECK: %[[OUT:.*]] = hip.transpose
// CHECK: return %[[OUT]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: hip.rocmlir
// CHECK-NOT: hip.transpose
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x4x8xf16>,
                      %w: tensor<16x3x3x8xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x4x4xf16>) -> tensor<1x4x4x16xf16> {
  %xe = tensor.empty() : tensor<1x8x4x4xf16>
  %xn = hip.transpose(%ctx) ins(%x : tensor<1x4x4x8xf16>)
      outs(%xe : tensor<1x8x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf16>
  %we = tensor.empty() : tensor<16x8x3x3xf16>
  %wn = hip.transpose(%ctx) ins(%w : tensor<16x3x3x8xf16>)
      outs(%we : tensor<16x8x3x3xf16>) {perm = [0, 3, 1, 2]} : tensor<16x8x3x3xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%xn, %wn, %b : tensor<1x8x4x4xf16>,
                                         tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ce = tensor.empty() : tensor<1x4x4x16xf16>
  %cl = hip.transpose(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%ce : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  %te = tensor.empty() : tensor<1x16x4x4xf16>
  %t = hip.transpose(%ctx) ins(%cl : tensor<1x4x4x16xf16>)
      outs(%te : tensor<1x16x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%t, %r : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %oe = tensor.empty() : tensor<1x4x4x16xf16>
  %o = hip.transpose(%ctx) ins(%a : tensor<1x16x4x4xf16>)
      outs(%oe : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  return %o : tensor<1x4x4x16xf16>
}

// -----

// A unary op on the activation is a prologue of the kernel when nothing else
// reads it: the kernel takes the f32 tensor and casts as it loads.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<1x4x4x8xf32>,
// CHECK: hip.cast({{.*}}) ins(%[[X]] : tensor<1x4x4x8xf32>)
// CHECK: hip.transpose
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.transpose
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x4x8xf32>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %he = tensor.empty() : tensor<1x4x4x8xf16>
  %h = hip.cast(%ctx) ins(%x : tensor<1x4x4x8xf32>)
      outs(%he : tensor<1x4x4x8xf16>) {to = 10 : i64} : tensor<1x4x4x8xf16>
  %xe = tensor.empty() : tensor<1x8x4x4xf16>
  %xn = hip.transpose(%ctx) ins(%h : tensor<1x4x4x8xf16>)
      outs(%xe : tensor<1x8x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%xn, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A transpose with a reader outside the kernel would keep running for that
// reader, so absorbing it saves nothing. It stays outside, and so does the
// cast that feeds it.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<1x8x4x4xf16>,
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.transpose
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[CAST:.*]] = hip.cast
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[CAST]]
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @rocMlir{{[0-9]*}} ins(%[[T]], {{.*}} :
// CHECK: return %[[K]], %[[T]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x4x8xf32>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> (tensor<1x16x4x4xf16>, tensor<1x8x4x4xf16>) {
  %he = tensor.empty() : tensor<1x4x4x8xf16>
  %h = hip.cast(%ctx) ins(%x : tensor<1x4x4x8xf32>)
      outs(%he : tensor<1x4x4x8xf16>) {to = 10 : i64} : tensor<1x4x4x8xf16>
  %xe = tensor.empty() : tensor<1x8x4x4xf16>
  %xn = hip.transpose(%ctx) ins(%h : tensor<1x4x4x8xf16>)
      outs(%xe : tensor<1x8x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%xn, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c, %xn : tensor<1x16x4x4xf16>, tensor<1x8x4x4xf16>
}

// -----

// The filter never takes a pointwise prologue: the cast would rerun in every
// workgroup that loads the filter. It stays a separate op.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<16x8x3x3xf16>
// CHECK-NOT: hip.cast

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.cast
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %we = tensor.empty() : tensor<16x8x3x3xf16>
  %wh = hip.cast(%ctx) ins(%w : tensor<16x8x3x3xf32>)
      outs(%we : tensor<16x8x3x3xf16>) {to = 10 : i64} : tensor<16x8x3x3xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %wh, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A matmul takes the transposes on both sides, as a convolution does: the
// kernel reads the untransposed A and writes the transposed result.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<16x8xf16>, %{{.*}}: tensor<16x32xf16>) -> tensor<32x8xf16>
// CHECK: %[[AT:.*]] = hip.transpose({{.*}}) ins(%[[A]]
// CHECK: %[[M:.*]] = hip.matmul({{.*}}) ins(%[[AT]],
// CHECK: %[[O:.*]] = hip.transpose({{.*}}) ins(%[[M]]
// CHECK: return %[[O]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: hip.rocmlir
// CHECK-NOT: hip.transpose
func.func @main_graph(%ctx: !hip.context, %a: tensor<16x8xf16>,
                      %b: tensor<16x32xf16>) -> tensor<32x8xf16> {
  %ae = tensor.empty() : tensor<8x16xf16>
  %at = hip.transpose(%ctx) ins(%a : tensor<16x8xf16>)
      outs(%ae : tensor<8x16xf16>) {perm = [1, 0]} : tensor<8x16xf16>
  %e = tensor.empty() : tensor<8x32xf16>
  %m = hip.matmul(%ctx) ins(%at, %b : tensor<8x16xf16>, tensor<16x32xf16>)
      outs(%e : tensor<8x32xf16>) : tensor<8x32xf16>
  %oe = tensor.empty() : tensor<32x8xf16>
  %o = hip.transpose(%ctx) ins(%m : tensor<8x32xf16>)
      outs(%oe : tensor<32x8xf16>) {perm = [1, 0]} : tensor<32x8xf16>
  return %o : tensor<32x8xf16>
}

// -----

// A group split folded back into channels is a reshape on the activation. The
// kernel reads the grouped tensor and collapses it as it loads.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[XG:[a-z0-9]+]]: tensor<1x2x4x4x4xf16>,
// CHECK: tensor.collapse_shape %[[XG]]
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.collapse_shape
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %xg: tensor<1x2x4x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %x = tensor.collapse_shape %xg [[0], [1, 2], [3], [4]]
      : tensor<1x2x4x4x4xf16> into tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A Split into a conv is a unit-stride channel slice of the activation. The
// kernel takes the whole tensor and reads only its half.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[XW:[a-z0-9]+]]: tensor<1x16x4x4xf16>,
// CHECK: tensor.extract_slice %[[XW]][0, 8, 0, 0] [1, 8, 4, 4] [1, 1, 1, 1]
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %xw: tensor<1x16x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %x = tensor.extract_slice %xw[0, 8, 0, 0] [1, 8, 4, 4] [1, 1, 1, 1]
      : tensor<1x16x4x4xf16> to tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// HipToTosa has no lowering for a strided slice, so it stays in main_graph.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<1x8x4x4xf16>,
// CHECK-NOT: tensor.extract_slice

// CHECK-LABEL: func.func @main_graph
// CHECK: tensor.extract_slice
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %xw: tensor<1x16x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %x = tensor.extract_slice %xw[0, 0, 0, 0] [1, 8, 4, 4] [1, 2, 1, 1]
      : tensor<1x16x4x4xf16> to tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A filter stored flat is reshaped in the kernel. Views are free on the
// filter, unlike a pointwise prologue.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<1x8x4x4xf16>, %[[WF:[a-z0-9]+]]: tensor<16x72xf16>,
// CHECK: tensor.expand_shape %[[WF]]
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.expand_shape
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %wf: tensor<16x72xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %w = tensor.expand_shape %wf [[0], [1, 2, 3]] output_shape [16, 8, 3, 3]
      : tensor<16x72xf16> into tensor<16x8x3x3xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A per-channel epilogue operand broadcast by hip.expand is read in its
// compact form. The constant target shape is cloned with the expand, so it is
// not a kernel argument.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<16xf16>, %[[S:[a-z0-9]+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x4x4xf16>
// CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 16, 4, 4]> : tensor<4xi64>
// CHECK-DAG: hip.conv
// CHECK: %[[SB:.*]] = hip.expand({{.*}}) ins(%[[S]], %[[SHAPE]]
// CHECK: hip.mul({{.*}}) ins(%{{.*}}, %[[SB]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.expand
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %s: tensor<1x16x1x1xf16>) -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %shape = arith.constant dense<[1, 16, 4, 4]> : tensor<4xi64>
  %se = tensor.empty() : tensor<1x16x4x4xf16>
  %sb = hip.expand(%ctx) ins(%s, %shape : tensor<1x16x1x1xf16>, tensor<4xi64>)
      outs(%se : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%c, %sb : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%me : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}

// -----

// A target shape computed at run time cannot be cloned into the kernel, so the
// broadcast stays in main_graph and the kernel reads its full result.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<16xf16>, %{{[a-z0-9]+}}: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
// CHECK-NOT: hip.expand

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.expand
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %s: tensor<1x16x1x1xf16>, %shape: tensor<4xi64>)
    -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %se = tensor.empty() : tensor<1x16x4x4xf16>
  %sb = hip.expand(%ctx) ins(%s, %shape : tensor<1x16x1x1xf16>, tensor<4xi64>)
      outs(%se : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%c, %sb : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%me : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}

// -----

// Views stack on an epilogue operand as on the anchor's: a flat NHWC residual
// is expanded and transposed inside the kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<16xf16>, %[[RF:[a-z0-9]+]]: tensor<1x16x16xf16>) -> tensor<1x16x4x4xf16>
// CHECK: hip.conv
// CHECK: %[[RH:.*]] = tensor.expand_shape %[[RF]]
// CHECK: %[[RT:.*]] = hip.transpose({{.*}}) ins(%[[RH]]
// CHECK: hip.add({{.*}}) ins(%{{.*}}, %[[RT]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: hip.transpose
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %rf: tensor<1x16x16xf16>) -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %rh = tensor.expand_shape %rf [[0], [1, 2], [3]] output_shape [1, 4, 4, 16]
      : tensor<1x16x16xf16> into tensor<1x4x4x16xf16>
  %re = tensor.empty() : tensor<1x16x4x4xf16>
  %rt = hip.transpose(%ctx) ins(%rh : tensor<1x4x4x16xf16>)
      outs(%re : tensor<1x16x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%c, %rt : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %a : tensor<1x16x4x4xf16>
}

// -----

// A transposed conv output that is both returned and read as another kernel's
// residual stays the first kernel's epilogue. The second kernel does not take
// the transpose as a view: that would leave the transpose running for the
// other reader and stop the first conv from growing through it.

// CHECK-LABEL: func.func @rocMlir
// CHECK-LABEL: func.func @rocMlir
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: %[[K1:.*]] = hip.rocmlir
// CHECK-SAME: outs({{.*}} : tensor<1x4x4x16xf16>)
// CHECK-NOT: hip.transpose
// CHECK: %[[K2:.*]] = hip.rocmlir({{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}, %[[K1]] :
// CHECK: return %[[K2]], %[[K1]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %x2: tensor<1x8x4x4xf16>, %w: tensor<16x8x3x3xf16>,
                      %b: tensor<16xf16>)
    -> (tensor<1x4x4x16xf16>, tensor<1x4x4x16xf16>) {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %te = tensor.empty() : tensor<1x4x4x16xf16>
  %t = hip.transpose(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%te : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  %e2 = tensor.empty() : tensor<1x16x4x4xf16>
  %c2 = hip.conv(%ctx) ins(%x2, %w, %b : tensor<1x8x4x4xf16>,
                                         tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e2 : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %t2e = tensor.empty() : tensor<1x4x4x16xf16>
  %t2 = hip.transpose(%ctx) ins(%c2 : tensor<1x16x4x4xf16>)
      outs(%t2e : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  %ae = tensor.empty() : tensor<1x4x4x16xf16>
  %a = hip.add(%ctx) ins(%t2, %t : tensor<1x4x4x16xf16>, tensor<1x4x4x16xf16>)
      outs(%ae : tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16>
  return %a, %t : tensor<1x4x4x16xf16>, tensor<1x4x4x16xf16>
}

// -----

// The epilogue runs through a reshape of the conv result: a 1-D conv widened
// to 2-D is collapsed back before its residual add, and the add joins the
// kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: %[[R:[a-z0-9]+]]: tensor<1x16x4xf16>) -> tensor<1x16x4xf16>
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[V:.*]] = tensor.collapse_shape %[[C]]
// CHECK: hip.add({{.*}}) ins(%[[V]], %[[R]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: hip.add
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x1x4xf16>,
                      %w: tensor<16x8x1x3xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x4xf16>) -> tensor<1x16x4xf16> {
  %e = tensor.empty() : tensor<1x16x1x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x1x4xf16>,
                                       tensor<16x8x1x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x1x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 3],
       pads = [0, 1, 0, 1], strides = [1, 1]} : tensor<1x16x1x4xf16>
  %v = tensor.collapse_shape %c [[0], [1], [2, 3]]
      : tensor<1x16x1x4xf16> into tensor<1x16x4xf16>
  %ae = tensor.empty() : tensor<1x16x4xf16>
  %a = hip.add(%ctx) ins(%v, %r : tensor<1x16x4xf16>, tensor<1x16x4xf16>)
      outs(%ae : tensor<1x16x4xf16>) -> tensor<1x16x4xf16>
  return %a : tensor<1x16x4xf16>
}

// -----

// Views stack on the way out as on the way in: the NCHW result is transposed
// to NHWC and flattened to a token sequence before the add.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x16xf16>
// CHECK: hip.conv
// CHECK: hip.transpose
// CHECK: tensor.collapse_shape
// CHECK: hip.add

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK-NOT: tensor.collapse_shape
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x16xf16>) -> tensor<1x16x16xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %te = tensor.empty() : tensor<1x4x4x16xf16>
  %t = hip.transpose(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%te : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  %v = tensor.collapse_shape %t [[0], [1, 2], [3]]
      : tensor<1x4x4x16xf16> into tensor<1x16x16xf16>
  %ae = tensor.empty() : tensor<1x16x16xf16>
  %a = hip.add(%ctx) ins(%v, %r : tensor<1x16x16xf16>, tensor<1x16x16xf16>)
      outs(%ae : tensor<1x16x16xf16>) -> tensor<1x16x16xf16>
  return %a : tensor<1x16x16xf16>
}

// -----

// rocMLIR cannot invert a broadcast of the conv result, so the forward chain
// stops before it and the kernel holds only the conv.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x1x1xf16>
// CHECK-NOT: hip.expand
// CHECK-NOT: hip.add

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.expand
// CHECK: hip.add
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x1x1xf16>,
                      %w: tensor<16x8x1x1xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x1x1xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x1x1xf16>,
                                       tensor<16x8x1x1xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x1x1xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>
  %shape = arith.constant dense<[1, 16, 4, 4]> : tensor<4xi64>
  %be = tensor.empty() : tensor<1x16x4x4xf16>
  %bc = hip.expand(%ctx) ins(%c, %shape : tensor<1x16x1x1xf16>, tensor<4xi64>)
      outs(%be : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%bc, %r : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %a : tensor<1x16x4x4xf16>
}

// -----

// A slice of the conv result would make the kernel compute channels it then
// drops, so the slice and its reader stay outside.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x4x4xf16>
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: hip.add

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: tensor.extract_slice
// CHECK: hip.add
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %v = tensor.extract_slice %c[0, 0, 0, 0] [1, 8, 4, 4] [1, 1, 1, 1]
      : tensor<1x16x4x4xf16> to tensor<1x8x4x4xf16>
  %ae = tensor.empty() : tensor<1x8x4x4xf16>
  %a = hip.add(%ctx) ins(%v, %r : tensor<1x8x4x4xf16>, tensor<1x8x4x4xf16>)
      outs(%ae : tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16>
  return %a : tensor<1x8x4x4xf16>
}

// -----

// QDQ around a float conv: the dequantize of the activation is the kernel's
// prologue and the quantize after the activation function its epilogue, so
// the kernel reads and writes ui8. The rank-0 scale and zero point are
// inline constants cloned into the kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[XQ:[a-z0-9]+]]: tensor<1x8x4x4xui8>, %{{.*}}: tensor<16x8x3x3xf32>, %{{.*}}: tensor<16xf32>) -> tensor<1x16x4x4xui8>
// CHECK: hip.dequantize_linear({{.*}}) ins(%[[XQ]]
// CHECK: hip.conv
// CHECK: hip.leaky_relu
// CHECK: %[[Q:.*]] = hip.quantize_linear
// CHECK: return %[[Q]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.dequantize_linear
// CHECK-NOT: hip.quantize_linear
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %xq: tensor<1x8x4x4xui8>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>)
    -> tensor<1x16x4x4xui8> {
  %s = hip.constant {value = dense<1.250000e-02> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %de = tensor.empty() : tensor<1x8x4x4xf32>
  %x = hip.dequantize_linear(%ctx) ins(%xq, %s : tensor<1x8x4x4xui8>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%de : tensor<1x8x4x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x4x4xf32>
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf32>,
                                       tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  %le = tensor.empty() : tensor<1x16x4x4xf32>
  %l = hip.leaky_relu(%ctx) ins(%c : tensor<1x16x4x4xf32>)
      outs(%le : tensor<1x16x4x4xf32>) {alpha = 2.000000e-01 : f64}
      : tensor<1x16x4x4xf32>
  %qe = tensor.empty() : tensor<1x16x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%l, %s : tensor<1x16x4x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%qe : tensor<1x16x4x4xui8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<1x16x4x4xui8>
  return %q : tensor<1x16x4x4xui8>
}

// -----

// A dequantize after the epilogue's quantize would widen the kernel output
// back to f32, so the chain stops at the quantize; the dequantize waits for
// the kernel that reads it. A quantize that wraps (saturate = 0) has no TOSA
// lowering and is not taken either.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x4x4xui8>
// CHECK: hip.quantize_linear
// CHECK-NOT: hip.dequantize_linear
// CHECK: return
// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x4x4xf32>
// CHECK-NOT: hip.quantize_linear

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.dequantize_linear
// CHECK: hip.rocmlir
// CHECK: hip.quantize_linear{{.*}}saturate = 0
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf32>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>)
    -> (tensor<1x16x4x4xf32>, tensor<1x16x4x4xui8>) {
  %s = hip.constant {value = dense<1.250000e-02> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf32>,
                                       tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  %qe = tensor.empty() : tensor<1x16x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%c, %s : tensor<1x16x4x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%qe : tensor<1x16x4x4xui8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<1x16x4x4xui8>
  %de = tensor.empty() : tensor<1x16x4x4xf32>
  %d = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x16x4x4xui8>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%de : tensor<1x16x4x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x16x4x4xf32>
  %e2 = tensor.empty() : tensor<1x16x4x4xf32>
  %c2 = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf32>,
                                        tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e2 : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  %qe2 = tensor.empty() : tensor<1x16x4x4xui8>
  %q2 = hip.quantize_linear(%ctx) ins(%c2, %s : tensor<1x16x4x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%qe2 : tensor<1x16x4x4xui8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 0 : i64} : tensor<1x16x4x4xui8>
  return %d, %q2 : tensor<1x16x4x4xf32>, tensor<1x16x4x4xui8>
}

// -----

// A quantize with a second reader keeps running, so the conv's kernel takes
// only its own dequantize and reads the ui8 tensor the quantize writes.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[XQ:[a-z0-9]+]]: tensor<1x8x4x4xui8>,
// CHECK-NOT: hip.quantize_linear
// CHECK: hip.dequantize_linear({{.*}}) ins(%[[XQ]]
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[Q:.*]] = hip.quantize_linear
// CHECK: hip.rocmlir({{.*}}) @rocMlir{{[0-9]+}} ins(%[[Q]],
// CHECK: return {{.*}}, %[[Q]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf32>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>)
    -> (tensor<1x16x4x4xf32>, tensor<1x8x4x4xui8>) {
  %s = hip.constant {value = dense<1.250000e-02> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x8x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x8x4x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%qe : tensor<1x8x4x4xui8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<1x8x4x4xui8>
  %de = tensor.empty() : tensor<1x8x4x4xf32>
  %d = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x8x4x4xui8>, tensor<f32>)
      zero_point(%zp : tensor<ui8>) outs(%de : tensor<1x8x4x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x4x4xf32>
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%d, %w, %b : tensor<1x8x4x4xf32>,
                                       tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  return %c, %q : tensor<1x16x4x4xf32>, tensor<1x8x4x4xui8>
}

// -----

// A conv result with a reader outside the kernel still takes its epilogue. The
// kernel returns the conv result as well, and the outside reader, which sat
// between the conv and the epilogue, moves below the dispatch.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> (tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[S:.*]] = hip.sigmoid({{.*}}) ins(%[[C]]
// CHECK: return %[[C]], %[[S]]

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG: %[[EC:.*]] = tensor.empty() : tensor<1x16x4x4xf16>
// CHECK-DAG: %[[ES:.*]] = tensor.empty() : tensor<1x16x4x4xf16>
// CHECK: %[[R:.*]]:2 = hip.rocmlir({{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}) outs(%[[EC]], %[[ES]] : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
// CHECK: %[[N:.*]] = hip.isnan({{.*}}) ins(%[[R]]#0
// CHECK: return %[[R]]#1, %[[N]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> (tensor<1x16x4x4xf16>, tensor<1x16x4x4xi1>) {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ne = tensor.empty() : tensor<1x16x4x4xi1>
  %n = hip.isnan(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%ne : tensor<1x16x4x4xi1>) : tensor<1x16x4x4xi1>
  %se = tensor.empty() : tensor<1x16x4x4xf16>
  %s = hip.sigmoid(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%se : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  return %s, %n : tensor<1x16x4x4xf16>, tensor<1x16x4x4xi1>
}

// -----

// A value in the middle of the epilogue escapes: the residual sum is a graph
// output and also feeds the activation. The kernel returns both, in chain
// order.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{.*}}: tensor<1x8x4x4xf16>, %{{.*}}: tensor<16x8x3x3xf16>, %{{.*}}: tensor<16xf16>, %[[RES:.*]]: tensor<1x16x4x4xf16>) -> (tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[C]], %[[RES]]
// CHECK: %[[T:.*]] = hip.tanh({{.*}}) ins(%[[A]]
// CHECK: return %[[A]], %[[T]]

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[R:.*]]:2 = hip.rocmlir
// CHECK: return %[[R]]#1, %[[R]]#0
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %res: tensor<1x16x4x4xf16>)
    -> (tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>) {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%c, %res : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %te = tensor.empty() : tensor<1x16x4x4xf16>
  %t = hip.tanh(%ctx) ins(%a : tensor<1x16x4x4xf16>)
      outs(%te : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  return %t, %a : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>
}

// -----

// An epilogue operand computed outside from the conv result would make the
// kernel read its own output, so the add stays outside and the kernel holds
// the conv alone.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x4x4xf16>
// CHECK-NOT: hip.add
// CHECK: return

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[R:.*]] = hip.rocmlir
// CHECK: %[[N:.*]] = hip.isnan({{.*}}) ins(%[[R]]
// CHECK: %[[W:.*]] = hip.where({{.*}}) ins(%[[N]], %[[R]], %[[R]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ne = tensor.empty() : tensor<1x16x4x4xi1>
  %n = hip.isnan(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%ne : tensor<1x16x4x4xi1>) : tensor<1x16x4x4xi1>
  %we = tensor.empty() : tensor<1x16x4x4xf16>
  %r = hip.where(%ctx) ins(%n, %c, %c : tensor<1x16x4x4xi1>,
                       tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%we : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}

// -----

// SiLU written out as x * sigmoid(x): the conv result has two readers, both
// in the kernel, so nothing escapes and the kernel has one result.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x16x4x4xf16>
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[S:.*]] = hip.sigmoid({{.*}}) ins(%[[C]]
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[C]], %[[S]]
// CHECK: return %[[M]] :

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[R:.*]] = hip.rocmlir({{.*}}) @rocMlir{{[0-9]+}} ins({{.*}}) outs(%{{.*}} : tensor<1x16x4x4xf16>)
// CHECK-NEXT: return %[[R]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %se = tensor.empty() : tensor<1x16x4x4xf16>
  %s = hip.sigmoid(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%se : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%c, %s : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%me : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}
