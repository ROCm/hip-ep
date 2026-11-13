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

// Static int4 MatMulNBits splits into unpack, a blocked dequant, and a
// transposed matmul, then all three are outlined. The omitted zero point
// becomes a constant 8 inside the kernel. The packed blob stays outside,
// flattened from [N, k_blocks, blob] to [N, K/2].

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<2x16xf16>, %[[PACKED:.*]]: tensor<2x8xui8>, %[[SCALES:.*]]: tensor<2x1xf16>)
// CHECK: %[[CTX:.*]] = ub.poison : !hip.context
// CHECK: hip.unpack_int4(%[[CTX]]) ins(%[[PACKED]] : tensor<2x8xui8>)
// CHECK: hip.constant
// CHECK-SAME: dense<8.000000e+00> : tensor<2x1xf16>
// CHECK: hip.dequantize_linear
// CHECK-SAME: block_size = 16
// CHECK: hip.matmul
// CHECK-SAME: transB = 1
// CHECK-NOT: hip.matmul_nbits

// CHECK-LABEL: func.func @main_graph
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor<2x8xui8>
// CHECK: hip.rocmlir
// CHECK-NOT: hip.matmul_nbits
// CHECK-NOT: hip.unpack_int4
func.func @main_graph(%ctx: !hip.context, %a: tensor<2x16xf16>,
                      %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>)
    -> tensor<2x2xf16> {
  %e = tensor.empty() : tensor<2x2xf16>
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        outs(%e : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// -----

// Bias and a following GELU stay in the same kernel. An explicit zero point
// is a kernel argument rather than the constant 8.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<2x16xf16>, %[[BIAS:.*]]: tensor<2xf16>, %[[PACKED:.*]]: tensor<2x8xui8>, %[[SCALES:.*]]: tensor<2x1xf16>, %[[ZP:.*]]: tensor<2x1xf16>)
// CHECK: hip.unpack_int4
// CHECK: hip.dequantize_linear
// CHECK: hip.matmul
// CHECK-SAME: transB = 1
// CHECK: hip.add
// CHECK: hip.gelu
// CHECK-NOT: hip.constant
// CHECK-NOT: hip.matmul_nbits

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-NOT: hip.matmul_nbits
// CHECK-NOT: hip.unpack_int4
func.func @main_graph(%ctx: !hip.context, %a: tensor<2x16xf16>,
                      %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>,
                      %zp: tensor<2x1xf16>, %bias: tensor<2xf16>)
    -> tensor<2x2xf16> {
  %e = tensor.empty() : tensor<2x2xf16>
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        zero_points(%zp : tensor<2x1xf16>)
        bias(%bias : tensor<2xf16>)
        outs(%e : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<2x2xf16>
  %ge = tensor.empty() : tensor<2x2xf16>
  %g = hip.gelu(%ctx) ins(%r : tensor<2x2xf16>) outs(%ge : tensor<2x2xf16>)
      : tensor<2x2xf16>
  return %g : tensor<2x2xf16>
}

// -----

// A dynamic activation never forms the rock kernel.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.matmul_nbits
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.unpack_int4
func.func @main_graph(%ctx: !hip.context, %a: tensor<?x16xf16>,
                      %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>,
                      %init: tensor<?x2xf16>) -> tensor<?x2xf16> {
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<?x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        outs(%init : tensor<?x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<?x2xf16>
  return %r : tensor<?x2xf16>
}

// -----

// ONNX BOOL is ui8 outside the kernel. A ui8 value used specifically as a
// Where condition becomes signless i8 in the private rocMLIR kernel ABI, so
// its byte storage can feed rocMLIR arithmetic without a conversion op.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: {{.*}}tensor<1x2x4x3xf32>{{.*}}tensor<1x1x3x5xf32>{{.*}}tensor<1x1x4x5xi8>
// CHECK: hip.where
// CHECK-SAME: tensor<1x1x4x5xi8>

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x2x4x3xf32>,
                      %b: tensor<1x1x3x5xf32>,
                      %cond: tensor<1x1x4x5xui8>,
                      %x: tensor<f32>) -> tensor<1x2x4x5xf32> {
  %matmulInit = tensor.empty() : tensor<1x2x4x5xf32>
  %matmul = hip.matmul(%ctx) ins(%a, %b :
      tensor<1x2x4x3xf32>, tensor<1x1x3x5xf32>)
      outs(%matmulInit : tensor<1x2x4x5xf32>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<1x2x4x5xf32>
  %whereInit = tensor.empty() : tensor<1x2x4x5xf32>
  // CHECK: hip.rocmlir
  // CHECK-SAME: %{{.*}} : tensor<1x2x4x3xf32>, tensor<1x1x3x5xf32>, tensor<1x1x4x5xui8>, tensor<1xf32>)
  %result = hip.where(%ctx) ins(%cond, %x, %matmul :
      tensor<1x1x4x5xui8>, tensor<f32>, tensor<1x2x4x5xf32>)
      outs(%whereInit : tensor<1x2x4x5xf32>) : tensor<1x2x4x5xf32>
  return %result : tensor<1x2x4x5xf32>
}

// -----

// A single-use hip.dequant_bnb4 stays in main_graph. Absorbing it makes
// HipToTosa emit a tosa.gather of the codebook, which rocMLIR rejects. The
// outlined matmul reads the dequantized f32 weight. transB swaps the [N, K]
// dequant result into the matmul's [K, N] weight.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.matmul
// CHECK-SAME: tensor<2x8xf32>, tensor<4x8xf32>
// CHECK-NOT: hip.dequant_bnb4

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequant_bnb4
// CHECK: hip.rocmlir
// CHECK-SAME: tensor<2x8xf32>, tensor<4x8xf32>
// CHECK-NOT: tensor<16xui8>
func.func @main_graph(%ctx: !hip.context, %a: tensor<2x8xf32>,
                      %packed: tensor<16xui8>, %absmax: tensor<2xf32>)
    -> tensor<2x4xf32> {
  %weight = tensor.empty() : tensor<4x8xf32>
  %w = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
      tensor<16xui8>, tensor<2xf32>) outs(%weight : tensor<4x8xf32>)
      {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
      : tensor<4x8xf32>
  %e = tensor.empty() : tensor<2x4xf32>
  %c = hip.matmul(%ctx) ins(%a, %w : tensor<2x8xf32>, tensor<4x8xf32>)
      outs(%e : tensor<2x4xf32>)
      {transA = 0 : i64, transB = 1 : i64} : tensor<2x4xf32>
  return %c : tensor<2x4xf32>
}

// -----

// A file-backed zero point is Pure and operand-free, but it has no dense
// value for HIP-to-TOSA. It stays a kernel argument instead of being cloned
// into the rock kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<2x16xf16>, %[[PACKED:.*]]: tensor<2x8xui8>, %[[SCALES:.*]]: tensor<2x1xf16>, %[[ZP:.*]]: tensor<2x1xf16>)
// CHECK-NOT: hip.constant
// CHECK: hip.unpack_int4
// CHECK: hip.dequantize_linear
// CHECK: hip.matmul
// CHECK-SAME: transB = 1
// CHECK-NOT: hip.constant

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.constant
// CHECK-SAME: location = "zp.bin"
// CHECK: hip.rocmlir
// CHECK-NOT: hip.matmul_nbits
// CHECK-NOT: hip.unpack_int4
func.func @main_graph(%ctx: !hip.context, %a: tensor<2x16xf16>,
                      %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>)
    -> tensor<2x2xf16> {
  %zp = hip.constant {location = "zp.bin", offset = 0 : i64, size = 4 : i64}
      : tensor<2x1xf16>
  %e = tensor.empty() : tensor<2x2xf16>
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        zero_points(%zp : tensor<2x1xf16>)
        outs(%e : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// -----

// A memory-backed zero point is the same kind of carrier: no dense value,
// so the outlined kernel receives it as an argument.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<2x16xf16>, %[[PACKED:.*]]: tensor<2x8xui8>, %[[SCALES:.*]]: tensor<2x1xf16>, %[[ZP:.*]]: tensor<2x1xf16>)
// CHECK-NOT: hip.constant
// CHECK: hip.unpack_int4
// CHECK: hip.dequantize_linear
// CHECK-NOT: memory_address

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.constant
// CHECK-SAME: memory_address = 4096
// CHECK: hip.rocmlir
// CHECK-NOT: hip.matmul_nbits
func.func @main_graph(%ctx: !hip.context, %a: tensor<2x16xf16>,
                      %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>)
    -> tensor<2x2xf16> {
  %zp = hip.constant {memory_address = 4096 : i64, size = 4 : i64}
      : tensor<2x1xf16>
  %e = tensor.empty() : tensor<2x2xf16>
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        zero_points(%zp : tensor<2x1xf16>)
        outs(%e : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// -----

// A static integer nearest upsample (asymmetric, floor) in front of a
// convolution is a repeated read. The kernel receives the small tensor and
// rebuilds the 2x view; hip.resize is not launched.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x4x8x8xf16>, %[[W:.*]]: tensor<8x4x3x3xf16>, %[[B:.*]]: tensor<8xf16>)
// CHECK: tensor.expand_shape %[[IN]]
// CHECK: hip.expand
// CHECK: tensor.collapse_shape
// CHECK: hip.conv
// CHECK-NOT: hip.resize

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-SAME: tensor<1x4x8x8xf16>
// CHECK-NOT: hip.resize
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x4x8x8xf16>,
                      %w: tensor<8x4x3x3xf16>, %b: tensor<8xf16>)
    -> tensor<1x8x8x8xf16> {
  %up = tensor.empty() : tensor<1x4x16x16xf16>
  %r = hip.resize(%ctx) ins(%in : tensor<1x4x8x8xf16>)
      outs(%up : tensor<1x4x16x16xf16>)
      {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 2 : i64}
      : tensor<1x4x16x16xf16>
  %e = tensor.empty() : tensor<1x8x8x8xf16>
  %c = hip.conv(%ctx) ins(%r, %w, %b : tensor<1x4x16x16xf16>,
                                      tensor<8x4x3x3xf16>, tensor<8xf16>)
      outs(%e : tensor<1x8x8x8xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x8x8x8xf16>
  return %c : tensor<1x8x8x8xf16>
}

// -----

// Linear resize is not a repeated read, so it stays a kernel and the
// convolution receives the materialized tensor.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<1x4x16x16xf16>
// CHECK-NOT: tensor.expand_shape
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.resize
// CHECK-SAME: mode = 1
// CHECK: hip.rocmlir
// CHECK-SAME: tensor<1x4x16x16xf16>
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x4x8x8xf16>,
                      %w: tensor<8x4x3x3xf16>, %b: tensor<8xf16>)
    -> tensor<1x8x8x8xf16> {
  %up = tensor.empty() : tensor<1x4x16x16xf16>
  %r = hip.resize(%ctx) ins(%in : tensor<1x4x8x8xf16>)
      outs(%up : tensor<1x4x16x16xf16>)
      {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
      : tensor<1x4x16x16xf16>
  %e = tensor.empty() : tensor<1x8x8x8xf16>
  %c = hip.conv(%ctx) ins(%r, %w, %b : tensor<1x4x16x16xf16>,
                                      tensor<8x4x3x3xf16>, tensor<8xf16>)
      outs(%e : tensor<1x8x8x8xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x8x8x8xf16>
  return %c : tensor<1x8x8x8xf16>
}

// -----

// A pointwise epilogue replaces the mul, not the convolution. The original
// convolution would still read the resize until dead intermediates are
// dropped; the absorb path must erase those first so hip.resize is not left
// live, and the outlined kernel still rebuilds the 2x view.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[IN:.*]]: tensor<1x4x8x8xf16>, %[[W:.*]]: tensor<8x4x3x3xf16>, %[[B:.*]]: tensor<8xf16>)
// CHECK: tensor.expand_shape %[[IN]]
// CHECK: hip.expand
// CHECK: tensor.collapse_shape
// CHECK: hip.conv
// CHECK: hip.mul
// CHECK-NOT: hip.resize

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-SAME: tensor<1x4x8x8xf16>
// CHECK-NOT: hip.resize
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x4x8x8xf16>,
                      %w: tensor<8x4x3x3xf16>, %b: tensor<8xf16>)
    -> tensor<1x8x8x8xf16> {
  %scale = hip.constant {value = dense<1.999510e-01> : tensor<f16>} : tensor<f16>
  %up = tensor.empty() : tensor<1x4x16x16xf16>
  %r = hip.resize(%ctx) ins(%in : tensor<1x4x8x8xf16>)
      outs(%up : tensor<1x4x16x16xf16>)
      {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 2 : i64}
      : tensor<1x4x16x16xf16>
  %e = tensor.empty() : tensor<1x8x8x8xf16>
  %c = hip.conv(%ctx) ins(%r, %w, %b : tensor<1x4x16x16xf16>,
                                      tensor<8x4x3x3xf16>, tensor<8xf16>)
      outs(%e : tensor<1x8x8x8xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x8x8x8xf16>
  %mulInit = tensor.empty() : tensor<1x8x8x8xf16>
  %m = hip.mul(%ctx) ins(%c, %scale : tensor<1x8x8x8xf16>, tensor<f16>)
      outs(%mulInit : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8xf16>
  return %m : tensor<1x8x8x8xf16>
}

// -----

// A scales attribute that is not exactly the integer ratio changes
// nearest-floor sampling, so the resize stays a kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: tensor<1x4x16x16xf16>
// CHECK-NOT: tensor.expand_shape
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.resize
// CHECK-SAME: scales = [{{.*}}2.000000{{.*}}, {{.*}}2.000000{{.*}}]
// CHECK: hip.rocmlir
// CHECK-SAME: tensor<1x4x16x16xf16>
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x4x8x8xf16>,
                      %w: tensor<8x4x3x3xf16>, %b: tensor<8xf16>)
    -> tensor<1x8x8x8xf16> {
  %up = tensor.empty() : tensor<1x4x16x16xf16>
  %r = hip.resize(%ctx) ins(%in : tensor<1x4x8x8xf16>)
      outs(%up : tensor<1x4x16x16xf16>)
      {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 2 : i64,
       scales = [1.000000e+00, 1.000000e+00, 2.0000004, 2.0000004]}
      : tensor<1x4x16x16xf16>
  %e = tensor.empty() : tensor<1x8x8x8xf16>
  %c = hip.conv(%ctx) ins(%r, %w, %b : tensor<1x4x16x16xf16>,
                                      tensor<8x4x3x3xf16>, tensor<8xf16>)
      outs(%e : tensor<1x8x8x8xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x8x8x8xf16>
  return %c : tensor<1x8x8x8xf16>
}
