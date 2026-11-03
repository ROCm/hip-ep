// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A layout op only moves elements, so with identical parameters on both ends
// the Q/DQ pair is a no-op and the op can run on the quantized tensor. Every
// case here therefore expects the dequantize and the quantize to be gone, not
// replaced by a fused op.
//
// The scale and zero point are one shared constant per case, because the two
// ends agreeing is the whole precondition. The two negative cases at the end
// are the same chains with one parameter perturbed.
//
// Storage width is deliberately not part of the match, unlike the hip.q*
// fusions: no new kernel is involved, only the layout op's existing lowering.
//
// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// hip.transpose is a DPS op, so the rebuilt copy needs a fresh init carrying
// the quantized element type.
// CHECK-LABEL: func.func @qdq_transpose
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x2x3x4xui16>) -> tensor<1x3x2x4xui16> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x3x2x4xui16>
// CHECK-NEXT:    %[[T:.*]] = hip.transpose(%[[CTX]]) ins(%[[X]] : tensor<1x2x3x4xui16>) outs(%[[INIT]] : tensor<1x3x2x4xui16>) {perm = [0, 2, 1, 3]} : tensor<1x3x2x4xui16>
// CHECK-NEXT:    return %[[T]] : tensor<1x3x2x4xui16>
// CHECK-NEXT:  }
func.func @qdq_transpose(%ctx: !hip.context,
                         %x: tensor<1x2x3x4xui16>) -> tensor<1x3x2x4xui16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x2x3x4xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<1x2x3x4xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<1x2x3x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x3x4xf32>

  %e1 = tensor.empty() : tensor<1x3x2x4xf32>
  %t = hip.transpose(%ctx) ins(%dq : tensor<1x2x3x4xf32>)
      outs(%e1 : tensor<1x3x2x4xf32>) {perm = [0, 2, 1, 3]}
      : tensor<1x3x2x4xf32>

  %e2 = tensor.empty() : tensor<1x3x2x4xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%t, %scale : tensor<1x3x2x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e2 : tensor<1x3x2x4xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x3x2x4xui16>

  return %q : tensor<1x3x2x4xui16>
}

// -----

// onnx.Squeeze, Reshape and Flatten all converge on tensor.collapse_shape, so
// one case covers what took three patterns at the ONNX layer. Being pure
// metadata it carries no init, only a reassociation that survives untouched.
// CHECK-LABEL: func.func @qdq_collapse_shape
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x1x3xui16>) -> tensor<2x3xui16> {
// CHECK-NEXT:    %[[C:.*]] = tensor.collapse_shape %[[X]] {{\[}}[0, 1], [2]] : tensor<2x1x3xui16> into tensor<2x3xui16>
// CHECK-NEXT:    return %[[C]] : tensor<2x3xui16>
// CHECK-NEXT:  }
func.func @qdq_collapse_shape(%ctx: !hip.context,
                              %x: tensor<2x1x3xui16>) -> tensor<2x3xui16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x1x3xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x1x3xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x1x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x1x3xf32>

  %c = tensor.collapse_shape %dq [[0, 1], [2]]
      : tensor<2x1x3xf32> into tensor<2x3xf32>

  %e1 = tensor.empty() : tensor<2x3xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%c, %scale : tensor<2x3xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e1 : tensor<2x3xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<2x3xui16>

  return %q : tensor<2x3xui16>
}

// -----

// The expanding direction, where output_shape also has to carry over.
// CHECK-LABEL: func.func @qdq_expand_shape
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3xui16>) -> tensor<2x1x3xui16> {
// CHECK-NEXT:    %[[E:.*]] = tensor.expand_shape %[[X]] {{\[}}[0, 1], [2]] output_shape [2, 1, 3] : tensor<2x3xui16> into tensor<2x1x3xui16>
// CHECK-NEXT:    return %[[E]] : tensor<2x1x3xui16>
// CHECK-NEXT:  }
func.func @qdq_expand_shape(%ctx: !hip.context,
                            %x: tensor<2x3xui16>) -> tensor<2x1x3xui16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x3xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x3xf32>

  %e = tensor.expand_shape %dq [[0, 1], [2]] output_shape [2, 1, 3]
      : tensor<2x3xf32> into tensor<2x1x3xf32>

  %e1 = tensor.empty() : tensor<2x1x3xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%e, %scale : tensor<2x1x3xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e1 : tensor<2x1x3xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<2x1x3xui16>

  return %q : tensor<2x1x3xui16>
}

// -----

// onnx.Identity is replaced by its operand during conversion, so what used to
// be a three-op chain arrives as an adjacent pair with nothing between it.
// There is no op left to rebuild: the quantize's users read the dequantize's
// input directly.
// CHECK-LABEL: func.func @qdq_pair
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3xui16>) -> tensor<2x3xui16> {
// CHECK-NEXT:    return %[[X]] : tensor<2x3xui16>
// CHECK-NEXT:  }
func.func @qdq_pair(%ctx: !hip.context,
                    %x: tensor<2x3xui16>) -> tensor<2x3xui16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x3xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x3xf32>

  %e1 = tensor.empty() : tensor<2x3xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%dq, %scale : tensor<2x3xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e1 : tensor<2x3xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<2x3xui16>

  return %q : tensor<2x3xui16>
}

// -----

// Signed 8-bit storage takes the same path: the elimination leans on the
// layout op's own lowering, which is width-agnostic, so nothing about the
// match is pinned to one width or signedness.
// CHECK-LABEL: func.func @qdq_transpose_i8
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3xi8>) -> tensor<3x2xi8> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<3x2xi8>
// CHECK-NEXT:    %[[T:.*]] = hip.transpose(%[[CTX]]) ins(%[[X]] : tensor<2x3xi8>) outs(%[[INIT]] : tensor<3x2xi8>) {perm = [1, 0]} : tensor<3x2xi8>
// CHECK-NEXT:    return %[[T]] : tensor<3x2xi8>
// CHECK-NEXT:  }
func.func @qdq_transpose_i8(%ctx: !hip.context,
                            %x: tensor<2x3xi8>) -> tensor<3x2xi8> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<-5> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<2x3xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>)
      outs(%e0 : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x3xf32>

  %e1 = tensor.empty() : tensor<3x2xf32>
  %t = hip.transpose(%ctx) ins(%dq : tensor<2x3xf32>)
      outs(%e1 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>

  %e2 = tensor.empty() : tensor<3x2xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%t, %scale : tensor<3x2xf32>, tensor<f32>)
      zero_point(%zp : tensor<i8>)
      outs(%e2 : tensor<3x2xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<3x2xi8>

  return %q : tensor<3x2xi8>
}

// -----

// Different scales make the pair a real requantization, not a round trip, so
// all three ops have to survive.
// CHECK-LABEL: func.func @different_scale_stays_unfused
// CHECK:         hip.dequantize_linear
// CHECK:         hip.transpose
// CHECK:         hip.quantize_linear
func.func @different_scale_stays_unfused(%ctx: !hip.context,
                                         %x: tensor<2x3xui16>) -> tensor<3x2xui16> {
  %in_s = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %out_s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x3xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<2x3xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x3xf32>

  %e1 = tensor.empty() : tensor<3x2xf32>
  %t = hip.transpose(%ctx) ins(%dq : tensor<2x3xf32>)
      outs(%e1 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>

  %e2 = tensor.empty() : tensor<3x2xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%t, %out_s : tensor<3x2xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e2 : tensor<3x2xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<3x2xui16>

  return %q : tensor<3x2xui16>
}

// -----

// Same for a shifted zero point: the codes come back offset by one, which a
// bare collapse_shape would not reproduce.
// CHECK-LABEL: func.func @different_zp_stays_unfused
// CHECK:         hip.dequantize_linear
// CHECK:         tensor.collapse_shape
// CHECK:         hip.quantize_linear
func.func @different_zp_stays_unfused(%ctx: !hip.context,
                                      %x: tensor<2x1x3xui16>) -> tensor<2x3xui16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %out_z = hip.constant {value = dense<32769> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x1x3xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x1x3xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<2x1x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x1x3xf32>

  %c = tensor.collapse_shape %dq [[0, 1], [2]]
      : tensor<2x1x3xf32> into tensor<2x3xf32>

  %e1 = tensor.empty() : tensor<2x3xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%c, %scale : tensor<2x3xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e1 : tensor<2x3xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<2x3xui16>

  return %q : tensor<2x3xui16>
}

// -----

// An embedding lookup: Gather only picks rows, so the table stays quantized
// and only the gathered rows are produced. The quantize's init is sized from
// tensor.dim of the gather result, a shape-only read that must not count as a
// second consumer.
// CHECK-LABEL: func.func @qdq_gather_dynamic
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[IDX:.*]]: tensor<?x?xi64>) -> tensor<?x?x8xui16> {
// CHECK:         %[[TABLE:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, size = 256 : i64} : tensor<16x8xui16>
// CHECK-NOT:     hip.dequantize_linear
// CHECK:         %[[G:.*]] = hip.gather(%[[CTX]]) ins(%[[TABLE]], %[[IDX]] : tensor<16x8xui16>, tensor<?x?xi64>) outs(%{{.*}} : tensor<?x?x8xui16>) : tensor<?x?x8xui16>
// CHECK-NOT:     hip.quantize_linear
// CHECK:         return %[[G]] : tensor<?x?x8xui16>
func.func @qdq_gather_dynamic(%ctx: !hip.context,
                              %idx: tensor<?x?xi64>) -> tensor<?x?x8xui16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %table = hip.constant {location = "w.bin", offset = 0 : i64, size = 256 : i64} : tensor<16x8xui16>
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<16x8xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%table, %scale : tensor<16x8xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<16x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<16x8xf32>

  %b = tensor.dim %idx, %c0 : tensor<?x?xi64>
  %s = tensor.dim %idx, %c1 : tensor<?x?xi64>
  %e1 = tensor.empty(%b, %s) : tensor<?x?x8xf32>
  %g = hip.gather(%ctx) ins(%dq, %idx : tensor<16x8xf32>, tensor<?x?xi64>)
      outs(%e1 : tensor<?x?x8xf32>) : tensor<?x?x8xf32>

  %gb = tensor.dim %g, %c0 : tensor<?x?x8xf32>
  %gs = tensor.dim %g, %c1 : tensor<?x?x8xf32>
  %e2 = tensor.empty(%gb, %gs) : tensor<?x?x8xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%g, %scale : tensor<?x?x8xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e2 : tensor<?x?x8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<?x?x8xui16>

  return %q : tensor<?x?x8xui16>
}
