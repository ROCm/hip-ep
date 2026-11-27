// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A per-output-column add on a hip.matmul_nbits product folds into the op's
// bias operand, in either operand order. Adds that broadcast the product, a
// bias of the wrong shape or type, a product with another user, or a matmul
// that already has a bias stay as they are.

// CHECK-LABEL: func.func @bias_rhs
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<1x4x64xf16>, %[[BIAS:.*]]: tensor<128xf16>
// CHECK-NOT: hip.add
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x4x128xf16>
// CHECK: %[[Y:.*]] = hip.matmul_nbits(%[[CTX]]) ins(%[[A]], %{{.*}}, %{{.*}} : tensor<1x4x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) bias(%[[BIAS]] : tensor<128xf16>) outs(%[[INIT]] : tensor<1x4x128xf16>)
// CHECK-SAME: K = 64 : i64, N = 128 : i64
// CHECK: return %[[Y]]
func.func @bias_rhs(%ctx: !hip.context, %a: tensor<1x4x64xf16>, %bias: tensor<128xf16>) -> tensor<1x4x128xf16> {
  %b = hip.constant {value = dense<1> : tensor<128x2x16xui8>} : tensor<128x2x16xui8>
  %s = hip.constant {value = dense<1.0> : tensor<128x2xf16>} : tensor<128x2xf16>
  %e0 = tensor.empty() : tensor<1x4x128xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %b, %s : tensor<1x4x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e0 : tensor<1x4x128xf16>) {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x4x128xf16>
  %e1 = tensor.empty() : tensor<1x4x128xf16>
  %z = hip.add(%ctx) ins(%y, %bias : tensor<1x4x128xf16>, tensor<128xf16>) outs(%e1 : tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
  return %z : tensor<1x4x128xf16>
}

// -----

// The bias on the left, as a constant, with a dynamic token count. The
// product's tensor.dim query sits before the add, so it is retargeted to the
// matmul's init, which dominates it, rather than to the fused result.
// CHECK-LABEL: func.func @bias_lhs_dynamic
// CHECK-NOT: hip.add
// CHECK: %[[E0:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x128xf16>
// CHECK: tensor.dim %[[E0]]
// CHECK: hip.matmul_nbits({{.*}}) bias(%{{.*}} : tensor<128xf16>) outs(%{{.*}} : tensor<1x?x128xf16>)
func.func @bias_lhs_dynamic(%ctx: !hip.context, %a: tensor<1x?x64xf16>) -> (tensor<1x?x128xf16>, index) {
  %c1 = arith.constant 1 : index
  %n = tensor.dim %a, %c1 : tensor<1x?x64xf16>
  %b = hip.constant {value = dense<1> : tensor<128x2x16xui8>} : tensor<128x2x16xui8>
  %s = hip.constant {value = dense<1.0> : tensor<128x2xf16>} : tensor<128x2xf16>
  %bias = hip.constant {value = dense<0.5> : tensor<128xf16>} : tensor<128xf16>
  %e0 = tensor.empty(%n) : tensor<1x?x128xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %b, %s : tensor<1x?x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e0 : tensor<1x?x128xf16>) {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x?x128xf16>
  %d = tensor.dim %y, %c1 : tensor<1x?x128xf16>
  %e1 = tensor.empty(%n) : tensor<1x?x128xf16>
  %z = hip.add(%ctx) ins(%bias, %y : tensor<128xf16>, tensor<1x?x128xf16>) outs(%e1 : tensor<1x?x128xf16>) -> tensor<1x?x128xf16>
  return %z, %d : tensor<1x?x128xf16>, index
}

// -----

// The product has a second user, so it is still needed without the bias.
// CHECK-LABEL: func.func @product_has_other_user
// CHECK: hip.matmul_nbits
// CHECK-NOT: bias(
// CHECK: hip.add
func.func @product_has_other_user(%ctx: !hip.context, %a: tensor<1x4x64xf16>, %bias: tensor<128xf16>) -> (tensor<1x4x128xf16>, tensor<1x4x128xf16>) {
  %b = hip.constant {value = dense<1> : tensor<128x2x16xui8>} : tensor<128x2x16xui8>
  %s = hip.constant {value = dense<1.0> : tensor<128x2xf16>} : tensor<128x2xf16>
  %e0 = tensor.empty() : tensor<1x4x128xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %b, %s : tensor<1x4x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e0 : tensor<1x4x128xf16>) {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x4x128xf16>
  %e1 = tensor.empty() : tensor<1x4x128xf16>
  %z = hip.add(%ctx) ins(%y, %bias : tensor<1x4x128xf16>, tensor<128xf16>) outs(%e1 : tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
  return %z, %y : tensor<1x4x128xf16>, tensor<1x4x128xf16>
}

// -----

// A full-shape addend is not a per-column bias.
// CHECK-LABEL: func.func @addend_not_vector
// CHECK: hip.matmul_nbits
// CHECK-NOT: bias(
// CHECK: hip.add
func.func @addend_not_vector(%ctx: !hip.context, %a: tensor<1x4x64xf16>, %r: tensor<1x4x128xf16>) -> tensor<1x4x128xf16> {
  %b = hip.constant {value = dense<1> : tensor<128x2x16xui8>} : tensor<128x2x16xui8>
  %s = hip.constant {value = dense<1.0> : tensor<128x2xf16>} : tensor<128x2xf16>
  %e0 = tensor.empty() : tensor<1x4x128xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %b, %s : tensor<1x4x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e0 : tensor<1x4x128xf16>) {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x4x128xf16>
  %e1 = tensor.empty() : tensor<1x4x128xf16>
  %z = hip.add(%ctx) ins(%y, %r : tensor<1x4x128xf16>, tensor<1x4x128xf16>) outs(%e1 : tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
  return %z : tensor<1x4x128xf16>
}

// -----

// A bias in another element type would be read at the wrong width.
// CHECK-LABEL: func.func @bias_wrong_type
// CHECK: hip.matmul_nbits
// CHECK-NOT: bias(
// CHECK: hip.add
func.func @bias_wrong_type(%ctx: !hip.context, %a: tensor<1x4x64xf16>, %bias: tensor<128xf32>) -> tensor<1x4x128xf16> {
  %b = hip.constant {value = dense<1> : tensor<128x2x16xui8>} : tensor<128x2x16xui8>
  %s = hip.constant {value = dense<1.0> : tensor<128x2xf16>} : tensor<128x2xf16>
  %e0 = tensor.empty() : tensor<1x4x128xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %b, %s : tensor<1x4x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e0 : tensor<1x4x128xf16>) {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x4x128xf16>
  %e1 = tensor.empty() : tensor<1x4x128xf16>
  %z = hip.add(%ctx) ins(%y, %bias : tensor<1x4x128xf16>, tensor<128xf32>) outs(%e1 : tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
  return %z : tensor<1x4x128xf16>
}

// -----

// A matmul that already carries a bias keeps the add.
// CHECK-LABEL: func.func @already_biased
// CHECK: hip.matmul_nbits
// CHECK: hip.add
func.func @already_biased(%ctx: !hip.context, %a: tensor<1x4x64xf16>, %b0: tensor<128xf16>, %b1: tensor<128xf16>) -> tensor<1x4x128xf16> {
  %b = hip.constant {value = dense<1> : tensor<128x2x16xui8>} : tensor<128x2x16xui8>
  %s = hip.constant {value = dense<1.0> : tensor<128x2xf16>} : tensor<128x2xf16>
  %e0 = tensor.empty() : tensor<1x4x128xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %b, %s : tensor<1x4x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>) bias(%b0 : tensor<128xf16>) outs(%e0 : tensor<1x4x128xf16>) {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x4x128xf16>
  %e1 = tensor.empty() : tensor<1x4x128xf16>
  %z = hip.add(%ctx) ins(%y, %b1 : tensor<1x4x128xf16>, tensor<128xf16>) outs(%e1 : tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
  return %z : tensor<1x4x128xf16>
}
