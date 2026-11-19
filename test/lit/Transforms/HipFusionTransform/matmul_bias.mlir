// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.matmul + row-bias hip.add -> hip.matmul with a bias operand. The pattern
// checks structure only (shapes, single use, one float type for a float
// chain), so float and quant-typed chains fuse alike and kernel availability
// is not its concern.

// RUN: hip-mlir-opt --hip-fold-qdq-to-quant-types --hip-fusion-transform --split-input-file %s | FileCheck %s

// The op keeps A's rank-3 shape: no reshape, no gemm.

// CHECK-LABEL: func.func @rank3
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<1x196x512x!quant.uniform<u16:f32, 2.500000e-01:32768>>, %[[W:.*]]: tensor<512x1536x!quant.uniform<i8:f32, 1.250000e-01:3>>, %[[BIAS:.*]]: tensor<1536x!quant.uniform<u16:f32, 6.250000e-02:32768>>)
// CHECK:         %[[Y:.*]] = hip.matmul(%[[CTX]]) ins(%[[A]], %[[W]], %[[BIAS]] : tensor<1x196x512x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<512x1536x!quant.uniform<i8:f32, 1.250000e-01:3>>, tensor<1536x!quant.uniform<u16:f32, 6.250000e-02:32768>>) outs(%{{.*}} : tensor<1x196x1536x!quant.uniform<u16:f32, 1.000000e+00:32768>>) : tensor<1x196x1536x!quant.uniform<u16:f32, 1.000000e+00:32768>>
// CHECK-NEXT:    return %[[Y]]
// CHECK-NOT:     hip.add
// CHECK-NOT:     hip.gemm
// CHECK-NOT:     tensor.collapse_shape
// CHECK-NOT:     tensor.expand_shape
func.func @rank3(%ctx: !hip.context, %a: tensor<1x196x512x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<512x1536x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<1536x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> tensor<1x196x1536x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty() : tensor<1x196x1536x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<1x196x512x!quant.uniform<u16:f32, 0.25:32768>>, tensor<512x1536x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<1x196x1536x!quant.uniform<u16:f32, 0.5:32768>>) : tensor<1x196x1536x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<1x196x1536x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<1x196x1536x!quant.uniform<u16:f32, 0.5:32768>>, tensor<1536x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<1x196x1536x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<1x196x1536x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<1x196x1536x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// Addition commutes, a [1, N] bias collapses to [N], and transB carries over.

// CHECK-LABEL: func.func @rank2_bias_first
// CHECK:         %[[B1:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]] : tensor<1x6x!quant.uniform<u16:f32, 6.250000e-02:32768>> into tensor<6x!quant.uniform<u16:f32, 6.250000e-02:32768>>
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %[[B1]] : {{.*}}) outs(%{{.*}} : tensor<4x6x!quant.uniform<u16:f32, 1.000000e+00:32768>>) {transB = 1 : i64} : tensor<4x6x!quant.uniform<u16:f32, 1.000000e+00:32768>>
// CHECK-NOT:     hip.add
func.func @rank2_bias_first(%ctx: !hip.context, %a: tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<6x8x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<1x6x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, tensor<6x8x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) {transB = 1 : i64} : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%bias, %m : tensor<1x6x!quant.uniform<u16:f32, 0.0625:32768>>, tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>)
      outs(%e1 : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// A dynamic extent is fine: the leading dims fold at lowering, and the add's
// init (which carries the dynamic size) becomes the matmul's.

// CHECK-LABEL: func.func @dynamic_batch
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<?x196x512x{{.*}}>, tensor<512x1536x{{.*}}>, tensor<1536x{{.*}}>) outs(%{{.*}} : tensor<?x196x1536x{{.*}}>)
// CHECK-NOT:     hip.add
func.func @dynamic_batch(%ctx: !hip.context, %a: tensor<?x196x512x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<512x1536x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<1536x!quant.uniform<u16:f32, 0.0625:32768>>, %d: index)
    -> tensor<?x196x1536x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty(%d) : tensor<?x196x1536x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<?x196x512x!quant.uniform<u16:f32, 0.25:32768>>, tensor<512x1536x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<?x196x1536x!quant.uniform<u16:f32, 0.5:32768>>) : tensor<?x196x1536x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty(%d) : tensor<?x196x1536x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<?x196x1536x!quant.uniform<u16:f32, 0.5:32768>>, tensor<1536x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<?x196x1536x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<?x196x1536x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<?x196x1536x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// A second user of the matmul result keeps the matmul alive, so the add stays.

// CHECK-LABEL: func.func @two_users
// CHECK:         hip.matmul
// CHECK:         hip.add
func.func @two_users(%ctx: !hip.context, %a: tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> (tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>, tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) {
  %e0 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>, tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  return %y, %m : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>, tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
}

// -----

// Only a row bias is a bias: a full [M, N] addend is a residual add.

// CHECK-LABEL: func.func @residual_not_bias
// CHECK:         hip.matmul
// CHECK:         hip.add
func.func @residual_not_bias(%ctx: !hip.context, %a: tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<4x6x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>, tensor<4x6x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// Folding A's leading dims into the rows needs them outermost: transA with a
// rank-3 A stays unfused.

// CHECK-LABEL: func.func @rank3_trans_a
// CHECK:         hip.matmul
// CHECK:         hip.add
func.func @rank3_trans_a(%ctx: !hip.context, %a: tensor<2x8x4x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> tensor<2x4x6x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty() : tensor<2x4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<2x8x4x!quant.uniform<u16:f32, 0.25:32768>>, tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<2x4x6x!quant.uniform<u16:f32, 0.5:32768>>) {transA = 1 : i64} : tensor<2x4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<2x4x6x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<2x4x6x!quant.uniform<u16:f32, 0.5:32768>>, tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<2x4x6x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<2x4x6x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<2x4x6x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// A rank-2 A with transA has no leading dims to fold, so it fuses; the lowering
// reads M and K through transA.

// CHECK-LABEL: func.func @rank2_trans_a
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<8x4x{{.*}}>, tensor<8x6x{{.*}}>, tensor<6x{{.*}}>) outs(%{{.*}} : tensor<4x6x{{.*}}>) {transA = 1 : i64}
// CHECK-NOT:     hip.add
func.func @rank2_trans_a(%ctx: !hip.context, %a: tensor<8x4x!quant.uniform<u16:f32, 0.25:32768>>,
                         %w: tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>, %bias: tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<8x4x!quant.uniform<u16:f32, 0.25:32768>>, tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e0 : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) {transA = 1 : i64} : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>, tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// No kernel takes a 16-bit weight today; the fusion does not care.

// CHECK-LABEL: func.func @w16_no_kernel_check
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-NOT:     hip.add
func.func @w16_no_kernel_check(%ctx: !hip.context, %a: tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, %w: tensor<8x6x!quant.uniform<u16:f32, 0.125:32768>>, %bias: tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
    -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>> {
  %e0 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, tensor<8x6x!quant.uniform<u16:f32, 0.125:32768>>)
      outs(%e0 : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %e1 = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>, tensor<6x!quant.uniform<u16:f32, 0.0625:32768>>)
      outs(%e1 : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>) -> tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
  return %y : tensor<4x6x!quant.uniform<u16:f32, 1.0:32768>>
}

// -----

// A float chain fuses the same way.

// CHECK-LABEL: func.func @float_fused
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8xf16>, tensor<8x6xf16>, tensor<6xf16>) outs(%{{.*}} : tensor<4x6xf16>) : tensor<4x6xf16>
// CHECK-NOT:     hip.add
func.func @float_fused(%ctx: !hip.context, %a: tensor<4x8xf16>,
                       %w: tensor<8x6xf16>, %bias: tensor<6xf16>)
    -> tensor<4x6xf16> {
  %e0 = tensor.empty() : tensor<4x6xf16>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8xf16>, tensor<8x6xf16>)
      outs(%e0 : tensor<4x6xf16>) : tensor<4x6xf16>
  %e1 = tensor.empty() : tensor<4x6xf16>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<4x6xf16>, tensor<6xf16>)
      outs(%e1 : tensor<4x6xf16>) -> tensor<4x6xf16>
  return %y : tensor<4x6xf16>
}

// -----

// A float bias of another precision has no single meaning on the fused op.

// CHECK-LABEL: func.func @mixed_float_types
// CHECK:         hip.matmul
// CHECK:         hip.add
func.func @mixed_float_types(%ctx: !hip.context, %a: tensor<4x8xf16>,
                             %w: tensor<8x6xf16>, %bias: tensor<6xf32>)
    -> tensor<4x6xf16> {
  %e0 = tensor.empty() : tensor<4x6xf16>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8xf16>, tensor<8x6xf16>)
      outs(%e0 : tensor<4x6xf16>) : tensor<4x6xf16>
  %e1 = tensor.empty() : tensor<4x6xf16>
  %y = hip.add(%ctx) ins(%m, %bias : tensor<4x6xf16>, tensor<6xf32>)
      outs(%e1 : tensor<4x6xf16>) -> tensor<4x6xf16>
  return %y : tensor<4x6xf16>
}

// -----

// The full QDQ chain, from the Q/DQ ops: matmul -> Q -> DQ -> add <- DQ(bias)
// -> Q. The Q/DQ fold into types, the casts between matmul and add cancel, and
// the bias fusion then fires: one matmul, Q/DQ gone.

// CHECK-LABEL: func.func @qdq_chain
// CHECK-NOT:     hip.quantize_linear
// CHECK-NOT:     hip.dequantize_linear
// CHECK-NOT:     hip.add
// CHECK:         %[[M:.*]] = hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<8x6x!quant.uniform<i8:f32, 1.250000e-01:3>>, tensor<6x!quant.uniform<u16:f32, 6.250000e-02:32768>>) outs(%{{.*}} : tensor<4x6x!quant.uniform<u16:f32, 1.000000e+00:32768>>)
// CHECK:         hip.storage_cast %[[M]] : tensor<4x6x!quant.uniform<u16:f32, 1.000000e+00:32768>> to tensor<4x6xui16>
func.func @qdq_chain(%ctx: !hip.context, %a: tensor<4x8xui16>,
                     %w: tensor<8x6xi8>, %bias: tensor<6xui16>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sw = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %zw = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %sm = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zm = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sb = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %zb = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sy = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %zy = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %dw_e = tensor.empty() : tensor<8x6xf32>
  %dw = hip.dequantize_linear(%ctx)
      ins(%w, %sw : tensor<8x6xi8>, tensor<f32>)
      zero_point(%zw : tensor<i8>)
      outs(%dw_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %e0 = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%da, %dw : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e0 : tensor<4x6xf32>) : tensor<4x6xf32>
  %qm_e = tensor.empty() : tensor<4x6xui16>
  %qm = hip.quantize_linear(%ctx) ins(%m, %sm : tensor<4x6xf32>, tensor<f32>)
      zero_point(%zm : tensor<ui16>) outs(%qm_e : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  %dm_e = tensor.empty() : tensor<4x6xf32>
  %dm = hip.dequantize_linear(%ctx)
      ins(%qm, %sm : tensor<4x6xui16>, tensor<f32>)
      zero_point(%zm : tensor<ui16>)
      outs(%dm_e : tensor<4x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x6xf32>
  %db_e = tensor.empty() : tensor<6xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%bias, %sb : tensor<6xui16>, tensor<f32>)
      zero_point(%zb : tensor<ui16>)
      outs(%db_e : tensor<6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<6xf32>
  %e1 = tensor.empty() : tensor<4x6xf32>
  %s = hip.add(%ctx) ins(%dm, %db : tensor<4x6xf32>, tensor<6xf32>)
      outs(%e1 : tensor<4x6xf32>) -> tensor<4x6xf32>
  %qy_e = tensor.empty() : tensor<4x6xui16>
  %qy = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%zy : tensor<ui16>) outs(%qy_e : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  return %qy : tensor<4x6xui16>
}

// -----

// The common ONNX form: an int32 bias at A_scale * W_scale. The add (u16 + i32
// -> u16) has no elementwise kernel, but its Q/DQ fold anyway, so the bias
// fusion sees a typed add and produces one matmul with the int32 bias.

// CHECK-LABEL: func.func @qdq_chain_int32_bias
// CHECK-NOT:     hip.quantize_linear
// CHECK-NOT:     hip.dequantize_linear
// CHECK-NOT:     hip.add
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<8x6x!quant.uniform<i8:f32, 1.250000e-01>>, tensor<6x!quant.uniform<i32:f32, 3.125000e-02>>) outs(%{{.*}} : tensor<4x6x!quant.uniform<u16:f32, 1.000000e+00:32768>>)
func.func @qdq_chain_int32_bias(%ctx: !hip.context, %a: tensor<4x8xui16>,
                             %w: tensor<8x6xi8>, %bias: tensor<6xi32>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sw = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %zw = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %sm = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zm = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sb = hip.constant {value = dense<0.03125> : tensor<f32>} : tensor<f32>
  %zb = hip.constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %sy = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %zy = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %e0 = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %sa : tensor<4x8xui16>, tensor<f32>) zero_point(%za : tensor<ui16>) outs(%e0 : tensor<4x8xf32>) {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %e1 = tensor.empty() : tensor<8x6xf32>
  %dw = hip.dequantize_linear(%ctx) ins(%w, %sw : tensor<8x6xi8>, tensor<f32>) zero_point(%zw : tensor<i8>) outs(%e1 : tensor<8x6xf32>) {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %e2 = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%da, %dw : tensor<4x8xf32>, tensor<8x6xf32>) outs(%e2 : tensor<4x6xf32>) : tensor<4x6xf32>
  %e3 = tensor.empty() : tensor<4x6xui16>
  %qm = hip.quantize_linear(%ctx) ins(%m, %sm : tensor<4x6xf32>, tensor<f32>) zero_point(%zm : tensor<ui16>) outs(%e3 : tensor<4x6xui16>) {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64, saturate = 1 : i64} : tensor<4x6xui16>
  %e4 = tensor.empty() : tensor<4x6xf32>
  %dm = hip.dequantize_linear(%ctx) ins(%qm, %sm : tensor<4x6xui16>, tensor<f32>) zero_point(%zm : tensor<ui16>) outs(%e4 : tensor<4x6xf32>) {axis = 1 : i64, block_size = 0 : i64} : tensor<4x6xf32>
  %e5 = tensor.empty() : tensor<6xf32>
  %db = hip.dequantize_linear(%ctx) ins(%bias, %sb : tensor<6xi32>, tensor<f32>) zero_point(%zb : tensor<i32>) outs(%e5 : tensor<6xf32>) {axis = 0 : i64, block_size = 0 : i64} : tensor<6xf32>
  %e6 = tensor.empty() : tensor<4x6xf32>
  %s = hip.add(%ctx) ins(%dm, %db : tensor<4x6xf32>, tensor<6xf32>) outs(%e6 : tensor<4x6xf32>) -> tensor<4x6xf32>
  %e7 = tensor.empty() : tensor<4x6xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<4x6xf32>, tensor<f32>) zero_point(%zy : tensor<ui16>) outs(%e7 : tensor<4x6xui16>) {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64, saturate = 1 : i64} : tensor<4x6xui16>
  return %y : tensor<4x6xui16>
}
