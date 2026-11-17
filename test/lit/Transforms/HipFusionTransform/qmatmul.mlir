// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Per-tensor B: the Q/DQ fold into the matmul's element types.
// CHECK-LABEL: func.func @qmatmul
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x128xi8>, %[[B:.*]]: tensor<128x32xi8>) -> tensor<64x32xi8> {
// CHECK-NEXT:    %[[QA:.*]] = hip.storage_cast %[[A]] : tensor<64x128xi8> to tensor<64x128x!quant.uniform<i8:f32, 2.500000e-01:-5>>
// CHECK-NEXT:    %[[QB:.*]] = hip.storage_cast %[[B]] : tensor<128x32xi8> to tensor<128x32x!quant.uniform<i8:f32, 1.250000e-01:3>>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>>
// CHECK-NEXT:    %[[MM:.*]] = hip.matmul(%[[CTX]]) ins(%[[QA]], %[[QB]] : tensor<64x128x!quant.uniform<i8:f32, 2.500000e-01:-5>>, tensor<128x32x!quant.uniform<i8:f32, 1.250000e-01:3>>) outs(%[[INIT]] : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>>) : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>>
// CHECK-NEXT:    %[[OUT:.*]] = hip.storage_cast %[[MM]] : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>> to tensor<64x32xi8>
// CHECK-NEXT:    return %[[OUT]] : tensor<64x32xi8>
// CHECK-NEXT:  }
func.func @qmatmul(%ctx: !hip.context,
                   %a: tensor<64x128xi8>,
                   %b: tensor<128x32xi8>) -> tensor<64x32xi8> {
  %a_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<-5> : tensor<i8>} : tensor<i8>
  %b_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %b_zp = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %y_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<7> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<64x128xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<64x128xi8>, tensor<f32>)
      zero_point(%a_zp : tensor<i8>)
      outs(%e0 : tensor<64x128xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<64x128xf32>

  %e1 = tensor.empty() : tensor<128x32xf32>
  %dq_b = hip.dequantize_linear(%ctx)
      ins(%b, %b_scale : tensor<128x32xi8>, tensor<f32>)
      zero_point(%b_zp : tensor<i8>)
      outs(%e1 : tensor<128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<128x32xf32>

  %e2 = tensor.empty() : tensor<64x32xf32>
  %prod = hip.matmul(%ctx)
    ins(%dq_a, %dq_b : tensor<64x128xf32>, tensor<128x32xf32>)
    outs(%e2 : tensor<64x32xf32>)
    {transA = 0 : i64, transB = 0 : i64} : tensor<64x32xf32>

  %e3 = tensor.empty() : tensor<64x32xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%prod, %y_scale : tensor<64x32xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<i8>)
      outs(%e3 : tensor<64x32xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<64x32xi8>

  return %q : tensor<64x32xi8>
}

// -----

// A16 activation, packed INT4 weights quantized per output column. The weight
// scale and zero point stay operands, so their carriers survive the fusion.
// CHECK-LABEL: func.func @qmatmul_w4_per_column
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<4x8xui16>) -> tensor<4x6xui16> {
// CHECK-NEXT:    %[[W:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, size = 24 : i64} : tensor<8x6xi8>
// CHECK-NEXT:    %[[WZP:.*]] = hip.constant {location = "w.bin", offset = 24 : i64, size = 3 : i64} : tensor<6xi8>
// CHECK-NEXT:    %[[WSCALE:.*]] = hip.constant {location = "w.bin", offset = 32 : i64, size = 24 : i64} : tensor<6xf32>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<4x6xui16>
// CHECK-NEXT:    %[[QMM:.*]] = hip.qmatmul(%[[CTX]]) ins(%[[A]], %[[W]], %[[WSCALE]], %[[WZP]] : tensor<4x8xui16>, tensor<8x6xi8>, tensor<6xf32>, tensor<6xi8>) outs(%[[INIT]] : tensor<4x6xui16>) {A_scale = 1.638800e-04 : f32, A_zero_point = 35275 : i64, B_quant_axis = 1 : i64, Y_scale = 3.687020e-04 : f32, Y_zero_point = 36322 : i64, packed_int4} : tensor<4x6xui16>
// CHECK-NEXT:    return %[[QMM]] : tensor<4x6xui16>
// CHECK-NEXT:  }
func.func @qmatmul_w4_per_column(%ctx: !hip.context,
                                 %a: tensor<4x8xui16>) -> tensor<4x6xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 24 : i64} : tensor<8x6xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 24 : i64, size = 3 : i64} : tensor<6xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 32 : i64, size = 24 : i64} : tensor<6xf32>
  %a_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<4x8xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<4x8xui16>, tensor<f32>)
      zero_point(%a_zp : tensor<ui16>)
      outs(%e0 : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>

  %e1 = tensor.empty() : tensor<8x6xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<8x6xi8>, tensor<6xf32>)
      zero_point(%w_zp : tensor<6xi8>)
      outs(%e1 : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_int4} : tensor<8x6xf32>

  %e2 = tensor.empty() : tensor<4x6xf32>
  %prod = hip.matmul(%ctx)
      ins(%dq_a, %dq_w : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e2 : tensor<4x6xf32>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<4x6xf32>

  %e3 = tensor.empty() : tensor<4x6xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%prod, %y_scale : tensor<4x6xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>

  return %q : tensor<4x6xui16>
}

// -----

// The W4 case with full-width INT8 weights: same shapes, no packed_int4 marker
// on either the matched dequantize or the fused op.
// CHECK-LABEL: func.func @qmatmul_w8_per_column
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<4x8xui16>) -> tensor<4x6xui16> {
// CHECK-NEXT:    %[[W:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, size = 48 : i64} : tensor<8x6xi8>
// CHECK-NEXT:    %[[WZP:.*]] = hip.constant {location = "w.bin", offset = 48 : i64, size = 6 : i64} : tensor<6xi8>
// CHECK-NEXT:    %[[WSCALE:.*]] = hip.constant {location = "w.bin", offset = 56 : i64, size = 24 : i64} : tensor<6xf32>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<4x6xui16>
// CHECK-NEXT:    %[[QMM:.*]] = hip.qmatmul(%[[CTX]]) ins(%[[A]], %[[W]], %[[WSCALE]], %[[WZP]] : tensor<4x8xui16>, tensor<8x6xi8>, tensor<6xf32>, tensor<6xi8>) outs(%[[INIT]] : tensor<4x6xui16>) {A_scale = 1.638800e-04 : f32, A_zero_point = 35275 : i64, B_quant_axis = 1 : i64, Y_scale = 3.687020e-04 : f32, Y_zero_point = 36322 : i64} : tensor<4x6xui16>
// CHECK-NEXT:    return %[[QMM]] : tensor<4x6xui16>
// CHECK-NEXT:  }
func.func @qmatmul_w8_per_column(%ctx: !hip.context,
                                 %a: tensor<4x8xui16>) -> tensor<4x6xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 48 : i64} : tensor<8x6xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 48 : i64, size = 6 : i64} : tensor<6xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 56 : i64, size = 24 : i64} : tensor<6xf32>
  %a_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<4x8xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<4x8xui16>, tensor<f32>)
      zero_point(%a_zp : tensor<ui16>)
      outs(%e0 : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>

  %e1 = tensor.empty() : tensor<8x6xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<8x6xi8>, tensor<6xf32>)
      zero_point(%w_zp : tensor<6xi8>)
      outs(%e1 : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>

  %e2 = tensor.empty() : tensor<4x6xf32>
  %prod = hip.matmul(%ctx)
      ins(%dq_a, %dq_w : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e2 : tensor<4x6xf32>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<4x6xf32>

  %e3 = tensor.empty() : tensor<4x6xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%prod, %y_scale : tensor<4x6xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>

  return %q : tensor<4x6xui16>
}

// -----

// transA / transB stay attributes of the typed matmul.

// CHECK-LABEL: func.func @qmatmul_trans
// CHECK:         hip.matmul(%{{.*}}) ins({{.*}}) outs({{.*}}) {transA = 1 : i64, transB = 1 : i64}
// CHECK-NOT:     hip.quantize_linear
func.func @qmatmul_trans(%ctx: !hip.context, %a: tensor<128x64xi8>,
                         %b: tensor<32x128xi8>) -> tensor<64x32xi8> {
  %s = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %e0 = tensor.empty() : tensor<128x64xf32>
  %dq_a = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<128x64xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e0 : tensor<128x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<128x64xf32>
  %e1 = tensor.empty() : tensor<32x128xf32>
  %dq_b = hip.dequantize_linear(%ctx) ins(%b, %s : tensor<32x128xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e1 : tensor<32x128xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<32x128xf32>
  %e2 = tensor.empty() : tensor<64x32xf32>
  %mm = hip.matmul(%ctx) ins(%dq_a, %dq_b : tensor<128x64xf32>, tensor<32x128xf32>)
      outs(%e2 : tensor<64x32xf32>) {transA = 1 : i64, transB = 1 : i64}
      : tensor<64x32xf32>
  %e3 = tensor.empty() : tensor<64x32xi8>
  %q = hip.quantize_linear(%ctx) ins(%mm, %s : tensor<64x32xf32>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e3 : tensor<64x32xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<64x32xi8>
  return %q : tensor<64x32xi8>
}

// -----

// A float matmul is untouched.

// CHECK-LABEL: func.func @matmul_float
// CHECK-NEXT:    tensor.empty() : tensor<8x8xf32>
// CHECK-NEXT:    hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8x8xf32>, tensor<8x8xf32>) outs(%{{.*}} : tensor<8x8xf32>) : tensor<8x8xf32>
// CHECK-NEXT:    return
func.func @matmul_float(%ctx: !hip.context, %a: tensor<8x8xf32>,
                        %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %e = tensor.empty() : tensor<8x8xf32>
  %mm = hip.matmul(%ctx) ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%e : tensor<8x8xf32>) : tensor<8x8xf32>
  return %mm : tensor<8x8xf32>
}

// -----

// A16W8 with transB: the 16-bit activation and the transpose flag carry over,
// and the storage types differ per operand.

// CHECK-LABEL: func.func @matmul_a16w8_trans_b
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<6x8x!quant.uniform<i8:f32, 1.250000e-01:3>>) outs(%{{.*}} : tensor<4x6x!quant.uniform<u16:f32, 5.000000e-01:32768>>) {transB = 1 : i64}
// CHECK-NOT:     hip.quantize_linear
// CHECK-NOT:     hip.dequantize_linear
func.func @matmul_a16w8_trans_b(%ctx: !hip.context, %a: tensor<4x8xui16>,
                                %b: tensor<6x8xi8>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %zb = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zy = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %db_e = tensor.empty() : tensor<6x8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<6x8xi8>, tensor<f32>)
      zero_point(%zb : tensor<i8>)
      outs(%db_e : tensor<6x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<6x8xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %mm = hip.matmul(%ctx) ins(%da, %db : tensor<4x8xf32>, tensor<6x8xf32>)
      outs(%e : tensor<4x6xf32>) {transB = 1 : i64} : tensor<4x6xf32>
  %o = tensor.empty() : tensor<4x6xui16>
  %q = hip.quantize_linear(%ctx) ins(%mm, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%zy : tensor<ui16>) outs(%o : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  return %q : tensor<4x6xui16>
}

// -----

// The folds do not check kernels: a 16-bit weight (A16W16, as in attention)
// folds even though the quantized matmul kernel takes 8-bit B only.

// CHECK-LABEL: func.func @matmul_w16_no_kernel_check
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<8x6x!quant.uniform<u16:f32, 1.250000e-01:32768>>)
// CHECK-NOT:     hip.quantize_linear
// CHECK-NOT:     hip.dequantize_linear
func.func @matmul_w16_no_kernel_check(%ctx: !hip.context, %a: tensor<4x8xui16>,
                              %b: tensor<8x6xui16>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %zb = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zy = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %db_e = tensor.empty() : tensor<8x6xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8x6xui16>, tensor<f32>)
      zero_point(%zb : tensor<ui16>)
      outs(%db_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %mm = hip.matmul(%ctx) ins(%da, %db : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %o = tensor.empty() : tensor<4x6xui16>
  %q = hip.quantize_linear(%ctx) ins(%mm, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%zy : tensor<ui16>) outs(%o : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  return %q : tensor<4x6xui16>
}

// -----

// A float matmul is the same op with float types; nothing to fold.

// CHECK-LABEL: func.func @matmul_float
// CHECK-NEXT:    tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:    hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x6xf32>) outs(%{{.*}} : tensor<4x6xf32>) : tensor<4x6xf32>
// CHECK-NEXT:    return
func.func @matmul_float(%ctx: !hip.context, %a: tensor<4x8xf32>,
                        %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %e = tensor.empty() : tensor<4x6xf32>
  %mm = hip.matmul(%ctx) ins(%a, %b : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  return %mm : tensor<4x6xf32>
}
