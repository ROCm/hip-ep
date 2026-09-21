// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// transB=1 reads B as [N, K], so Y is [64, 32]; the rank-1 C broadcasts along
// M. transA keeps its default and so is elided on print, as is B_bits at 8.
// CHECK-LABEL: func.func @qgemm
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x128xi8>, %[[B:.*]]: tensor<32x128xi8>, %[[C:.*]]: tensor<32xi8>) -> tensor<64x32xi8> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<64x32xi8>
// CHECK-NEXT:    %[[QGEMM:.*]] = hip.qgemm(%[[CTX]]) ins(%[[A]], %[[B]], %[[C]] : tensor<64x128xi8>, tensor<32x128xi8>, tensor<32xi8>) outs(%[[INIT]] : tensor<64x32xi8>) {A_scale = 2.500000e-01 : f32, A_zero_point = -5 : i64, B_scale = 1.250000e-01 : f32, B_zero_point = 3 : i64, C_scale = 3.125000e-02 : f32, C_zero_point = -2 : i64, Y_scale = 5.000000e-01 : f32, Y_zero_point = 7 : i64, alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64} : tensor<64x32xi8>
// CHECK-NEXT:    return %[[QGEMM]] : tensor<64x32xi8>
// CHECK-NEXT:  }
func.func @qgemm(%ctx: !hip.context,
                 %a: tensor<64x128xi8>,
                 %b: tensor<32x128xi8>,
                 %c: tensor<32xi8>) -> tensor<64x32xi8> {
  %a_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<-5> : tensor<i8>} : tensor<i8>
  %b_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %b_zp = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %c_scale = hip.constant {value = dense<0.03125> : tensor<f32>} : tensor<f32>
  %c_zp = hip.constant {value = dense<-2> : tensor<i8>} : tensor<i8>
  %y_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<7> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<64x128xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<64x128xi8>, tensor<f32>)
      zero_point(%a_zp : tensor<i8>)
      outs(%e0 : tensor<64x128xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<64x128xf32>

  %e1 = tensor.empty() : tensor<32x128xf32>
  %dq_b = hip.dequantize_linear(%ctx)
      ins(%b, %b_scale : tensor<32x128xi8>, tensor<f32>)
      zero_point(%b_zp : tensor<i8>)
      outs(%e1 : tensor<32x128xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<32x128xf32>

  %e2 = tensor.empty() : tensor<32xf32>
  %dq_c = hip.dequantize_linear(%ctx)
      ins(%c, %c_scale : tensor<32xi8>, tensor<f32>)
      zero_point(%c_zp : tensor<i8>)
      outs(%e2 : tensor<32xf32>)
      {axis = 0 : i64, block_size = 0 : i64} : tensor<32xf32>

  %e3 = tensor.empty() : tensor<64x32xf32>
  %gemm = hip.gemm(%ctx)
      ins(%dq_a, %dq_b, %dq_c :
          tensor<64x128xf32>, tensor<32x128xf32>, tensor<32xf32>)
      outs(%e3 : tensor<64x32xf32>)
      {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32,
       transA = 0 : i64, transB = 1 : i64} : tensor<64x32xf32>

  %e4 = tensor.empty() : tensor<64x32xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%gemm, %y_scale : tensor<64x32xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<i8>)
      outs(%e4 : tensor<64x32xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<64x32xi8>

  return %q : tensor<64x32xi8>
}

// -----

// The bias-free form leaves C_scale / C_zero_point / beta unset, so they are
// absent here rather than carrying a value the fused op would ignore. A ui8
// activation also exercises the unsigned zero-point read.
// CHECK-LABEL: func.func @qgemm_no_bias
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<16x64xui8>, %[[B:.*]]: tensor<64x8xi8>) -> tensor<16x8xui8> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<16x8xui8>
// CHECK-NEXT:    %[[QGEMM:.*]] = hip.qgemm(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<16x64xui8>, tensor<64x8xi8>) outs(%[[INIT]] : tensor<16x8xui8>) {A_scale = 5.000000e-01 : f32, A_zero_point = 128 : i64, B_scale = 2.500000e-01 : f32, B_zero_point = -3 : i64, Y_scale = 1.250000e-01 : f32, Y_zero_point = 200 : i64} : tensor<16x8xui8>
// CHECK-NEXT:    return %[[QGEMM]] : tensor<16x8xui8>
// CHECK-NEXT:  }
func.func @qgemm_no_bias(%ctx: !hip.context,
                         %a: tensor<16x64xui8>,
                         %b: tensor<64x8xi8>) -> tensor<16x8xui8> {
  %a_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %b_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %b_zp = hip.constant {value = dense<-3> : tensor<i8>} : tensor<i8>
  %y_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<200> : tensor<ui8>} : tensor<ui8>

  %e0 = tensor.empty() : tensor<16x64xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<16x64xui8>, tensor<f32>)
      zero_point(%a_zp : tensor<ui8>)
      outs(%e0 : tensor<16x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<16x64xf32>

  %e1 = tensor.empty() : tensor<64x8xf32>
  %dq_b = hip.dequantize_linear(%ctx)
      ins(%b, %b_scale : tensor<64x8xi8>, tensor<f32>)
      zero_point(%b_zp : tensor<i8>)
      outs(%e1 : tensor<64x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<64x8xf32>

  %e2 = tensor.empty() : tensor<16x8xf32>
  %gemm = hip.gemm(%ctx)
      ins(%dq_a, %dq_b : tensor<16x64xf32>, tensor<64x8xf32>)
      outs(%e2 : tensor<16x8xf32>)
      {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32,
       transA = 0 : i64, transB = 0 : i64} : tensor<16x8xf32>

  %e3 = tensor.empty() : tensor<16x8xui8>
  %q = hip.quantize_linear(%ctx)
      ins(%gemm, %y_scale : tensor<16x8xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui8>)
      outs(%e3 : tensor<16x8xui8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<16x8xui8>

  return %q : tensor<16x8xui8>
}

// -----

// A16W4 quantized per output feature, with an int32 bias at a symmetric zero
// point -- what an ONNX quantizer emits for a Gemm at
// c_scale = a_scale * b_scale. transB=1 reads B as [N, K], which puts the
// feature axis at 0 and is what the dequantize's axis must agree with.
//
// The external constant form is load-bearing: with no inline payload the
// weight scale is not a readable splat, which is what keeps this off the
// per-tensor path. B_scale / B_zero_point stay unset because the per-feature
// operands replace them, and C_zero_point is 0 and therefore elided too.
// CHECK-LABEL: func.func @qgemm_per_channel_w4
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<8x32xui16>, %[[C:.*]]: tensor<16xi32>) -> tensor<8x16xui16> {
// CHECK-NEXT:    %[[B:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, size = 256 : i64} : tensor<16x32xi8>
// CHECK-NEXT:    %[[BZP:.*]] = hip.constant {location = "w.bin", offset = 256 : i64, size = 8 : i64} : tensor<16xi8>
// CHECK-NEXT:    %[[BSCALE:.*]] = hip.constant {location = "w.bin", offset = 264 : i64, size = 64 : i64} : tensor<16xf32>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<8x16xui16>
// CHECK-NEXT:    %[[QGEMM:.*]] = hip.qgemm(%[[CTX]]) ins(%[[A]], %[[B]], %[[C]] : tensor<8x32xui16>, tensor<16x32xi8>, tensor<16xi32>) b_per_channel(%[[BSCALE]], %[[BZP]] : tensor<16xf32>, tensor<16xi8>) outs(%[[INIT]] : tensor<8x16xui16>) {A_scale = 1.250000e-01 : f32, A_zero_point = 35275 : i64, B_bits = 4 : i64, C_scale = 3.125000e-02 : f32, Y_scale = 2.500000e-01 : f32, Y_zero_point = 36322 : i64, transB = 1 : i64} : tensor<8x16xui16>
// CHECK-NEXT:    return %[[QGEMM]] : tensor<8x16xui16>
// CHECK-NEXT:  }
func.func @qgemm_per_channel_w4(%ctx: !hip.context,
                                %a: tensor<8x32xui16>,
                                %c: tensor<16xi32>) -> tensor<8x16xui16> {
  %b = hip.constant {location = "w.bin", offset = 0 : i64, size = 256 : i64} : tensor<16x32xi8>
  %b_zp = hip.constant {location = "w.bin", offset = 256 : i64, size = 8 : i64} : tensor<16xi8>
  %b_scale = hip.constant {location = "w.bin", offset = 264 : i64, size = 64 : i64} : tensor<16xf32>
  %a_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %c_scale = hip.constant {value = dense<0.03125> : tensor<f32>} : tensor<f32>
  %c_zp = hip.constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %y_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<8x32xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<8x32xui16>, tensor<f32>)
      zero_point(%a_zp : tensor<ui16>)
      outs(%e0 : tensor<8x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x32xf32>

  %e1 = tensor.empty() : tensor<16x32xf32>
  %dq_b = hip.dequantize_linear(%ctx)
      ins(%b, %b_scale : tensor<16x32xi8>, tensor<16xf32>)
      zero_point(%b_zp : tensor<16xi8>)
      outs(%e1 : tensor<16x32xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4} : tensor<16x32xf32>

  %e2 = tensor.empty() : tensor<16xf32>
  %dq_c = hip.dequantize_linear(%ctx)
      ins(%c, %c_scale : tensor<16xi32>, tensor<f32>)
      zero_point(%c_zp : tensor<i32>)
      outs(%e2 : tensor<16xf32>)
      {axis = 0 : i64, block_size = 0 : i64} : tensor<16xf32>

  %e3 = tensor.empty() : tensor<8x16xf32>
  %gemm = hip.gemm(%ctx)
      ins(%dq_a, %dq_b, %dq_c :
          tensor<8x32xf32>, tensor<16x32xf32>, tensor<16xf32>)
      outs(%e3 : tensor<8x16xf32>)
      {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32,
       transA = 0 : i64, transB = 1 : i64} : tensor<8x16xf32>

  %e4 = tensor.empty() : tensor<8x16xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%gemm, %y_scale : tensor<8x16xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e4 : tensor<8x16xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<8x16xui16>

  return %q : tensor<8x16xui16>
}

// -----

// Per-feature weights need not be 4-bit: without the packed_int4 marker
// B_bits stays at its 8 default and is elided. transB is 0 here, which puts
// the feature axis -- and therefore the dequantize's axis -- on 1.
// CHECK-LABEL: func.func @qgemm_per_channel_w8_no_bias
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<4x6xi8>) -> tensor<4x3xi8> {
// CHECK-NEXT:    %[[B:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, size = 18 : i64} : tensor<6x3xi8>
// CHECK-NEXT:    %[[BZP:.*]] = hip.constant {location = "w.bin", offset = 18 : i64, size = 3 : i64} : tensor<3xi8>
// CHECK-NEXT:    %[[BSCALE:.*]] = hip.constant {location = "w.bin", offset = 24 : i64, size = 12 : i64} : tensor<3xf32>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<4x3xi8>
// CHECK-NEXT:    %[[QGEMM:.*]] = hip.qgemm(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4x6xi8>, tensor<6x3xi8>) b_per_channel(%[[BSCALE]], %[[BZP]] : tensor<3xf32>, tensor<3xi8>) outs(%[[INIT]] : tensor<4x3xi8>) {A_scale = 5.000000e-01 : f32, A_zero_point = -5 : i64, Y_scale = 1.250000e-01 : f32, Y_zero_point = 7 : i64} : tensor<4x3xi8>
// CHECK-NEXT:    return %[[QGEMM]] : tensor<4x3xi8>
// CHECK-NEXT:  }
func.func @qgemm_per_channel_w8_no_bias(%ctx: !hip.context,
                                        %a: tensor<4x6xi8>) -> tensor<4x3xi8> {
  %b = hip.constant {location = "w.bin", offset = 0 : i64, size = 18 : i64} : tensor<6x3xi8>
  %b_zp = hip.constant {location = "w.bin", offset = 18 : i64, size = 3 : i64} : tensor<3xi8>
  %b_scale = hip.constant {location = "w.bin", offset = 24 : i64, size = 12 : i64} : tensor<3xf32>
  %a_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %a_zp = hip.constant {value = dense<-5> : tensor<i8>} : tensor<i8>
  %y_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<7> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<4x6xf32>
  %dq_a = hip.dequantize_linear(%ctx)
      ins(%a, %a_scale : tensor<4x6xi8>, tensor<f32>)
      zero_point(%a_zp : tensor<i8>)
      outs(%e0 : tensor<4x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x6xf32>

  %e1 = tensor.empty() : tensor<6x3xf32>
  %dq_b = hip.dequantize_linear(%ctx)
      ins(%b, %b_scale : tensor<6x3xi8>, tensor<3xf32>)
      zero_point(%b_zp : tensor<3xi8>)
      outs(%e1 : tensor<6x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<6x3xf32>

  %e2 = tensor.empty() : tensor<4x3xf32>
  %gemm = hip.gemm(%ctx)
      ins(%dq_a, %dq_b : tensor<4x6xf32>, tensor<6x3xf32>)
      outs(%e2 : tensor<4x3xf32>)
      {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32,
       transA = 0 : i64, transB = 0 : i64} : tensor<4x3xf32>

  %e3 = tensor.empty() : tensor<4x3xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%gemm, %y_scale : tensor<4x3xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<i8>)
      outs(%e3 : tensor<4x3xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<4x3xi8>

  return %q : tensor<4x3xi8>
}
