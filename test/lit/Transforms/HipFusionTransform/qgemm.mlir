// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fold-qdq-to-quant-types --hip-fusion-transform --split-input-file %s | FileCheck %s

// transB=1 reads B as [N, K], so Y is [64, 32]; the rank-1 C broadcasts along
// M. The Q/DQ fold into the gemm's element types; alpha, beta and transB stay
// attributes.
// CHECK-LABEL: func.func @qgemm
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x128xi8>, %[[B:.*]]: tensor<32x128xi8>, %[[C:.*]]: tensor<32xi8>) -> tensor<64x32xi8> {
// CHECK-NEXT:    %[[QA:.*]] = hip.storage_cast %[[A]] : tensor<64x128xi8> to tensor<64x128x!quant.uniform<i8:f32, 2.500000e-01:-5>>
// CHECK-NEXT:    %[[QB:.*]] = hip.storage_cast %[[B]] : tensor<32x128xi8> to tensor<32x128x!quant.uniform<i8:f32, 1.250000e-01:3>>
// CHECK-NEXT:    %[[QC:.*]] = hip.storage_cast %[[C]] : tensor<32xi8> to tensor<32x!quant.uniform<i8:f32, 3.125000e-02:-2>>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>>
// CHECK-NEXT:    %[[GEMM:.*]] = hip.gemm(%[[CTX]]) ins(%[[QA]], %[[QB]], %[[QC]] : tensor<64x128x!quant.uniform<i8:f32, 2.500000e-01:-5>>, tensor<32x128x!quant.uniform<i8:f32, 1.250000e-01:3>>, tensor<32x!quant.uniform<i8:f32, 3.125000e-02:-2>>) outs(%[[INIT]] : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>>) {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64} : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>>
// CHECK-NEXT:    %[[OUT:.*]] = hip.storage_cast %[[GEMM]] : tensor<64x32x!quant.uniform<i8:f32, 5.000000e-01:7>> to tensor<64x32xi8>
// CHECK-NEXT:    return %[[OUT]] : tensor<64x32xi8>
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

// The bias-free gemm is a plain matmul, so it becomes hip.matmul and only A, B
// and the result carry quant types. A ui8 activation also exercises the
// unsigned zero-point read.
// CHECK-LABEL: func.func @qgemm_no_bias
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<16x64xui8>, %[[B:.*]]: tensor<64x8xi8>) -> tensor<16x8xui8> {
// CHECK-NEXT:    %[[QA:.*]] = hip.storage_cast %[[A]] : tensor<16x64xui8> to tensor<16x64x!quant.uniform<u8:f32, 5.000000e-01:128>>
// CHECK-NEXT:    %[[QB:.*]] = hip.storage_cast %[[B]] : tensor<64x8xi8> to tensor<64x8x!quant.uniform<i8:f32, 2.500000e-01:-3>>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<16x8x!quant.uniform<u8:f32, 1.250000e-01:200>>
// CHECK-NEXT:    %[[MM:.*]] = hip.matmul(%[[CTX]]) ins(%[[QA]], %[[QB]] : {{.*}}) outs(%[[INIT]] : tensor<16x8x!quant.uniform<u8:f32, 1.250000e-01:200>>) : tensor<16x8x!quant.uniform<u8:f32, 1.250000e-01:200>>
// CHECK-NEXT:    %[[OUT:.*]] = hip.storage_cast %[[MM]] : tensor<16x8x!quant.uniform<u8:f32, 1.250000e-01:200>> to tensor<16x8xui8>
// CHECK-NEXT:    return %[[OUT]] : tensor<16x8xui8>
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

// -----

// An int32 bias is the form an ONNX quantizer emits (C_scale = A_scale *
// B_scale). C may be 32-bit even though A and B may not.

// CHECK-LABEL: func.func @qgemm_int32_bias
// CHECK:         hip.storage_cast %{{.*}} : tensor<16xi32> to tensor<16x!quant.uniform<i32:f32, 3.125000e-02>>
// CHECK:         hip.matmul(%{{.*}}) ins({{.*}}) outs({{.*}}) {transB = 1 : i64}
// CHECK-NOT:     hip.quantize_linear
func.func @qgemm_int32_bias(%ctx: !hip.context, %a: tensor<8x32xi8>,
                            %b: tensor<16x32xi8>, %c: tensor<16xi32>)
    -> tensor<8x16xi8> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sc = hip.constant {value = dense<0.03125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %zp32 = hip.constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %e0 = tensor.empty() : tensor<8x32xf32>
  %dq_a = hip.dequantize_linear(%ctx) ins(%a, %sa : tensor<8x32xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e0 : tensor<8x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x32xf32>
  %e1 = tensor.empty() : tensor<16x32xf32>
  %dq_b = hip.dequantize_linear(%ctx) ins(%b, %sb : tensor<16x32xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e1 : tensor<16x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<16x32xf32>
  %e2 = tensor.empty() : tensor<16xf32>
  %dq_c = hip.dequantize_linear(%ctx) ins(%c, %sc : tensor<16xi32>, tensor<f32>)
      zero_point(%zp32 : tensor<i32>) outs(%e2 : tensor<16xf32>)
      {axis = 0 : i64, block_size = 0 : i64} : tensor<16xf32>
  %e3 = tensor.empty() : tensor<8x16xf32>
  %g = hip.gemm(%ctx) ins(%dq_a, %dq_b, %dq_c : tensor<8x32xf32>, tensor<16x32xf32>, tensor<16xf32>)
      outs(%e3 : tensor<8x16xf32>) {transB = 1 : i64} : tensor<8x16xf32>
  %e4 = tensor.empty() : tensor<8x16xi8>
  %q = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<8x16xf32>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e4 : tensor<8x16xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8x16xi8>
  return %q : tensor<8x16xi8>
}

// -----

// A packed 4-bit B is two values per byte, so it cannot be a quant type: a
// per-tensor one still fuses to hip.qgemm with B_bits = 4.

// CHECK-LABEL: func.func @qgemm_per_tensor_w4
// CHECK-NOT:     hip.storage_cast
// CHECK:         hip.qgemm
// CHECK-SAME:    B_bits = 4 : i64
func.func @qgemm_per_tensor_w4(%ctx: !hip.context, %a: tensor<8x32xi8>,
                               %b: tensor<16x32xi8>) -> tensor<8x16xi8> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %e0 = tensor.empty() : tensor<8x32xf32>
  %dq_a = hip.dequantize_linear(%ctx) ins(%a, %sa : tensor<8x32xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e0 : tensor<8x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x32xf32>
  %e1 = tensor.empty() : tensor<16x32xf32>
  %dq_b = hip.dequantize_linear(%ctx) ins(%b, %sb : tensor<16x32xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e1 : tensor<16x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_int4} : tensor<16x32xf32>
  %e3 = tensor.empty() : tensor<8x16xf32>
  %g = hip.gemm(%ctx) ins(%dq_a, %dq_b : tensor<8x32xf32>, tensor<16x32xf32>)
      outs(%e3 : tensor<8x16xf32>) {transB = 1 : i64} : tensor<8x16xf32>
  %e4 = tensor.empty() : tensor<8x16xi8>
  %q = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<8x16xf32>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%e4 : tensor<8x16xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8x16xi8>
  return %q : tensor<8x16xi8>
}

// -----

// An int32 bias, as an ONNX quantizer emits it, keeps its 32-bit storage type.
// alpha = beta = 1 with a row bias is a matmul with a bias.

// CHECK-LABEL: func.func @gemm_int32_bias
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<i8:f32, 5.000000e-01:-1>>, tensor<8x6x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<6x!quant.uniform<i32:f32, 1.250000e-01>>) outs(%{{.*}} : tensor<4x6x!quant.uniform<i8:f32, 1.000000e+00:2>>) : tensor<4x6x!quant.uniform<i8:f32, 1.000000e+00:2>>
func.func @gemm_int32_bias(%ctx: !hip.context, %a: tensor<4x8xi8>,
                           %b: tensor<8x6xi8>, %c: tensor<6xi32>) -> tensor<4x6xi8> {
  %sa = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<-1> : tensor<i8>} : tensor<i8>
  %sb = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %zb = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %sc = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %zc = hip.constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %sy = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %zy = hip.constant {value = dense<2> : tensor<i8>} : tensor<i8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xi8>, tensor<f32>)
      zero_point(%za : tensor<i8>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %db_e = tensor.empty() : tensor<8x6xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8x6xi8>, tensor<f32>)
      zero_point(%zb : tensor<i8>)
      outs(%db_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %dc_e = tensor.empty() : tensor<6xf32>
  %dc = hip.dequantize_linear(%ctx)
      ins(%c, %sc : tensor<6xi32>, tensor<f32>)
      zero_point(%zc : tensor<i32>)
      outs(%dc_e : tensor<6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %g = hip.gemm(%ctx) ins(%da, %db, %dc : tensor<4x8xf32>, tensor<8x6xf32>, tensor<6xf32>)
      outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %o = tensor.empty() : tensor<4x6xi8>
  %q = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%zy : tensor<i8>) outs(%o : tensor<4x6xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xi8>
  return %q : tensor<4x6xi8>
}

// -----

// A packed 4-bit weight is not one value per element, so even per-tensor it
// stays on hip.qgemm instead of becoming a quant type.

// CHECK-LABEL: func.func @gemm_packed_int4_per_tensor
// CHECK-NOT:     hip.storage_cast
// CHECK:         hip.qgemm
// CHECK-SAME:    B_bits = 4 : i64
func.func @gemm_packed_int4_per_tensor(%ctx: !hip.context, %a: tensor<4x8xi8>,
                                       %b: tensor<8x6xi8>) -> tensor<4x6xi8> {
  %sa = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<-1> : tensor<i8>} : tensor<i8>
  %sb = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %zb = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %sy = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %zy = hip.constant {value = dense<2> : tensor<i8>} : tensor<i8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xi8>, tensor<f32>)
      zero_point(%za : tensor<i8>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %db_e = tensor.empty() : tensor<8x6xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8x6xi8>, tensor<f32>)
      zero_point(%zb : tensor<i8>)
      outs(%db_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_int4} : tensor<8x6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %g = hip.gemm(%ctx) ins(%da, %db : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %o = tensor.empty() : tensor<4x6xi8>
  %q = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%zy : tensor<i8>) outs(%o : tensor<4x6xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xi8>
  return %q : tensor<4x6xi8>
}
