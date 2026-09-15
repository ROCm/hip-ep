// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ Gemm Fusion Patterns (Pure PDLL approach)
//
// Pattern fuses:
//   onnx.DequantizeLinear x2 or x3 -> onnx.Gemm -> onnx.QuantizeLinear
// into:
//   hip.qgemm
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {
  // transB=1 reads B as [N, K], so Y is [64, 32]; the rank-1 C broadcasts
  // along M.

// transA, and in the bias-free case alpha too, keep their ONNX defaults and so
// are elided on print.
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x128xi8>, %[[B:.*]]: tensor<32x128xi8>, %[[C:.*]]: tensor<32xi8>) -> tensor<64x32xi8> {
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<64x32xi8>
// CHECK-NEXT:    %[[QGEMM:.*]] = hip.qgemm(%[[CTX]]) ins(%[[A]], %[[B]], %[[C]] : tensor<64x128xi8>, tensor<32x128xi8>, tensor<32xi8>) outs(%[[EMPTY]] : tensor<64x32xi8>) {A_scale = 2.500000e-01 : f32, A_zero_point = -5 : i64, B_scale = 1.250000e-01 : f32, B_zero_point = 3 : i64, C_scale = 3.125000e-02 : f32, C_zero_point = -2 : i64, Y_scale = 5.000000e-01 : f32, Y_zero_point = 7 : i64, alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64} : tensor<64x32xi8>
// CHECK-NEXT:    return %[[QGEMM]] : tensor<64x32xi8>
  func.func @main_graph(%A: tensor<64x128xi8>,
                        %B: tensor<32x128xi8>,
                        %C: tensor<32xi8>) -> tensor<64x32xi8> {
    %a_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %b_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
    %c_scale = "onnx.Constant"() {value = dense<0.03125> : tensor<f32>} : () -> tensor<f32>
    %c_zp = "onnx.Constant"() {value = dense<-2> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<64x128xi8>, tensor<f32>, tensor<i8>) -> tensor<64x128xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp)
            : (tensor<32x128xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128xf32>
    %c_dq = "onnx.DequantizeLinear"(%C, %c_scale, %c_zp)
            : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>

    %y = "onnx.Gemm"(%a_dq, %b_dq, %c_dq)
         {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : si64}
         : (tensor<64x128xf32>, tensor<32x128xf32>, tensor<32xf32>) -> tensor<64x32xf32>

    %result = "onnx.QuantizeLinear"(%y, %y_scale, %y_zp)
              : (tensor<64x32xf32>, tensor<f32>, tensor<i8>) -> tensor<64x32xi8>

    return %result : tensor<64x32xi8>
  }

  // Trailing optional inputs may simply be omitted, which is the bias-free
  // form. A ui8 activation also exercises the unsigned zero-point read.

// The bias-free form leaves C_scale / C_zero_point / beta unset, so they are
// absent here rather than carrying a meaningless value.
// CHECK-LABEL: func.func @gemm_no_bias
// CHECK-SAME:  (%[[CTX2:.*]]: !hip.context, %[[A2:.*]]: tensor<16x64xui8>, %[[B2:.*]]: tensor<64x8xi8>) -> tensor<16x8xui8> {
// CHECK-NEXT:    %[[EMPTY2:.*]] = tensor.empty() : tensor<16x8xui8>
// CHECK-NEXT:    %[[QGEMM2:.*]] = hip.qgemm(%[[CTX2]]) ins(%[[A2]], %[[B2]] : tensor<16x64xui8>, tensor<64x8xi8>) outs(%[[EMPTY2]] : tensor<16x8xui8>) {A_scale = 5.000000e-01 : f32, A_zero_point = 128 : i64, B_scale = 2.500000e-01 : f32, B_zero_point = -3 : i64, Y_scale = 1.250000e-01 : f32, Y_zero_point = 200 : i64} : tensor<16x8xui8>
// CHECK-NEXT:    return %[[QGEMM2]] : tensor<16x8xui8>
  func.func @gemm_no_bias(%A: tensor<16x64xui8>,
                          %B: tensor<64x8xi8>) -> tensor<16x8xui8> {
    %a_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<128> : tensor<ui8>} : () -> tensor<ui8>
    %b_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<-3> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<200> : tensor<ui8>} : () -> tensor<ui8>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<16x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<16x64xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp)
            : (tensor<64x8xi8>, tensor<f32>, tensor<i8>) -> tensor<64x8xf32>

    %y = "onnx.Gemm"(%a_dq, %b_dq)
         : (tensor<16x64xf32>, tensor<64x8xf32>) -> tensor<16x8xf32>

    %result = "onnx.QuantizeLinear"(%y, %y_scale, %y_zp)
              : (tensor<16x8xf32>, tensor<f32>, tensor<ui8>) -> tensor<16x8xui8>

    return %result : tensor<16x8xui8>
  }

  // A16W4 per output feature. transB=1 reads B as [N, K], which puts the
  // channel axis at 0 and is what the DequantizeLinear's axis must agree with.
  //
  // The external constant form is load-bearing: 4-bit weights import as i8 at
  // the LOGICAL element count, so packing is observable only as a backing byte
  // count of ceil(numel/2) — 256 bytes for 512 weights, 8 for 16 zero points.
  // That is what turns into B_bits = 4.
  //
  // The int32 bias with a symmetric zero point is what an ONNX quantizer emits
  // for a Gemm at c_scale = a_scale * b_scale.

// B_scale / B_zero_point are absent: the per-channel operands replace them and
// their identity defaults are what collapse M_ab to alpha * A_scale / Y_scale.
// C_zero_point is 0 and therefore elided too.
// CHECK-LABEL: func.func @gemm_per_channel_w4
// CHECK-SAME:  (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: tensor<8x32xui16>, %[[C3:.*]]: tensor<16xi32>) -> tensor<8x16xui16> {
// CHECK-NEXT:    %[[B3:.*]] = hip.constant {{.*}}tensor<16x32xi8>
// CHECK-NEXT:    %[[BZP3:.*]] = hip.constant {{.*}}tensor<16xi8>
// CHECK-NEXT:    %[[BS3:.*]] = hip.constant {{.*}}tensor<16xf32>
// CHECK-NEXT:    %[[EMPTY3:.*]] = tensor.empty() : tensor<8x16xui16>
// CHECK-NEXT:    %[[QGEMM3:.*]] = hip.qgemm(%[[CTX3]]) ins(%[[A3]], %[[B3]], %[[C3]] : tensor<8x32xui16>, tensor<16x32xi8>, tensor<16xi32>) b_per_channel(%[[BS3]], %[[BZP3]] : tensor<16xf32>, tensor<16xi8>) outs(%[[EMPTY3]] : tensor<8x16xui16>) {A_scale = 1.250000e-01 : f32, A_zero_point = 35275 : i64, B_bits = 4 : i64, C_scale = 3.125000e-02 : f32, Y_scale = 2.500000e-01 : f32, Y_zero_point = 36322 : i64, transB = 1 : i64} : tensor<8x16xui16>
// CHECK-NEXT:    return %[[QGEMM3]] : tensor<8x16xui16>
  func.func @gemm_per_channel_w4(%A: tensor<8x32xui16>,
                                 %C: tensor<16xi32>) -> tensor<8x16xui16> {
    %B = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 256 : i64}
         : () -> tensor<16x32xi8>
    %b_zp = "onnx.Constant"() {location = "w.bin", offset = 256 : i64, size = 8 : i64}
            : () -> tensor<16xi8>
    %b_scale = "onnx.Constant"() {location = "w.bin", offset = 264 : i64, size = 64 : i64}
               : () -> tensor<16xf32>
    %a_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<35275> : tensor<ui16>} : () -> tensor<ui16>
    %c_scale = "onnx.Constant"() {value = dense<0.03125> : tensor<f32>} : () -> tensor<f32>
    %c_zp = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %y_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<36322> : tensor<ui16>} : () -> tensor<ui16>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<8x32xui16>, tensor<f32>, tensor<ui16>) -> tensor<8x32xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64}
            : (tensor<16x32xi8>, tensor<16xf32>, tensor<16xi8>) -> tensor<16x32xf32>
    %c_dq = "onnx.DequantizeLinear"(%C, %c_scale, %c_zp)
            : (tensor<16xi32>, tensor<f32>, tensor<i32>) -> tensor<16xf32>

    %y = "onnx.Gemm"(%a_dq, %b_dq, %c_dq) {transB = 1 : si64}
         : (tensor<8x32xf32>, tensor<16x32xf32>, tensor<16xf32>) -> tensor<8x16xf32>

    %result = "onnx.QuantizeLinear"(%y, %y_scale, %y_zp)
              : (tensor<8x16xf32>, tensor<f32>, tensor<ui16>) -> tensor<8x16xui16>

    return %result : tensor<8x16xui16>
  }

  // Per-channel weights need not be 4-bit: 18 weights in 18 bytes is plain
  // int8, so B_bits stays at its 8 default and is elided. transB is absent
  // here, which puts N — and therefore the DequantizeLinear's axis — on 1.

// CHECK-LABEL: func.func @gemm_per_channel_w8_no_bias
// CHECK-SAME:  (%[[CTX4:.*]]: !hip.context, %[[A4:.*]]: tensor<4x6xi8>) -> tensor<4x3xi8> {
// CHECK-NEXT:    %[[B4:.*]] = hip.constant {{.*}}tensor<6x3xi8>
// CHECK-NEXT:    %[[BZP4:.*]] = hip.constant {{.*}}tensor<3xi8>
// CHECK-NEXT:    %[[BS4:.*]] = hip.constant {{.*}}tensor<3xf32>
// CHECK-NEXT:    %[[EMPTY4:.*]] = tensor.empty() : tensor<4x3xi8>
// CHECK-NEXT:    %[[QGEMM4:.*]] = hip.qgemm(%[[CTX4]]) ins(%[[A4]], %[[B4]] : tensor<4x6xi8>, tensor<6x3xi8>) b_per_channel(%[[BS4]], %[[BZP4]] : tensor<3xf32>, tensor<3xi8>) outs(%[[EMPTY4]] : tensor<4x3xi8>) {A_scale = 5.000000e-01 : f32, A_zero_point = -5 : i64, Y_scale = 1.250000e-01 : f32, Y_zero_point = 7 : i64} : tensor<4x3xi8>
// CHECK-NEXT:    return %[[QGEMM4]] : tensor<4x3xi8>
  func.func @gemm_per_channel_w8_no_bias(%A: tensor<4x6xi8>) -> tensor<4x3xi8> {
    %B = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 18 : i64}
         : () -> tensor<6x3xi8>
    %b_zp = "onnx.Constant"() {location = "w.bin", offset = 18 : i64, size = 3 : i64}
            : () -> tensor<3xi8>
    %b_scale = "onnx.Constant"() {location = "w.bin", offset = 24 : i64, size = 12 : i64}
               : () -> tensor<3xf32>
    %a_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<4x6xi8>, tensor<f32>, tensor<i8>) -> tensor<4x6xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp) {axis = 1 : si64, block_size = 0 : si64}
            : (tensor<6x3xi8>, tensor<3xf32>, tensor<3xi8>) -> tensor<6x3xf32>

    %y = "onnx.Gemm"(%a_dq, %b_dq)
         : (tensor<4x6xf32>, tensor<6x3xf32>) -> tensor<4x3xf32>

    %result = "onnx.QuantizeLinear"(%y, %y_scale, %y_zp)
              : (tensor<4x3xf32>, tensor<f32>, tensor<i8>) -> tensor<4x3xi8>

    return %result : tensor<4x3xi8>
  }
}

// Nothing of the fused chains survives: no onnx QDQ/Gemm ops anywhere in the
// output.
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Gemm
// CHECK-NOT: onnx.QuantizeLinear
