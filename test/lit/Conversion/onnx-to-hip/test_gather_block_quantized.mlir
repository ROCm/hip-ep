// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX com.microsoft.GatherBlockQuantized is correctly lowered to
// hip.gather_block_quantized in tensor-first mode.
//
// Validates:
//   - Custom-op match on (function_name=GatherBlockQuantized,
//     domain_name=com.microsoft)
//   - Optional zero_points operand handling (present + absent)
//   - Attribute propagation (bits, block_size, gather_axis, quantize_axis,
//     unsigned_quant_storage for INT4/UINT4 prepare path)
//   - Output shape derivation (data[:gather_axis] ++ indices ++
//     data[gather_axis+1:])
//   - Dynamic-shape `tensor.dim` + `tensor.empty` plumbing for the gathered
//     dimension
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: 4-bit GatherBlockQuantized with zero_points =====
  // data is [vocab=2048, hidden_packed=96] (packed uint8: 96 bytes per row =
  // 192 4-bit elements). scales/zp are [vocab=2048, blocks=12] with
  // block_size=16. quantize_axis=1 (last). gather_axis=0 (rows).

  func.func @main_graph(%indices: tensor<8xi64>) -> tensor<8x96xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xui8>} : () -> tensor<2048x96xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %zp = "onnx.Constant"() {value = dense<8> : tensor<2048x12xui8>} : () -> tensor<2048x12xui8>
    %out = "onnx.Custom"(%data, %indices, %scales, %zp) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_0"
    } : (tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>, tensor<2048x12xui8>) -> tensor<8x96xf16>
    return %out : tensor<8x96xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IDX:.*]]: tensor<8xi64>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<8x96xf16>
  // CHECK: hip.gather_block_quantized(%[[CTX]]) ins({{.*}}, %[[IDX]], {{.*}} : tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>)
  // CHECK-SAME: zero_points(%{{.*}} : tensor<2048x12xui8>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<8x96xf16>)
  // CHECK-SAME: bits = 4
  // CHECK-SAME: block_size = 16
  // CHECK-SAME: gather_axis = 0
  // CHECK-SAME: quantize_axis = 1
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom

  // ===== Test 2: 8-bit GatherBlockQuantized without zero_points =====

  func.func @test_gbq_uint8_no_zp(%indices: tensor<4xi32>) -> tensor<4x64xf32> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<512x64xui8>} : () -> tensor<512x64xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<512x2xf32>} : () -> tensor<512x2xf32>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 8 : si64,
      block_size = 32 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_1"
    } : (tensor<512x64xui8>, tensor<4xi32>, tensor<512x2xf32>) -> tensor<4x64xf32>
    return %out : tensor<4x64xf32>
  }

  // CHECK-LABEL: func.func @test_gbq_uint8_no_zp
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x64xf32>
  // CHECK: hip.gather_block_quantized({{.*}}) ins({{.*}}, %{{.*}}, %{{.*}} : tensor<512x64xui8>, tensor<4xi32>, tensor<512x2xf32>)
  // CHECK-NOT: zero_points
  // CHECK-SAME: outs(%[[INIT]] : tensor<4x64xf32>)
  // CHECK-SAME: bits = 8
  // CHECK-SAME: block_size = 32
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom

  // ===== Test 3: Dynamic indices length =====
  // The first output dim follows %indices's dynamic dim 0; the trailing 96
  // comes straight from data[1] (static).

  func.func @test_gbq_dynamic_indices(%indices: tensor<?xi64>) -> tensor<?x96xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xui8>} : () -> tensor<2048x96xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_2"
    } : (tensor<2048x96xui8>, tensor<?xi64>, tensor<2048x12xf16>) -> tensor<?x96xf16>
    return %out : tensor<?x96xf16>
  }

  // CHECK-LABEL: func.func @test_gbq_dynamic_indices
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IDX:.*]]: tensor<?xi64>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[IDX]], %[[C0]] : tensor<?xi64>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]]) : tensor<?x96xf16>
  // CHECK: hip.gather_block_quantized(%[[CTX]]) ins({{.*}}, %[[IDX]], {{.*}} : tensor<2048x96xui8>, tensor<?xi64>, tensor<2048x12xf16>) outs(%[[INIT]] : tensor<?x96xf16>)
  // CHECK-NOT: onnx.Custom

  // ===== Test 4: prepare-annotated UINT4 (signless i8 + unsigned flag) =====
  // Packed byte shape with logical quantize_axis already halved; unsigned
  // storage is carried on the Custom op, not via ui8 element type.

  func.func @test_gbq_unsigned_quant_storage(%indices: tensor<8xi64>) -> tensor<8x96xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xi8>} : () -> tensor<2048x96xi8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      unsigned_quant_storage,
      onnx_node_name = "GatherBlockQuantized_3"
    } : (tensor<2048x96xi8>, tensor<8xi64>, tensor<2048x12xf16>) -> tensor<8x96xf16>
    return %out : tensor<8x96xf16>
  }

  // CHECK-LABEL: func.func @test_gbq_unsigned_quant_storage
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<8x96xf16>
  // CHECK: hip.gather_block_quantized({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<2048x96xi8>, tensor<8xi64>, tensor<2048x12xf16>) outs(%[[INIT]] : tensor<8x96xf16>)
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom

  // ===== Test 5: UINT8 bits=4, quantize_axis inferred from shapes =====
  // ONNX often omits quantize_axis; shape invariants identify axis 1.

  func.func @test_gbq_infer_quantize_axis(%indices: tensor<8xi64>) -> tensor<8x128xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<512x128xui8>} : () -> tensor<512x128xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<512x8xf16>} : () -> tensor<512x8xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 32 : si64,
      gather_axis = 0 : si64,
      onnx_node_name = "GatherBlockQuantized_4"
    } : (tensor<512x128xui8>, tensor<8xi64>, tensor<512x8xf16>) -> tensor<8x128xf16>
    return %out : tensor<8x128xf16>
  }

  // CHECK-LABEL: func.func @test_gbq_infer_quantize_axis
  // CHECK: hip.gather_block_quantized({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<512x128xui8>, tensor<8xi64>, tensor<512x8xf16>)
  // CHECK-SAME: quantize_axis = 1
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom

  // ===== Tests 6-8: T1 storage width from the importer's element-type mark ==
  // The importer stamps `onnx.element_type` (the ONNX TensorProto code) on
  // every constant. These three cases are the reason it has to: they are
  // pairwise indistinguishable by MLIR type and `bits` alone, yet ONNX gives
  // them different default zero points (0 for int4/uint4, 2^(bits-1) for
  // uint8). Only `quant_storage_bits` separates them downstream.

  // Test 6: UINT8 (code 2) carrying two nibbles per byte -> storage width 8.

  func.func @test_gbq_storage_uint8_packed(%indices: tensor<8xi64>) -> tensor<8x96xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xui8>, onnx.element_type = 2 : i64} : () -> tensor<2048x96xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_5"
    } : (tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>) -> tensor<8x96xf16>
    return %out : tensor<8x96xf16>
  }

  // CHECK-LABEL: func.func @test_gbq_storage_uint8_packed
  // CHECK: hip.gather_block_quantized
  // CHECK-SAME: bits = 4
  // CHECK-SAME: quant_storage_bits = 8
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom

  // Test 7: UINT4 (code 21) -> storage width 4, even though `data` is the same
  // ui8 byte tensor with the same `bits` as test 6. Getting 8 here is exactly
  // the bug that biases every uint4 embedding by +(8 * scale).

  func.func @test_gbq_storage_uint4(%indices: tensor<8xi64>) -> tensor<8x96xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xui8>, onnx.element_type = 21 : i64} : () -> tensor<2048x96xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_6"
    } : (tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>) -> tensor<8x96xf16>
    return %out : tensor<8x96xf16>
  }

  // CHECK-LABEL: func.func @test_gbq_storage_uint4
  // CHECK: hip.gather_block_quantized
  // CHECK-SAME: bits = 4
  // CHECK-SAME: quant_storage_bits = 4
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom

  // Test 8: INT4 (code 22) -> storage width 4 and signed, so no
  // `unsigned_quant_storage`.

  func.func @test_gbq_storage_int4(%indices: tensor<8xi64>) -> tensor<8x96xf16> {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xi8>, onnx.element_type = 22 : i64} : () -> tensor<2048x96xi8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_7"
    } : (tensor<2048x96xi8>, tensor<8xi64>, tensor<2048x12xf16>) -> tensor<8x96xf16>
    return %out : tensor<8x96xf16>
  }

  // CHECK-LABEL: func.func @test_gbq_storage_int4
  // CHECK: hip.gather_block_quantized
  // CHECK-SAME: bits = 4
  // CHECK-SAME: quant_storage_bits = 4
  // CHECK-NOT: unsigned_quant_storage
  // CHECK-NOT: onnx.Custom
}
