// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: 4-bit-packed (INT4/UINT4) constant support + DequantizeLinear marking.
//
// ONNX INT4/UINT4 has no MLIR element type, so 4-bit initializers are imported
// as an i8/ui8 tensor of the LOGICAL element count while the backing buffer is
// only ceil(numel/2) packed bytes (two nibbles/byte). Two things must happen:
//   1. The hip.constant carrier must ACCEPT the half-size external source
//      (size == ceil(numel/2)) for an 8-bit result -- the verifier otherwise
//      rejects "size does not match result byte size" and int4 models never
//      compile.
//   2. lowerOnnxConstants stamps `packed_int4` on the consuming DequantizeLinear
//      so downstream lowering nibble-unpacks instead of over-reading the packed
//      buffer as full-width i8.
//
// A real (size == numel) int8 weight is NOT packed and must NOT be marked.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph() -> (tensor<8xf32>, tensor<8xf32>, tensor<7xf32>) {
    // Packed INT4 weight: i8 tensor of 8 LOGICAL elements backed by 4 bytes
    // (external size = 4 = ceil(8/2)) -> hip.constant accepts it; DQ is marked.
    %w4 = "onnx.Constant"() {location = "weights.bin", offset = 0 : i64,
                             size = 4 : i64} : () -> tensor<8xi8>
    %s4 = "onnx.Constant"() {value = dense<0.02> : tensor<f32>} : () -> tensor<f32>
    %z4 = "onnx.Constant"() {location = "weights.bin", offset = 4 : i64,
                             size = 1 : i64} : () -> tensor<1xi8>
    %dq4 = "onnx.Custom"(%w4, %s4, %z4) {
      function_name = "DequantizeLinear", domain_name = "com.microsoft",
      onnx_node_name = "DQ_int4"
    } : (tensor<8xi8>, tensor<f32>, tensor<1xi8>) -> tensor<8xf32>

    // Real INT8 weight: i8 tensor of 8 elements backed by 8 bytes
    // (size == numel) -> not 4-bit; its DequantizeLinear must NOT be marked.
    %w8 = "onnx.Constant"() {location = "weights.bin", offset = 16 : i64,
                             size = 8 : i64} : () -> tensor<8xi8>
    %s8 = "onnx.Constant"() {value = dense<0.02> : tensor<f32>} : () -> tensor<f32>
    %z8 = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %dq8 = "onnx.Custom"(%w8, %s8, %z8) {
      function_name = "DequantizeLinear", domain_name = "com.microsoft",
      onnx_node_name = "DQ_int8"
    } : (tensor<8xi8>, tensor<f32>, tensor<i8>) -> tensor<8xf32>

    // Odd-count packed INT4 weight: 7 LOGICAL elements backed by 4 bytes
    // (size = 4 = ceil(7/2); the last byte's high nibble is padding) -> still
    // packed, so it must be accepted and marked.
    %w4o = "onnx.Constant"() {location = "weights.bin", offset = 24 : i64,
                              size = 4 : i64} : () -> tensor<7xi8>
    %z4o = "onnx.Constant"() {location = "weights.bin", offset = 28 : i64,
                              size = 1 : i64} : () -> tensor<1xi8>
    %dq4o = "onnx.Custom"(%w4o, %s4, %z4o) {
      function_name = "DequantizeLinear", domain_name = "com.microsoft",
      onnx_node_name = "DQ_int4_odd"
    } : (tensor<7xi8>, tensor<f32>, tensor<1xi8>) -> tensor<7xf32>

    return %dq4, %dq8, %dq4o : tensor<8xf32>, tensor<8xf32>, tensor<7xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// The packed-int4 weight lowers to a hip.constant with the half-size external
// source (size = 4 for 8 logical i8 elements) -- previously rejected.
// CHECK: hip.constant
// CHECK-SAME: size = 4 : i64

// Its DequantizeLinear carries packed_int4; the real int8 one does not.
// (DequantizeLinear is plugin-handled, so it survives convert-onnx-to-hip and
// the marker rides on the surviving onnx.Custom op.)
// CHECK: onnx.Custom
// CHECK-SAME: onnx_node_name = "DQ_int4"
// CHECK-SAME: packed_int4

// CHECK: onnx.Custom
// CHECK-NOT: packed_int4
// CHECK-SAME: onnx_node_name = "DQ_int8"

// The odd-count (7-element, size 4 = ceil(7/2)) packed weight is also accepted
// and marked -- odd logical counts pack their last nibble into a padded byte.
// CHECK: onnx.Custom
// CHECK-SAME: onnx_node_name = "DQ_int4_odd"
// CHECK-SAME: packed_int4
