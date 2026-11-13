// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --canonicalize --verify-each -o %t
// RUN: FileCheck %s < %t
// RUN: hip-mlir-opt %t --canonicalize --verify-each -o %t.again
// RUN: diff %t %t.again

// Preserve the GPU value for other consumers, but recover its host dimension
// without loading tensor payload or bypassing the i64-to-i32 truncation.
// CHECK-LABEL: func.func @dimension_with_tensor_user
// CHECK: %[[DIM:.*]] = tensor.dim
// CHECK: %[[I64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK: %[[PACKED:.*]] = tensor.from_elements %[[I64]] : tensor<i64>
// CHECK: %[[GPU:.*]] = hip.cast{{.*}} ins(%[[PACKED]] : tensor<i64>)
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: tensor.extract
// CHECK: %[[HOST:.*]] = arith.trunci %[[I64]] : i64 to i32
// CHECK-NOT: hip.readback_scalar
// CHECK: return %[[GPU]], %[[HOST]]
func.func @dimension_with_tensor_user(%ctx: !hip.context, %data: tensor<?x?xi64>)
    -> (tensor<i32>, i32) {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %data, %c1 : tensor<?x?xi64>
  %wide = arith.index_cast %dim : index to i64
  %packed = tensor.from_elements %wide : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %gpu, %host : tensor<i32>, i32
}

// A dead tensor cast needs no special erasure logic in the readback pattern.
// CHECK-LABEL: func.func @dimension_only_host_user
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: tensor.from_elements
// CHECK: %[[WIDE:.*]] = arith.index_cast {{.*}} : index to i64
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.readback_scalar
// CHECK-NOT: tensor.from_elements
// CHECK: %[[NARROW:.*]] = arith.trunci %[[WIDE]] : i64 to i32
// CHECK-NEXT: return %[[NARROW]]
func.func @dimension_only_host_user(%ctx: !hip.context, %data: tensor<?xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %data, %c0 : tensor<?xi64>
  %wide = arith.index_cast %dim : index to i64
  %packed = tensor.from_elements %wide : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// Folding a static dimension may replace from_elements with a dense constant
// before the readback pattern is visited. Keep the low 32 bits, not saturation.
// CHECK-LABEL: func.func @static_dimension
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT: return %[[ZERO]]
func.func @static_dimension(%ctx: !hip.context, %data: tensor<4294967296xi8>) -> i32 {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %data, %c0 : tensor<4294967296xi8>
  %wide = arith.index_cast %dim : index to i64
  %packed = tensor.from_elements %wide : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @constant_signed_boundary
// CHECK: %[[MIN:.*]] = arith.constant -2147483648 : i32
// CHECK-NEXT: return %[[MIN]]
func.func @constant_signed_boundary(%ctx: !hip.context) -> i32 {
  %packed = arith.constant dense<2147483648> : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @constant_negative
// CHECK: %[[NEG:.*]] = arith.constant -1 : i32
// CHECK-NEXT: return %[[NEG]]
func.func @constant_negative(%ctx: !hip.context) -> i32 {
  %packed = arith.constant dense<-1> : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// Unknown payloads, including GPU-produced values, must stay synchronized.
// CHECK-LABEL: func.func @unknown_payload
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @unknown_payload(%ctx: !hip.context, %input: tensor<i64>) -> i32 {
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%input : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @gpu_producer
// CHECK: hip.cast
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @gpu_producer(%ctx: !hip.context, %input: tensor<f32>) -> i32 {
  %wideInit = tensor.empty() : tensor<i64>
  %wide = hip.cast(%ctx) ins(%input : tensor<f32>) outs(%wideInit : tensor<i64>)
      {to = 7 : i64} : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%wide : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// The proof does not chase arbitrary host SSA or tensor-payload extractions.
// CHECK-LABEL: func.func @arbitrary_scalar
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @arbitrary_scalar(%ctx: !hip.context, %input: i64) -> i32 {
  %packed = tensor.from_elements %input : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @arbitrary_index
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @arbitrary_index(%ctx: !hip.context, %input: index) -> i32 {
  %wide = arith.index_cast %input : index to i64
  %packed = tensor.from_elements %wide : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @extracted_payload
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @extracted_payload(%ctx: !hip.context, %input: tensor<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %element = tensor.extract %input[%c0] : tensor<1xi64>
  %packed = tensor.from_elements %element : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%packed : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @memref_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @memref_unchanged(%ctx: !hip.context, %input: memref<i64>, %output: memref<i32>) -> i32 {
  hip.cast(%ctx) ins(%input : memref<i64>) outs(%output : memref<i32>) {to = 6 : i64}
  %host = hip.readback_scalar(%ctx, %output : memref<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @rank_one_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @rank_one_unchanged(%ctx: !hip.context) -> i32 {
  %input = arith.constant dense<8> : tensor<1xi64>
  %init = tensor.empty() : tensor<1xi32>
  %gpu = hip.cast(%ctx) ins(%input : tensor<1xi64>) outs(%init : tensor<1xi32>)
      {to = 6 : i64} : tensor<1xi32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<1xi32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @widening_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @widening_unchanged(%ctx: !hip.context) -> i64 {
  %input = arith.constant dense<8> : tensor<i32>
  %init = tensor.empty() : tensor<i64>
  %gpu = hip.cast(%ctx) ins(%input : tensor<i32>) outs(%init : tensor<i64>)
      {to = 7 : i64} : tensor<i64>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i64>) -> i64
  return %host : i64
}

// CHECK-LABEL: func.func @float_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @float_unchanged(%ctx: !hip.context) -> i32 {
  %input = arith.constant dense<8.0> : tensor<f32>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%input : tensor<f32>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @different_context_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @different_context_unchanged(%ctx: !hip.context, %other: !hip.context) -> i32 {
  %input = arith.constant dense<8> : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%input : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%other, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// The rule does not interpret tensor encodings or other narrowing widths.
// CHECK-LABEL: func.func @encoded_input_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @encoded_input_unchanged(%ctx: !hip.context) -> i32 {
  %input = arith.constant dense<8> : tensor<i64, "test.encoding">
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%input : tensor<i64, "test.encoding">)
      outs(%init : tensor<i32>) {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @encoded_output_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @encoded_output_unchanged(%ctx: !hip.context) -> i32 {
  %input = arith.constant dense<8> : tensor<i64>
  %init = tensor.empty() : tensor<i32, "test.encoding">
  %gpu = hip.cast(%ctx) ins(%input : tensor<i64>)
      outs(%init : tensor<i32, "test.encoding">) {to = 6 : i64}
      : tensor<i32, "test.encoding">
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32, "test.encoding">) -> i32
  return %host : i32
}

// CHECK-LABEL: func.func @other_narrowing_unchanged
// CHECK: hip.cast
// CHECK: hip.readback_scalar
func.func @other_narrowing_unchanged(%ctx: !hip.context) -> i16 {
  %input = arith.constant dense<8> : tensor<i64>
  %init = tensor.empty() : tensor<i16>
  %gpu = hip.cast(%ctx) ins(%input : tensor<i64>) outs(%init : tensor<i16>)
      {to = 5 : i64} : tensor<i16>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i16>) -> i16
  return %host : i16
}
