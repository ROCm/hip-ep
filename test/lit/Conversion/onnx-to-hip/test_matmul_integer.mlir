// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
    return %arg0 : tensor<1x4xui8>
  }

  // K = 300 exceeds the f32 mantissa. Scalar activation zero point,
  // per-column weight zero point, i32 result.
  func.func @matmul_integer_wide_per_column(
      %a: tensor<1x300xui8>, %b: tensor<300x4xui8>,
      %azp: tensor<ui8>, %bzp: tensor<4xui8>) -> tensor<1x4xi32> {
    %y = "onnx.MatMulInteger"(%a, %b, %azp, %bzp) : (
        tensor<1x300xui8>, tensor<300x4xui8>, tensor<ui8>, tensor<4xui8>
      ) -> tensor<1x4xi32>
    return %y : tensor<1x4xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_wide_per_column
  // K = 300 exceeds the f32 mantissa, so the matmul stays in i32.
  // CHECK: hip.cast
  // CHECK: hip.sub
  // The length-N zero point is reshaped to [1, N] so it broadcasts on the
  // columns, which the 4D elementwise ABI would also do for a bare vector.
  // CHECK: tensor<1x4xi32>
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<1x300xi32>
  // CHECK-SAME: tensor<300x4xi32>
  // CHECK-SAME: tensor<1x4xi32>
  // CHECK-NOT: onnx.MatMulInteger

  // A length-M zero point has to move to the row axis. Left as rank 1 it
  // would right-align onto K.
  func.func @matmul_integer_per_row(
      %a: tensor<2x4xui8>, %b: tensor<4x3xui8>,
      %azp: tensor<2xui8>) -> tensor<2x3xi32> {
    %y = "onnx.MatMulInteger"(%a, %b, %azp) : (
        tensor<2x4xui8>, tensor<4x3xui8>, tensor<2xui8>
      ) -> tensor<2x3xi32>
    return %y : tensor<2x3xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_per_row
  // CHECK: tensor<2x1xf32>
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<2x3xf32>
  // CHECK-NOT: onnx.MatMulInteger

  // Omitted zero points are defined as 0, so the subtract is dropped.
  func.func @matmul_integer_no_zp(
      %a: tensor<1x2xi8>, %b: tensor<2x1xi8>) -> tensor<1x1xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (
        tensor<1x2xi8>, tensor<2x1xi8>
      ) -> tensor<1x1xi32>
    return %y : tensor<1x1xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_no_zp
  // CHECK-NOT: hip.sub
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<1x2xf32>
  // CHECK-NOT: onnx.MatMulInteger

  // Leading batch dims broadcast the same way as onnx.MatMul.
  func.func @matmul_integer_batch(
      %a: tensor<1x2x3xui8>, %b: tensor<4x3x5xui8>,
      %azp: tensor<ui8>, %bzp: tensor<ui8>) -> tensor<4x2x5xi32> {
    %y = "onnx.MatMulInteger"(%a, %b, %azp, %bzp) : (
        tensor<1x2x3xui8>, tensor<4x3x5xui8>, tensor<ui8>, tensor<ui8>
      ) -> tensor<4x2x5xi32>
    return %y : tensor<4x2x5xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_batch
  // A's batch of 1 is expanded to B's batch of 4 before the matmul.
  // CHECK: hip.expand
  // CHECK-SAME: tensor<4x2x3xf32>
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<4x2x3xf32>
  // CHECK-SAME: tensor<4x3x5xf32>
  // CHECK-SAME: tensor<4x2x5xf32>
  // CHECK-NOT: onnx.MatMulInteger

  // K = 259 is the first length whose uint8 dot product is not an exact f32.
  func.func @matmul_integer_wide_k(
      %a: tensor<1x259xui8>, %b: tensor<259x1xui8>) -> tensor<1x1xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (
        tensor<1x259xui8>, tensor<259x1xui8>
      ) -> tensor<1x1xi32>
    return %y : tensor<1x1xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_wide_k
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<1x259xi32>
  // CHECK-SAME: tensor<259x1xi32>
  // CHECK-SAME: tensor<1x1xi32>
  // CHECK-NOT: onnx.MatMulInteger

  // Rank 1 is numpy's vector promotion. hip.matmul requires rank >= 2.
  func.func @matmul_integer_rank1_rejected(
      %a: tensor<4xui8>, %b: tensor<4x2xui8>) -> tensor<2xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (
        tensor<4xui8>, tensor<4x2xui8>
      ) -> tensor<2xi32>
    return %y : tensor<2xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_rank1_rejected
  // CHECK: onnx.MatMulInteger

  // A rank-2 B is one matrix. It is not copied into A's batch.
  func.func @matmul_integer_rank2_b(
      %a: tensor<2x2x3xui8>, %b: tensor<3x5xui8>) -> tensor<2x2x5xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (
        tensor<2x2x3xui8>, tensor<3x5xui8>
      ) -> tensor<2x2x5xi32>
    return %y : tensor<2x2x5xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_rank2_b
  // CHECK-NOT: hip.expand
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<2x2x3xf32>, tensor<3x5xf32>
  // CHECK-NOT: onnx.MatMulInteger

  // A rank-3 B with only size-one batch dimensions is a shared matrix too.
  func.func @matmul_integer_singleton_batch_b(
      %a: tensor<32x1x4xui8>, %b: tensor<1x4x4xui8>) -> tensor<32x1x4xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (
        tensor<32x1x4xui8>, tensor<1x4x4xui8>
      ) -> tensor<32x1x4xi32>
    return %y : tensor<32x1x4xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_singleton_batch_b
  // CHECK-NOT: hip.expand
  // CHECK: tensor.reshape
  // CHECK-SAME: tensor<4x4xf32>
  // CHECK-NOT: hip.expand
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<32x1x4xf32>, tensor<4x4xf32>
  // CHECK-NOT: onnx.MatMulInteger

  // A runtime-sized leading batch on A is preserved. B is a shared rank-2
  // weight, so hip.matmul receives it once and takes its batch count from A.
  // K = 300 selects the i32 accumulator path.
  func.func @matmul_integer_dynamic_batch(
      %a: tensor<?x2x300xui8>, %b: tensor<300x4xi8>,
      %azp: tensor<ui8>, %bzp: tensor<i8>) -> tensor<?x2x4xi32> {
    %y = "onnx.MatMulInteger"(%a, %b, %azp, %bzp) : (
        tensor<?x2x300xui8>, tensor<300x4xi8>, tensor<ui8>, tensor<i8>
      ) -> tensor<?x2x4xi32>
    return %y : tensor<?x2x4xi32>
  }

  // CHECK-LABEL: func.func @matmul_integer_dynamic_batch
  // CHECK: tensor.empty(%{{.*}}) : tensor<?x2x300xi32>
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<?x2x300xi32>, tensor<300x4xi32>
  // CHECK-SAME: tensor<?x2x4xi32>
  // CHECK-NOT: hip.expand
  // CHECK-NOT: onnx.MatMulInteger
}
