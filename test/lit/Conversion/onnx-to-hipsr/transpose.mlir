// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Transpose becomes hipsr.transpose, which always carries perm. Rejected
// forms are in transpose-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// An embedding graph turns NonZero's [rank, count] index matrix into the
// [count, rank] layout ScatterND takes. The placeholder gets no shape region
// here: hipsr.transpose is DPS, so hipsr-populate-shape-region fills it in
// later.
// CHECK-LABEL: func.func @explicit_perm(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.+]]: tensor<3x?xi64, #hipsr.mem<device>>) -> tensor<?x3xi64, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.transpose(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @explicit_perm(%ctx: !hipsr.context,
                         %input: tensor<3x?xi64>) -> tensor<?x3xi64> {
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<3x?xi64>) -> tensor<?x3xi64>
  "onnx.Return"(%0) : (tensor<?x3xi64>) -> ()
}

// -----

// An absent perm means the reverse permutation, which the hipsr op spells out.
// CHECK-LABEL: func.func @default_perm(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.+]]: tensor<2x3x4xf16, #hipsr.mem<device>>) -> tensor<4x3x2xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.transpose(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<4x3x2xf16, #hipsr.mem<device>>) {perm = array<i64: 2, 1, 0>} : tensor<4x3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @default_perm(%ctx: !hipsr.context,
                        %input: tensor<2x3x4xf16>) -> tensor<4x3x2xf16> {
  %0 = "onnx.Transpose"(%input) : (tensor<2x3x4xf16>) -> tensor<4x3x2xf16>
  "onnx.Return"(%0) : (tensor<4x3x2xf16>) -> ()
}
