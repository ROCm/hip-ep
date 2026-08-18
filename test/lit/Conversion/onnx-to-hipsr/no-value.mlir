// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.NoValue stands in for an operand the exporter omitted. It stays legal
// through the conversion so each consumer can drop it, and the pass erases the
// placeholders left dead afterwards.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file %s | FileCheck %s

// An unused placeholder is erased. The return directly after the signature is
// what proves it.
// CHECK-LABEL: func.func @dead_placeholder(
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<2x3xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    return %[[INPUT]] : tensor<2x3xf16, #hipsr.mem<device>>
func.func @dead_placeholder(%input: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  return %input : tensor<2x3xf16>
}
