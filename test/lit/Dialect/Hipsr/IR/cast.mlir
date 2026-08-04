// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s | FileCheck %s

// Cast round-trips without a shape region.
// CHECK-LABEL: func.func @cast(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>,
// CHECK-SAME: %[[INIT:.+]]: tensor<?x8xf16>) -> tensor<?x8xf16> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<?x8xf16>
// CHECK-NEXT: }
func.func @cast(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %result = hipsr.cast(%ctx)
      ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}
