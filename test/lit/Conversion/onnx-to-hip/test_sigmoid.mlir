// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Sigmoid is correctly lowered to hip.sigmoid.
//
// Assertions:
// - context argument prepended
// - input and output tensor types preserved
// - tensor.empty() used as output init (no hip.alloc)
// - hip.sigmoid result type matches input type
// ============================================================================

// RUN: udna-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16> {
    %output = "onnx.Sigmoid"(%input) : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    return %output : tensor<1x128x14336xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context,
// CHECK-SAME: %[[INPUT:.*]]: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
// CHECK: tensor.empty() : tensor<1x128x14336xf16>
// CHECK: hip.sigmoid(%[[CTX]], %[[INPUT]], {{.*}}) : (!hip.context, tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) : tensor<1x128x14336xf16>
// CHECK-NOT: hip.alloc
