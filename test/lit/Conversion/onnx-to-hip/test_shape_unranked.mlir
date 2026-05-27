// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Negative-case lockdown: an unranked tensor in @main_graph's signature must
// produce a hard error, not silent acceptance.  The first pipeline pass
// (--hip-add-context-arg) rejects unranked tensors before --convert-onnx-to-hip
// even runs, which is the desired behaviour: the dynseqlen pipeline requires
// ranked tensors with at most one dynamic dim per output (resolvable via
// DimSource).  Silently accepting unranked input would let onnx.Shape, onnx.Dim,
// or any other shape-dependent op fall through to a CPU-fallback no-op.
//
// If a future change relaxes --hip-add-context-arg to accept unranked types,
// this CHECK will fail loudly so the regression is caught at compile time.
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

// CHECK: error: non-tensor input type in @main_graph: 'tensor<*xf16>'

module {
  func.func @main_graph(%arg0: tensor<*xf16>) -> tensor<*xf16> {
    return %arg0 : tensor<*xf16>
  }
}
