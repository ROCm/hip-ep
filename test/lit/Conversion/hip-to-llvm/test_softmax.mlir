// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// The standalone Softmax runtime returns status. Its lowering must preserve
// that i32 ABI and route the result through the shared recorder.
// CHECK-LABEL: llvm.func @softmax_f16
// CHECK: %[[STATUS:.*]] = llvm.call @hip_miopen_softmax({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]])

module {
  func.func @softmax_f16(%ctx: !hip.context,
                         %input: memref<2x4xf16, 1>,
                         %output: memref<2x4xf16, 1>) {
    hip.miopen.softmax(%ctx)
        ins(%input : memref<2x4xf16, 1>)
        outs(%output : memref<2x4xf16, 1>)
    return
  }
}
