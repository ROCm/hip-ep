// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s -split-input-file --convert-to-llvm 2>&1 | FileCheck %s

// CHECK: enclosing function has no arguments
module {
  func.func @no_ctx() -> memref<4xf32, #hipsr.mem<device>> {
    %c = hipsr.constant {value = dense<1.0> : tensor<4xf32>, offset = 0 : i64, size = 16 : i64, index = 0 : i64}
       : memref<4xf32, #hipsr.mem<device>>
    return %c : memref<4xf32, #hipsr.mem<device>>
  }
}

// -----

// CHECK: reached LLVM lowering without externalization
module {
  func.func @not_externalized(%ctx: !llvm.ptr) -> memref<4xf32, #hipsr.mem<device>> {
    %c = hipsr.constant {value = dense<1.0> : tensor<4xf32>}
       : memref<4xf32, #hipsr.mem<device>>
    return %c : memref<4xf32, #hipsr.mem<device>>
  }
}
