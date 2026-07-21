// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Negative tests: --convert-hipsr-to-llvm must diagnose (not silently
// mis-lower) the hipsr.constant states it cannot handle. Each case emits a
// precise error and the pass fails; the framework's follow-on "failed to
// legalize" diagnostic is tolerated (FileCheck only matches our message).
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt %s -split-input-file --convert-hipsr-to-llvm 2>&1 | FileCheck %s

// An externalized constant needs the runtime context, taken from the enclosing
// function's !hip.context argument. A function without one cannot be lowered.
// CHECK: no !hip.context argument in enclosing function
module {
  func.func @no_ctx() -> memref<4xf32, #hipsr.mem<device>> {
    %c = hipsr.constant {value = dense<1.0> : tensor<4xf32>, offset = 0 : i64, size = 16 : i64}
       : memref<4xf32, #hipsr.mem<device>>
    return %c : memref<4xf32, #hipsr.mem<device>>
  }
}

// -----

// A constant that still wants to be externalized (offset/size not yet assigned)
// reached this pass before the externalization pass ran.
// CHECK: reached --convert-hipsr-to-llvm without externalization
module {
  func.func @not_externalized(%ctx: !hip.context) -> memref<4xf32, #hipsr.mem<device>> {
    %c = hipsr.constant {value = dense<1.0> : tensor<4xf32>}
       : memref<4xf32, #hipsr.mem<device>>
    return %c : memref<4xf32, #hipsr.mem<device>>
  }
}

// -----

// An inline (disableExternalize) constant with a device-memref result is not
// supported yet: arith.constant is illegal on memref, so this needs a
// memref.global follow-up.
// CHECK: inline (disableExternalize) lowering only supports a ranked-tensor result
module {
  func.func @inline_memref(%ctx: !hip.context) -> memref<4xf32, #hipsr.mem<device>> {
    %c = hipsr.constant {value = dense<1.0> : tensor<4xf32>, externalize = false}
       : memref<4xf32, #hipsr.mem<device>>
    return %c : memref<4xf32, #hipsr.mem<device>>
  }
}
