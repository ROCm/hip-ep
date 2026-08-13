// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Error paths under the same options as cast.mlir, which covers the rewrite
// itself. A tensor without the #hipsr.mem<device> encoding bufferizes to a
// space-less memref that the operand constraint rejects.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s

// The DPS verifier accepts a buffer input next to a tensor init, which is
// neither fully bufferized already nor fully rewritable.
func.func @mixed_buffer_and_tensor(%ctx: !hipsr.context,
                                   %in: memref<4x8xf32, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<4x8xf16, #hipsr.mem<device>>
  // expected-error@+2 {{does not have pure tensor semantics}}
  // expected-error@+1 {{failed to bufferize op}}
  %0 = hipsr.cast(%ctx) ins(%in : memref<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>)
      : tensor<4x8xf16, #hipsr.mem<device>>
  return
}

// -----

// The input arrives space-less through the function boundary.
func.func @input_without_encoding(%ctx: !hipsr.context, %in: tensor<4x8xf32>) {
  %init = tensor.empty() : tensor<4x8xf16, #hipsr.mem<device>>
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref, but got 'memref<4x8xf32>'}}
  %0 = hipsr.cast(%ctx) ins(%in : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>)
      : tensor<4x8xf16, #hipsr.mem<device>>
  return
}

// -----

// The init is space-less because its allocation has no encoding to follow.
func.func @init_without_encoding(%ctx: !hipsr.context,
                                 %in: tensor<4x8xf32, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<4x8xf16>
  // expected-error@+1 {{operand #2 must be ranked tensor or device memref, but got 'memref<4x8xf16>'}}
  %0 = hipsr.cast(%ctx) ins(%in : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return
}
