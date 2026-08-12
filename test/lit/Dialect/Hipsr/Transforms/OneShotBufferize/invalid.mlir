// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Error paths of one-shot-bufferize on hipsr.cast. The rewrite itself is
// covered in cast.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics --one-shot-bufferize %s

// The DPS verifier accepts a buffer input next to a tensor init, but the
// rewrite has no answer for it: the op is neither fully bufferized already nor
// fully rewritable, so it refuses instead of dropping the tensor result.
func.func @mixed_buffer_and_tensor(%ctx: !hipsr.context,
                                   %in: memref<4x8xf32, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<4x8xf16>
  // expected-error@+2 {{does not have pure tensor semantics}}
  // expected-error@+1 {{failed to bufferize op}}
  %0 = hipsr.cast(%ctx) ins(%in : memref<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return
}

// -----

// OPEN QUESTION: nothing tells one-shot-bufferize to allocate in
// #hipsr.mem<device>, so a tensor init and a tensor input bufferize to
// space-less memrefs that the operand constraint rejects. This case becomes a
// positive test once we decide where the space comes from.
func.func @no_memory_space(%ctx: !hipsr.context, %in: tensor<4x8xf32>) {
  %init = tensor.empty() : tensor<4x8xf16>
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  %0 = hipsr.cast(%ctx) ins(%in : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return
}
