// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// End-to-end pipeline test: bufferization -> buffer reuse -> alloc lowering.
//===----------------------------------------------------------------------===//

// RUN: %hip-opt --one-shot-bufferize="bufferize-function-boundaries" --hip-optimize-memrefs --hip-lower-allocs %s | %FileCheck %s

// The attention model has 8 tensor.empty + HIP ops. After the pipeline:
//   - one-shot-bufferize: 8 memref.alloc
//   - hip-optimize-memrefs: reduced to 4 memref.alloc
//   - hip-lower-allocs: 4 hip.alloc, 3 hip.free (one buffer is returned)

// CHECK-LABEL: func.func @attention_pipeline

// Exactly 4 hip.alloc ops after reuse.
// CHECK-COUNT-4: hip.alloc

// No memref.alloc should remain after lowering.
// CHECK-NOT:   memref.alloc

// Exactly 3 hip.free ops (the returned buffer is not freed).
// CHECK-COUNT-3: hip.free

// CHECK:       hip.destroy_handle
// CHECK:       return

func.func @attention_pipeline(
    %X: tensor<2x64x64xf32>,
    %Wq: tensor<64x64xf32>,
    %Wk: tensor<64x64xf32>,
    %Wv: tensor<64x64xf32>,
    %scale: tensor<f32>) -> tensor<2x64x64xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %handle = hip.create_handle() : !hip.handle

  %e0 = tensor.empty() : tensor<2x64x64xf32>
  %Q = hip.hipblaslt.matmul(%handle) ins(%X, %Wq : tensor<2x64x64xf32>, tensor<64x64xf32>) outs(%e0 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e1 = tensor.empty() : tensor<2x64x64xf32>
  %K = hip.hipblaslt.matmul(%handle) ins(%X, %Wk : tensor<2x64x64xf32>, tensor<64x64xf32>) outs(%e1 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e2 = tensor.empty() : tensor<2x64x64xf32>
  %V = hip.hipblaslt.matmul(%handle) ins(%X, %Wv : tensor<2x64x64xf32>, tensor<64x64xf32>) outs(%e2 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e3 = tensor.empty() : tensor<2x64x64xf32>
  %KT = hip.transpose(%handle, %c1, %c2) ins(%K : tensor<2x64x64xf32>) outs(%e3 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e4 = tensor.empty() : tensor<2x64x64xf32>
  %scores = hip.hipblaslt.matmul(%handle) ins(%Q, %KT : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%e4 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e5 = tensor.empty() : tensor<2x64x64xf32>
  %scaled = hip.miopen.mul(%handle) ins(%scores, %scale : tensor<2x64x64xf32>, tensor<f32>) outs(%e5 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e6 = tensor.empty() : tensor<2x64x64xf32>
  %probs = hip.miopen.softmax(%handle) ins(%scaled : tensor<2x64x64xf32>) outs(%e6 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  %e7 = tensor.empty() : tensor<2x64x64xf32>
  %out = hip.hipblaslt.matmul(%handle) ins(%probs, %V : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%e7 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  hip.destroy_handle(%handle) : !hip.handle
  return %out : tensor<2x64x64xf32>
}
