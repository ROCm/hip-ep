// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Full bufferization and pooling coverage for transient in-place softmax.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries" --hip-use-output-allocator --hip-optimize-memrefs --hip-pool-allocs %s | FileCheck %s

// The score tensor is 16 * 4096 * 4096 * sizeof(f16) = 512 MiB. Its
// softmax destination aliases the same pool view, while the final graph output
// is runtime-owned and therefore excluded from the transient pool.
// CHECK-LABEL: func.func @attention_scores
// CHECK:         %[[POOL_SIZE:.*]] = arith.constant 536870912 : index
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}%[[POOL_SIZE]]) : memref<?xi8>
// CHECK:         %[[SCORES:.*]] = memref.view %[[POOL]]
// CHECK-NOT:     memref.view %[[POOL]]
// CHECK:         hip.matmul{{.*}}outs(%[[SCORES]] :
// CHECK:         hip.miopen.softmax{{.*}}ins(%[[SCORES]] :{{.*}}outs(%[[SCORES]] :
// CHECK:         %[[OUTPUT:.*]] = hip.alloc_output
// CHECK:         hip.matmul{{.*}}ins(%[[SCORES]],{{.*}}outs(%[[OUTPUT]] :
// CHECK:         return %[[OUTPUT]]
func.func @attention_scores(
    %ctx: !hip.context,
    %query: tensor<16x4096x128xf16>,
    %key: tensor<16x128x4096xf16>,
    %value: tensor<16x4096x128xf16>) -> tensor<16x4096x128xf16> {
  %score_init = tensor.empty() : tensor<16x4096x4096xf16>
  %scores = hip.matmul(%ctx) ins(%query, %key : tensor<16x4096x128xf16>, tensor<16x128x4096xf16>) outs(%score_init : tensor<16x4096x4096xf16>) : tensor<16x4096x4096xf16>

  %probability_init = tensor.empty() : tensor<16x4096x4096xf16>
  %probabilities = hip.miopen.softmax(%ctx) ins(%scores : tensor<16x4096x4096xf16>) outs(%probability_init : tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16>

  %output_init = tensor.empty() : tensor<16x4096x128xf16>
  %output = hip.matmul(%ctx) ins(%probabilities, %value : tensor<16x4096x4096xf16>, tensor<16x4096x128xf16>) outs(%output_init : tensor<16x4096x128xf16>) : tensor<16x4096x128xf16>
  return %output : tensor<16x4096x128xf16>
}
