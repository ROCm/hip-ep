// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Isolated pass-interaction record -- NOT the production pipeline.
//
// The onnx-to-hip pipeline no longer runs ownership-based buffer deallocation
// at all: hip-pool-allocs owns every buffer lifetime, so there is nothing to
// deallocate (see lib/Dialect/Transforms/Pipelines.cpp buildOnnxToHipPipelineTail).
// This file is retained only to document the pass INTERACTION that motivated the
// output-allocator's placement, should deallocation ever be reintroduced:
//
// hip.alloc_output carries a Write effect but NO Allocate effect, so a value
// returned by func.return whose defining op has no Allocate effect is treated
// as UNOWNED. If hip-use-output-allocator ran BEFORE buffer-deallocation, the
// deallocation pass would insert a `bufferization.clone` at the return
// (returning the clone, not the EP buffer) -- an extra per-inference alloc +
// full-output copy defeating zero-copy; running AFTER leaves the output a plain
// memref.alloc (owned, returned directly, no clone). This file characterizes
// BOTH orderings against the same input.
//===----------------------------------------------------------------------===//

// Chosen ordering (deallocation THEN allocator): clean, zero-copy.
// RUN: hip-mlir-opt --buffer-deallocation-pipeline --hip-use-output-allocator %s 2>&1 | FileCheck %s

// Rejected ordering (allocator THEN deallocation): clones the output. Pinned
// so a future change back to this order (or a doc claiming it is safe) trips.
// RUN: hip-mlir-opt --hip-use-output-allocator --buffer-deallocation-pipeline %s 2>&1 | FileCheck --check-prefix=BADORDER %s

// --- Chosen ordering: the returned hip.alloc_output is the value written by
//     the last op (sigmoid) AND the value returned, with no clone between. ---
// CHECK-LABEL: func.func @main_graph
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%{{.*}}, %{{.*}}) {out_idx = 0 : i64} : memref<?x64xf16>
// CHECK:         hip.sigmoid(%{{.*}}) ins(%{{.*}} : memref<?x64xf16>) outs(%[[OUT]] : memref<?x64xf16>)
// CHECK-NOT:     bufferization.clone
// CHECK:         return %[[OUT]] : memref<?x64xf16>

// --- Rejected ordering: deallocation inserts a clone of the unowned
//     hip.alloc_output result and returns the clone. ---
// BADORDER-LABEL: func.func @main_graph
// BADORDER:         hip.alloc_output(%{{.*}}, %{{.*}}) {out_idx = 0 : i64} : memref<?x64xf16>
// BADORDER:         bufferization.clone

// Design-doc canonical case: Add -> MatMul -> Sigmoid, two pooled
// intermediates (%t0, %t1) plus one returned graph output (%out).
func.func @main_graph(%ctx: !hip.context,
                      %input: memref<?x64xf16>,
                      %bias: memref<?x64xf16>,
                      %w: memref<64x64xf16>) -> memref<?x64xf16> {
  %c0 = arith.constant 0 : index
  %M = memref.dim %input, %c0 : memref<?x64xf16>
  %t0 = memref.alloc(%M) : memref<?x64xf16>
  hip.add(%ctx) ins(%input, %bias : memref<?x64xf16>, memref<?x64xf16>)
               outs(%t0 : memref<?x64xf16>)
  %t1 = memref.alloc(%M) : memref<?x64xf16>
  hip.matmul(%ctx) ins(%t0, %w : memref<?x64xf16>, memref<64x64xf16>)
                  outs(%t1 : memref<?x64xf16>)
  %out = memref.alloc(%M) : memref<?x64xf16>
  hip.sigmoid(%ctx) ins(%t1 : memref<?x64xf16>) outs(%out : memref<?x64xf16>)
  return %out : memref<?x64xf16>
}
