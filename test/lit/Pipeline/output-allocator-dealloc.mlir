// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// R2 gate: pins the buffer-deallocation vs hip-use-output-allocator ordering.
//
// hip.alloc_output carries a Write effect but NO Allocate effect (so the
// ownership-based deallocation pass never FREES the EP-owned output buffer).
// But "not freed" is not the whole story: a value returned by func.return whose
// defining op has no Allocate effect is treated as UNOWNED, so if
// hip-use-output-allocator runs BEFORE buffer-deallocation the deallocation
// pass inserts a `bufferization.clone` at the return (returning the clone, not
// the EP buffer) -- an extra per-inference alloc + full-output copy that
// defeats the allocator's zero-copy goal.
//
// Therefore the onnx-to-hip allocator pipeline runs hip-use-output-allocator
// AFTER buffer-deallocation (slot 4.5), when the output is still a
// memref.alloc (Allocate effect => owned, returned directly with no clone).
// This file characterizes BOTH orderings against the same input so the choice
// is a committed, regression-guarded decision record. See
// lib/Dialect/Transforms/Pipelines.cpp (buildOnnxToHipPipelineTail) and
// docs/design/output-allocator-design.md.
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
