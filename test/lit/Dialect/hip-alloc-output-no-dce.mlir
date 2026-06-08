// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pins the memory-effect contract of hip.alloc_output: the Write effect must
// keep the op alive through later optimization. If the op were effect-free it
// would be trivially dead (removed by DCE/canonicalize when its result is
// unused) and mergeable by CSE -- neither is allowed for an EP allocator call.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --canonicalize %s | FileCheck %s
// RUN: hip-mlir-opt --cse %s | FileCheck --check-prefix=CSE %s

// --- Unused result: must survive canonicalize/DCE (Write effect). ---
// CHECK-LABEL: func.func @keeps_unused_alloc_output
// CHECK:         hip.alloc_output
func.func @keeps_unused_alloc_output(%ctx: !hip.context, %m: index) {
  %0 = hip.alloc_output(%ctx, %m) {out_idx = 0 : i64} : memref<?xf16>
  return
}

// --- Two identical ops: CSE must NOT merge them (side-effecting). ---
// CSE-LABEL: func.func @no_cse_alloc_output
// CSE-COUNT-2: hip.alloc_output
func.func @no_cse_alloc_output(%ctx: !hip.context, %m: index)
    -> (memref<?xf16>, memref<?xf16>) {
  %0 = hip.alloc_output(%ctx, %m) {out_idx = 0 : i64} : memref<?xf16>
  %1 = hip.alloc_output(%ctx, %m) {out_idx = 0 : i64} : memref<?xf16>
  return %0, %1 : memref<?xf16>, memref<?xf16>
}
