// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Common HIP cast and matmul wrappers return a status. A source pipeline that
// opts into fail-closed device copies must propagate those statuses too.
// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/main.mlir --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s --check-prefix=STATUS
// RUN: not hip-mlir-opt %t/no-metadata.mlir --assign-op-state-slots --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=NO-METADATA

// STATUS-LABEL: llvm.func private @main_graph_internal
// STATUS: %[[CAST_STATUS:.*]] = llvm.call @wrap_cast
// STATUS-SAME: {hip.runtime.fallible}
// STATUS: %[[CAST_FAILED:.*]] = llvm.icmp "ne" %[[CAST_STATUS]], %{{.*}} : i32
// STATUS: llvm.cond_br %[[CAST_FAILED]], ^bb{{[0-9]+}}(%[[CAST_STATUS]] : i32), ^bb
// STATUS: %[[MATMUL_STATUS:.*]] = llvm.call @wrap_hipblasLtMatmul
// STATUS-SAME: {hip.runtime.fallible}
// STATUS: %[[MATMUL_FAILED:.*]] = llvm.icmp "ne" %[[MATMUL_STATUS]], %{{.*}} : i32
// STATUS: llvm.cond_br %[[MATMUL_FAILED]], ^bb{{[0-9]+}}(%[[MATMUL_STATUS]] : i32), ^bb
// STATUS: llvm.return %{{.*}} : i32

//--- main.mlir
module attributes {
  hip.runtime.require_fallible_memref_copies,
  hipdnn.input_count = 4 : i64,
  hipdnn.output_count = 0 : i64,
  hipdnn.input_shapes = [array<i64: 1, 1, 1, 8>,
                         array<i64: 1, 1, 1, 8>,
                         array<i64: 1, 1, 4, 8>,
                         array<i64: 1, 1, 1, 4>],
  hipdnn.output_shapes = []
} {
  func.func @main_graph(
      %ctx: !hip.context,
      %input: memref<1x1x1x8xf32>,
      %cast: memref<1x1x1x8xbf16>,
      %weight: memref<1x1x4x8xbf16>,
      %output: memref<1x1x1x4xf32>) {
    hip.cast(%ctx)
        ins(%input : memref<1x1x1x8xf32>)
        outs(%cast : memref<1x1x1x8xbf16>)
        {to = 16 : i64}
    hip.matmul(%ctx)
        ins(%cast, %weight : memref<1x1x1x8xbf16>,
             memref<1x1x4x8xbf16>)
        outs(%output : memref<1x1x1x4xf32>)
        {transB = 1 : i64}
    return
  }
}

//--- no-metadata.mlir
// NO-METADATA: error: fallible runtime call requires @main_graph and complete hipdnn input/output metadata for status propagation
module attributes {hip.runtime.require_fallible_memref_copies} {
  func.func @cast_without_status_abi(
      %ctx: !hip.context,
      %input: memref<1x1x1x8xf32>,
      %output: memref<1x1x1x8xbf16>) {
    hip.cast(%ctx)
        ins(%input : memref<1x1x1x8xf32>)
        outs(%output : memref<1x1x1x8xbf16>)
        {to = 16 : i64}
    return
  }
}
