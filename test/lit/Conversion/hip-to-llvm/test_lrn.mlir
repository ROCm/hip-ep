// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.lrn -> wrap_lrn
//
// 1x64x56x56 is the first GoogLeNet LRN. spatial is 56 * 56. FLOAT is
// HIPDNN_EP_DATATYPE_FLOAT = 0. size is 5.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @lrn_googlenet_64
  // CHECK-DAG: llvm.mlir.constant(64 : i64)
  // CHECK-DAG: llvm.mlir.constant(56 : i64)
  // CHECK-DAG: llvm.mlir.constant(5 : i64)
  // CHECK: llvm.call @wrap_lrn
  func.func @lrn_googlenet_64(%ctx: !hip.context,
                              %x: memref<1x64x56x56xf32, 1>,
                              %y: memref<1x64x56x56xf32, 1>) {
    hip.lrn(%ctx) ins(%x : memref<1x64x56x56xf32, 1>)
                  outs(%y : memref<1x64x56x56xf32, 1>) {size = 5 : i64}
    return
  }
}
