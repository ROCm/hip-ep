// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @ngram_engram(
      %ctx: !hip.context,
      %ids: memref<?x?xi64, 1>,
      %mult: memref<3xi64, 1>,
      %vocab: memref<16xi64, 1>,
      %past: memref<?x2xi64, 1>,
      %off: memref<16xi64, 1>,
      %eos: memref<i64, 1>,
      %hash: memref<?x?x16xi64, 1>,
      %present: memref<?x2xi64, 1>) {
    // CHECK-LABEL: llvm.func @ngram_engram
    hip.ngram_hash_mapping(%ctx)
        ins(%ids, %mult, %vocab :
            memref<?x?xi64, 1>, memref<3xi64, 1>, memref<16xi64, 1>)
        past_ids(%past : memref<?x2xi64, 1>)
        head_offsets(%off : memref<16xi64, 1>)
        eos_token_id(%eos : memref<i64, 1>)
        outs(%hash, %present : memref<?x?x16xi64, 1>, memref<?x2xi64, 1>)
        {max_ngram_size = 3 : i64, n_head_per_ngram = 8 : i64,
         pad_id = 248044 : i64, reset_on_eos = 1 : i64}
    // CHECK-DAG: %[[N:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK-DAG: %[[HEADS:.*]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK-DAG: %[[PAD:.*]] = llvm.mlir.constant(248044 : i64) : i64
    // CHECK-DAG: %[[RESET:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: %[[ELEM:.*]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK: llvm.call @wrap_ngram_hash_mapping({{.*}}, %[[N]], %[[HEADS]], %[[PAD]], %[[RESET]], %[[ELEM]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  func.func @ngram_minimal(
      %ctx: !hip.context,
      %ids: memref<2x4xi32, 1>,
      %mult: memref<3xi32, 1>,
      %vocab: memref<4xi32, 1>,
      %hash: memref<2x4x4xi32, 1>) {
    // CHECK-LABEL: llvm.func @ngram_minimal
    hip.ngram_hash_mapping(%ctx)
        ins(%ids, %mult, %vocab :
            memref<2x4xi32, 1>, memref<3xi32, 1>, memref<4xi32, 1>)
        outs(%hash : memref<2x4x4xi32, 1>)
        {max_ngram_size = 3 : i64, n_head_per_ngram = 2 : i64, pad_id = 9 : i64}
    // CHECK-COUNT-5: llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @wrap_ngram_hash_mapping
    return
  }
}
