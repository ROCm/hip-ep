// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The runtime stores an i1 mask as one byte per element, so the data-movement
// kernels move it like ui8 (element size 1) instead of failing to lower.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @gather_i1
  // CHECK: llvm.call @wrap_gather(
  func.func @gather_i1(%ctx: !hip.context, %data: memref<3x4xi1, 1>,
                       %indices: memref<2xi64, 1>, %output: memref<2x4xi1, 1>) {
    hip.gather(%ctx)
        ins(%data, %indices : memref<3x4xi1, 1>, memref<2xi64, 1>)
        outs(%output : memref<2x4xi1, 1>) {axis = 0 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @compress_i1
  // CHECK: llvm.call @wrap_compress(
  func.func @compress_i1(%ctx: !hip.context, %input: memref<3x2xi1, 1>,
                         %condition: memref<3xi1, 1>,
                         %output: memref<2x2xi1, 1>) {
    hip.compress(%ctx)
        ins(%input, %condition : memref<3x2xi1, 1>, memref<3xi1, 1>)
        outs(%output : memref<2x2xi1, 1>) {axis = 0 : i64, flatten = false}
    return
  }

  // CHECK-LABEL: llvm.func @gather_elements_i1
  // CHECK: llvm.call @wrap_gather_elements(
  func.func @gather_elements_i1(%ctx: !hip.context, %data: memref<2x2xi1, 1>,
                                %indices: memref<2x2xi32, 1>,
                                %output: memref<2x2xi1, 1>) {
    hip.gather_elements(%ctx)
        ins(%data, %indices : memref<2x2xi1, 1>, memref<2x2xi32, 1>)
        outs(%output : memref<2x2xi1, 1>) {axis = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @scatter_elements_i1
  // CHECK: llvm.call @wrap_scatter_elements(
  func.func @scatter_elements_i1(%ctx: !hip.context, %data: memref<2x2xi1, 1>,
                                 %indices: memref<2x2xi32, 1>,
                                 %updates: memref<2x2xi1, 1>,
                                 %output: memref<2x2xi1, 1>) {
    hip.scatter_elements(%ctx)
        ins(%data, %indices, %updates :
            memref<2x2xi1, 1>, memref<2x2xi32, 1>, memref<2x2xi1, 1>)
        outs(%output : memref<2x2xi1, 1>) {axis = 1 : i64, reduction = "none"}
    return
  }

  // CHECK-LABEL: llvm.func @one_hot_i1
  // CHECK: llvm.call @wrap_one_hot(
  func.func @one_hot_i1(%ctx: !hip.context, %indices: memref<2x2xi64, 1>,
                        %depth: memref<i64, 1>, %values: memref<2xi1, 1>,
                        %output: memref<2x10x2xi1, 1>) {
    hip.one_hot(%ctx)
        ins(%indices, %depth, %values :
            memref<2x2xi64, 1>, memref<i64, 1>, memref<2xi1, 1>)
        outs(%output : memref<2x10x2xi1, 1>) {axis = 1 : i64}
    return
  }
}
