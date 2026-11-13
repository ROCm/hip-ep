// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm --cse | FileCheck %s

// hip.rms_norm_add lowers to one wrap_rms_norm_add call: the input, scale,
// residual and output buffers, the input and scale element counts, the
// reduction width hip.rms_norm derives from `axis`, the element size and
// epsilon. With axis = -1 the width is the last dim, which here is also the
// scale's element count.

module {
  // CHECK-LABEL: llvm.func @rms_norm_add_decode
  // CHECK-DAG: %[[ES:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-DAG: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
  // CHECK: llvm.call @wrap_rms_norm_add(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[W:[0-9]+]], %[[W]], %[[ES]], %[[EPS]]) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32) -> i32
  func.func @rms_norm_add_decode(%ctx: !hip.context,
                                 %x: memref<1x?x2560xf16, 1>,
                                 %g: memref<2560xf16, 1>,
                                 %r: memref<1x?x2560xf16, 1>,
                                 %y: memref<1x?x2560xf16, 1>) {
    hip.rms_norm_add(%ctx) ins(%x, %g, %r : memref<1x?x2560xf16, 1>,
                                memref<2560xf16, 1>, memref<1x?x2560xf16, 1>)
        outs(%y : memref<1x?x2560xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999997E-7 : f32}
    return
  }
}
