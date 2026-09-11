// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true
// ============================================================================
// hip.qconv -> wrap_qconv
//
// The ABI is narrower than wrap_conv's: a 1x1 kernel with unit stride, unit
// dilation and no padding needs no per-axis geometry, only how many output
// positions there are. Unit stride and zero padding also make the input's
// spatial extents identical, so one count covers both sides.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// ============================================================================

module {

// ===== 1: W4A16, the shape the fusion actually produces =====
// Weights and their zero points keep an i8 element type and their LOGICAL
// shape; packed_int4 is the only thing that says a byte holds two values, and
// it reaches the runtime as weight_bits = 4.
//
// Activation storage ui16 must arrive as HIPDNN_EP_DATATYPE_UINT16 = 9, and the
// signed i8 weights as INT8 = 5 -- the signedness decides how a nibble widens,
// so it cannot be dropped. Absent bias passes the UNSUPPORTED sentinel (-1) to
// mean "no bias type" alongside the null pointer.

// CHECK-LABEL: llvm.func @qconv_w4a16
// CHECK-DAG:   llvm.mlir.constant(2048 : i64)
// CHECK-DAG:   llvm.mlir.constant(1024 : i64)
// CHECK-DAG:   llvm.mlir.constant(128 : i64)
// CHECK-DAG:   llvm.mlir.constant(9 : i64)
// CHECK-DAG:   llvm.mlir.constant(5 : i64)
// CHECK-DAG:   llvm.mlir.constant(4 : i64)
// CHECK-DAG:   llvm.mlir.constant(-1 : i64)
// CHECK-DAG:   llvm.mlir.constant(35275 : i64)
// CHECK-DAG:   llvm.mlir.constant(36322 : i64)
// CHECK-DAG:   llvm.mlir.zero : !llvm.ptr
// CHECK:       llvm.call @wrap_qconv({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, f32, i64) -> i32
  func.func @qconv_w4a16(%ctx: !hip.context,
                         %in: memref<1x2048x1x128xui16, 1>,
                         %w: memref<1024x2048x1x1xi8, 1>,
                         %wscale: memref<1024xf32, 1>,
                         %wzp: memref<1024xi8, 1>,
                         %out: memref<1x1024x1x128xui16, 1>) {
    hip.qconv(%ctx) ins(%in, %w, %wscale, %wzp :
                        memref<1x2048x1x128xui16, 1>, memref<1024x2048x1x1xi8, 1>,
                        memref<1024xf32, 1>, memref<1024xi8, 1>)
                    outs(%out : memref<1x1024x1x128xui16, 1>)
                    {input_scale = 1.638800e-04 : f32, input_zp = 35275 : i64,
                     output_scale = 3.687020e-04 : f32, output_zp = 36322 : i64,
                     weight_axis = 0 : i64, kernel_shape = [1, 1],
                     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
                     group = 1 : i64, packed_int4}
    return
  }

// ===== 2: a bias reaches the call as a real pointer and a named dtype =====
// The fusion cannot produce this yet (it matches a two-operand Conv), so this
// is the only cover for the optional operand: f32 bias is FLOAT = 0, and there
// must be no llvm.mlir.zero standing in for the pointer.

// CHECK-LABEL: llvm.func @qconv_with_bias
// CHECK-NOT:   llvm.mlir.zero
// CHECK:       llvm.call @wrap_qconv
  func.func @qconv_with_bias(%ctx: !hip.context,
                             %in: memref<1x2048x1x128xui16, 1>,
                             %w: memref<1024x2048x1x1xi8, 1>,
                             %wscale: memref<1024xf32, 1>,
                             %wzp: memref<1024xi8, 1>,
                             %bias: memref<1024xf32, 1>,
                             %out: memref<1x1024x1x128xui16, 1>) {
    hip.qconv(%ctx) ins(%in, %w, %wscale, %wzp, %bias :
                        memref<1x2048x1x128xui16, 1>, memref<1024x2048x1x1xi8, 1>,
                        memref<1024xf32, 1>, memref<1024xi8, 1>,
                        memref<1024xf32, 1>)
                    outs(%out : memref<1x1024x1x128xui16, 1>)
                    {input_scale = 1.638800e-04 : f32, input_zp = 35275 : i64,
                     output_scale = 3.687020e-04 : f32, output_zp = 36322 : i64,
                     weight_axis = 0 : i64, kernel_shape = [1, 1],
                     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
                     group = 1 : i64, packed_int4}
    return
  }

// ===== 3: a dynamic token dimension keeps spatial_size a runtime product =====
// A shape-specialised LLM graph leaves the token extent dynamic, so the count
// has to come out of the memref descriptor. Folding it on the host would be
// wrong here, which is why there is a multiply rather than a constant.

// CHECK-LABEL: llvm.func @qconv_dynamic_tokens
// CHECK:       llvm.mul
// CHECK:       llvm.call @wrap_qconv
  func.func @qconv_dynamic_tokens(%ctx: !hip.context,
                                  %in: memref<1x2048x1x?xui16, 1>,
                                  %w: memref<1024x2048x1x1xi8, 1>,
                                  %wscale: memref<1024xf32, 1>,
                                  %wzp: memref<1024xi8, 1>,
                                  %out: memref<1x1024x1x?xui16, 1>) {
    hip.qconv(%ctx) ins(%in, %w, %wscale, %wzp :
                        memref<1x2048x1x?xui16, 1>, memref<1024x2048x1x1xi8, 1>,
                        memref<1024xf32, 1>, memref<1024xi8, 1>)
                    outs(%out : memref<1x1024x1x?xui16, 1>)
                    {input_scale = 1.638800e-04 : f32, input_zp = 35275 : i64,
                     output_scale = 3.687020e-04 : f32, output_zp = 36322 : i64,
                     weight_axis = 0 : i64, kernel_shape = [1, 1],
                     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
                     group = 1 : i64, packed_int4}
    return
  }

// ===== 4: unpacked 8-bit weights report their real width =====
// Without packed_int4 the same i8 buffer is plain INT8, so weight_bits restates
// the storage width as 8 rather than 4.

// CHECK-LABEL: llvm.func @qconv_unpacked_weights
// CHECK-DAG:   llvm.mlir.constant(8 : i64)
// CHECK:       llvm.call @wrap_qconv
  func.func @qconv_unpacked_weights(%ctx: !hip.context,
                                    %in: memref<1x2048x1x128xui16, 1>,
                                    %w: memref<1024x2048x1x1xi8, 1>,
                                    %wscale: memref<1024xf32, 1>,
                                    %wzp: memref<1024xi8, 1>,
                                    %out: memref<1x1024x1x128xui16, 1>) {
    hip.qconv(%ctx) ins(%in, %w, %wscale, %wzp :
                        memref<1x2048x1x128xui16, 1>, memref<1024x2048x1x1xi8, 1>,
                        memref<1024xf32, 1>, memref<1024xi8, 1>)
                    outs(%out : memref<1x1024x1x128xui16, 1>)
                    {input_scale = 1.638800e-04 : f32, input_zp = 35275 : i64,
                     output_scale = 3.687020e-04 : f32, output_zp = 36322 : i64,
                     weight_axis = 0 : i64, kernel_shape = [1, 1],
                     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
                     group = 1 : i64}
    return
  }

}
