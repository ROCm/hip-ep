// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @fast_gelu_no_bias_f16(
      %ctx: !hip.context,
      %input: memref<1x128x768xf16, 1>,
      %output: memref<1x128x768xf16, 1>) {
    // CHECK-LABEL: llvm.func @fast_gelu_no_bias_f16
    hip.fast_gelu(%ctx) ins(%input : memref<1x128x768xf16, 1>)
                        outs(%output : memref<1x128x768xf16, 1>)
    // CHECK: llvm.call @wrap_fast_gelu
    return
  }

  func.func @fast_gelu_with_bias_f16(
      %ctx: !hip.context,
      %input: memref<1x128x768xf16, 1>,
      %bias: memref<768xf16, 1>,
      %output: memref<1x128x768xf16, 1>) {
    // CHECK-LABEL: llvm.func @fast_gelu_with_bias_f16
    hip.fast_gelu(%ctx) ins(%input : memref<1x128x768xf16, 1>)
                        bias(%bias : memref<768xf16, 1>)
                        outs(%output : memref<1x128x768xf16, 1>)
    // CHECK: llvm.call @wrap_fast_gelu
    return
  }

  func.func @fast_gelu_dynamic_2d_f16(
      %ctx: !hip.context,
      %input: memref<?x?xf16, 1>,
      %bias: memref<?xf16, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @fast_gelu_dynamic_2d_f16
    hip.fast_gelu(%ctx) ins(%input : memref<?x?xf16, 1>)
                        bias(%bias : memref<?xf16, 1>)
                        outs(%output : memref<?x?xf16, 1>)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.call @wrap_fast_gelu
    return
  }
}
