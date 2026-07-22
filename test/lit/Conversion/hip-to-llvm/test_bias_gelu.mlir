// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @bias_gelu_static_f16(
      %ctx: !hip.context,
      %data: memref<1x128x768xf16, 1>,
      %bias: memref<768xf16, 1>,
      %output: memref<1x128x768xf16, 1>) {
    // CHECK-LABEL: llvm.func @bias_gelu_static_f16
    hip.bias_gelu(%ctx) ins(%data, %bias : memref<1x128x768xf16, 1>, memref<768xf16, 1>)
                        outs(%output : memref<1x128x768xf16, 1>)
    // CHECK: llvm.call @wrap_bias_gelu
    return
  }

  func.func @bias_gelu_dynamic_2d_f16(
      %ctx: !hip.context,
      %data: memref<?x?xf16, 1>,
      %bias: memref<?xf16, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @bias_gelu_dynamic_2d_f16
    hip.bias_gelu(%ctx) ins(%data, %bias : memref<?x?xf16, 1>, memref<?xf16, 1>)
                        outs(%output : memref<?x?xf16, 1>)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_bias_gelu
    return
  }
}
