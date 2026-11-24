// A byte cast converts by value, so i64 5 or f32 2.5 would be stored in the
// mask as 5 or 2 instead of 1. Only a 0/1 byte source may target i1.

// RUN: not hip-mlir-opt %s --convert-hip-to-llvm 2>&1 | FileCheck %s

// CHECK: hip.cast to i1 needs a != 0 normalisation kernel
module {
  func.func @cast_i64_to_i1(
      %ctx: !hip.context,
      %input: memref<8xi64, 1>,
      %output: memref<8xi1, 1>) {
    hip.cast(%ctx) ins(%input : memref<8xi64, 1>)
                   outs(%output : memref<8xi1, 1>) {to = 9 : i64}
    return
  }
}
