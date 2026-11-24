// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: hip-compiler --mode HIP_IR.
//
// The compile writes the module once the ONNX source has become tensor-form
// HIP IR, naming "hip" as its source pipeline; that file compiles on its own,
// from bufferization on, to the same artifact as the source, and as HIP IR to
// itself. See lib/Compiler/CompilerDriver.cpp (writeHipIr) and
// docs/design/compilation-options.md, "HIP IR output".
//===----------------------------------------------------------------------===//

// RUN: hip-compiler %s --mode HIP_IR -o %t.hip.mlir
// RUN: FileCheck --input-file=%t.hip.mlir %s
// RUN: hip-compiler %t.hip.mlir --mode HIP_IR -o %t.again.hip.mlir
// RUN: diff %t.hip.mlir %t.again.hip.mlir
// RUN: hip-compiler %s -o %t.bc
// RUN: hip-compiler %t.hip.mlir -o %t.from-hip.bc
// RUN: cmp %t.bc %t.from-hip.bc

// CHECK: module attributes {
// CHECK-SAME: hip.compiler.source_pipeline = "hip"
// CHECK-NOT: onnx.
// CHECK: func.func @main_graph(%{{.*}}: !hip.context
// CHECK: hip.

module {
  func.func @main_graph(%arg0: tensor<1x3x16xf32>, %arg1: tensor<1x3x16xf32>) -> tensor<1x3x16xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "/Add"} : (tensor<1x3x16xf32>, tensor<1x3x16xf32>) -> tensor<1x3x16xf32>
    "onnx.Return"(%0) {onnx_node_name = "/Return"} : (tensor<1x3x16xf32>) -> ()
  }
}
