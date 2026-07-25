// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_power
// CHECK-NOT: onnx.Sqrt

module {
  func.func @main_graph(%arg0: tensor<1x128x512xf16>) -> (tensor<1x128x512xf16>) {
    %0 = "onnx.Sqrt"(%arg0) : (tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
    "onnx.Return"(%0) : (tensor<1x128x512xf16>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
