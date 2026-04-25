// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_concat
// CHECK-NOT: onnx.Concat

module {
  func.func @main_graph(%a: tensor<2x4xf16>, %b: tensor<2x6xf16>)
      -> (tensor<2x10xf16>) {
    %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
        : (tensor<2x4xf16>, tensor<2x6xf16>) -> tensor<2x10xf16>
    "onnx.Return"(%0) : (tensor<2x10xf16>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
