// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_constant_of_shape
// CHECK-NOT: onnx.ConstantOfShape

module {
  // ConstantOfShape requires a shape input (i64 1-D).  We pass it via the
  // function argument so the lowering pipeline sees a real input slot.
  func.func @main_graph(%shape: tensor<2xi64>) -> (tensor<3x4xf32>) {
    %0 = "onnx.ConstantOfShape"(%shape)
        {value = dense<1.5> : tensor<1xf32>}
        : (tensor<2xi64>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
