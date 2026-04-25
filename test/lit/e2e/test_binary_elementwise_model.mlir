// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_elementwise_binary
// CHECK-NOT: onnx.Pow

// We use Pow (instead of Div) here so the test runner's default zero-filled
// inputs don't produce NaN in the output and trip the validator.  The Div
// kernel itself is exercised by the conversion-pass LIT tests.
module {
  func.func @main_graph(%a: tensor<1x128x512xf16>, %b: tensor<1x128x512xf16>)
      -> (tensor<1x128x512xf16>) {
    %0 = "onnx.Pow"(%a, %b)
        : (tensor<1x128x512xf16>, tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
    "onnx.Return"(%0) : (tensor<1x128x512xf16>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
