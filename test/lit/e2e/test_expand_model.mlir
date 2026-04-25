// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_expand
// CHECK-NOT: onnx.Expand

// Trailing-axis broadcast: tensor<1x256x1xf32> --> tensor<1x256x128xf32>.
// This is the canonical Kokoro-style Expand: a singleton last dim is
// broadcast out to a real extent (here mimicking expand-then-multiply
// patterns in the duration aligner).

module {
  func.func @main_graph(%arg0: tensor<1x256x1xf32>) -> tensor<1x256x128xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 256, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
    %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x256x1xf32>, tensor<3xi64>) -> tensor<1x256x128xf32>
    "onnx.Return"(%0) : (tensor<1x256x128xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
