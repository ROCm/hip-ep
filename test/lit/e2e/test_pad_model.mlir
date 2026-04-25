// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_pad
// CHECK-NOT: onnx.Pad

module {
  func.func @main_graph(%arg0: tensor<1x1x16xf16>) -> tensor<1x1x18xf16> {
    %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 0, 0, 1]> : tensor<6xi64>} : () -> tensor<6xi64>
    %0 = "onnx.Pad"(%arg0, %pads) {mode = "reflect"}
        : (tensor<1x1x16xf16>, tensor<6xi64>) -> tensor<1x1x18xf16>
    "onnx.Return"(%0) : (tensor<1x1x18xf16>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
