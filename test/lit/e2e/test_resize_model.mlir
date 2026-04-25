// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_resize
// CHECK-NOT: onnx.Resize

module {
  func.func @main_graph(%arg0: tensor<1x32x1x100xf16>) -> tensor<1x32x1x200xf16> {
    %roi    = "onnx.Constant"() {value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
    %scales = "onnx.Constant"() {value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
    %sizes  = "onnx.Constant"() {value = dense<[1, 32, 1, 200]> : tensor<4xi64>} : () -> tensor<4xi64>
    %0 = "onnx.Resize"(%arg0, %roi, %scales, %sizes) {
        mode = "nearest", coordinate_transformation_mode = "half_pixel"
    } : (tensor<1x32x1x100xf16>, tensor<0xf32>, tensor<0xf32>, tensor<4xi64>) -> tensor<1x32x1x200xf16>
    "onnx.Return"(%0) : (tensor<1x32x1x200xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
