// ONNX-MLIR model with a single Conv operation
// This is what onnx-mlir produces from an ONNX model
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    // Weights and bias as constants (embedded in model)
    %0 = "onnx.Constant"() {value = dense<1.0> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<0.5> : tensor<64xf32>} : () -> tensor<64xf32>

    // Conv operation
    %2 = "onnx.Conv"(%arg0, %0, %1) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>

    return %2 : tensor<1x64x112x112xf32>
  }
}
