module {
  func.func @main_graph(%lhs: tensor<1x4x4xf32>, %rhs: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
    %output = "onnx.Add"(%lhs, %rhs) : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    return %output : tensor<1x4x4xf32>
  }
}
