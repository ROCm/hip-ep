module {
  func.func @main_graph(%arg0: tensor<4x8xf16>, %arg1: tensor<1x2x4xf16>) -> tensor<1x2x8xf16> {
    %_none = "torch.constant.none"() : () -> !torch.none
    %linear = "torch.aten.linear"(%arg1, %arg0, %_none) : (tensor<1x2x4xf16>, tensor<4x8xf16>, !torch.none) -> tensor<1x2x8xf16>
    return %linear : tensor<1x2x8xf16>
  }
}
