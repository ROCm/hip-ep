module {
  func.func @main_graph(%arg0: tensor<1024x3072xf16>, %arg1: tensor<1024x3072xf16>, %arg2: tensor<3072x1024xf16>, %arg3: tensor<1x2x1024xf16>) -> tensor<1x2x1024xf16> {
    %_none = "torch.constant.none"() : () -> !torch.none
    %linear = "torch.aten.linear"(%arg3, %arg0, %_none) : (tensor<1x2x1024xf16>, tensor<1024x3072xf16>, !torch.none) -> tensor<1x2x3072xf16>
    %silu = "torch.aten.silu"(%linear) : (tensor<1x2x3072xf16>) -> tensor<1x2x3072xf16>
    %linear_1 = "torch.aten.linear"(%arg3, %arg1, %_none) : (tensor<1x2x1024xf16>, tensor<1024x3072xf16>, !torch.none) -> tensor<1x2x3072xf16>
    %mul = "torch.aten.mul.Tensor"(%silu, %linear_1) : (tensor<1x2x3072xf16>, tensor<1x2x3072xf16>) -> tensor<1x2x3072xf16>
    %linear_2 = "torch.aten.linear"(%mul, %arg2, %_none) : (tensor<1x2x3072xf16>, tensor<3072x1024xf16>, !torch.none) -> tensor<1x2x1024xf16>
    return %linear_2 : tensor<1x2x1024xf16>
  }
}
