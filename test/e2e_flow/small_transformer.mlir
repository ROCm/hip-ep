module {
  func.func @main_graph(%arg0: tensor<128xf16>, %arg1: tensor<128x128xf16>, %arg2: tensor<128x128xf16>, %arg3: tensor<128x128xf16>, %arg4: tensor<1x4x128xf16>) -> tensor<1x4x128xf16> {
    %_k1_i128 = "torch.constant.int"() {value = 128 : i64} : () -> !torch.int
    %_list_2112866542096 = "torch.prim.ListConstruct"(%_k1_i128) : (!torch.int) -> !torch.list<int>
    %_k2_f = "torch.constant.float"() {value = 1.000000e-06 : f64} : () -> !torch.float
    %rms_norm = "torch.aten.rms_norm"(%arg4, %_list_2112866542096, %arg0, %_k2_f) : (tensor<1x4x128xf16>, !torch.list<int>, tensor<128xf16>, !torch.float) -> tensor<1x4x128xf16>
    %_none = "torch.constant.none"() : () -> !torch.none
    %linear = "torch.aten.linear"(%rms_norm, %arg1, %_none) : (tensor<1x4x128xf16>, tensor<128x128xf16>, !torch.none) -> tensor<1x4x128xf16>
    %silu = "torch.aten.silu"(%linear) : (tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
    %linear_1 = "torch.aten.linear"(%rms_norm, %arg2, %_none) : (tensor<1x4x128xf16>, tensor<128x128xf16>, !torch.none) -> tensor<1x4x128xf16>
    %mul = "torch.aten.mul.Tensor"(%silu, %linear_1) : (tensor<1x4x128xf16>, tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
    %linear_2 = "torch.aten.linear"(%mul, %arg3, %_none) : (tensor<1x4x128xf16>, tensor<128x128xf16>, !torch.none) -> tensor<1x4x128xf16>
    %_k3_i1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %add = "torch.aten.add.Tensor"(%arg4, %linear_2, %_k3_i1) : (tensor<1x4x128xf16>, tensor<1x4x128xf16>, !torch.int) -> tensor<1x4x128xf16>
    return %add : tensor<1x4x128xf16>
  }
}
