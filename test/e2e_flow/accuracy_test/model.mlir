module {
  func.func @main_graph(%arg0: tensor<32xf16>, %arg1: tensor<32x32xf16>, %arg2: tensor<32x32xf16>, %arg3: tensor<32x32xf16>, %arg4: tensor<1x4x32xf16>, %arg5: tensor<32x32xf16>, %arg6: tensor<32x32xf16>, %arg7: tensor<32x32xf16>) -> tensor<1x4x32xf16> {
    %_to_copy = "torch.aten._to_copy"(%arg4) : (tensor<1x4x32xf16>) -> tensor<1x4x32xf32>
    %_k1_i2 = "torch.constant.int"() {value = 2 : i64} : () -> !torch.int
    %pow_1 = "torch.aten.pow.Tensor_Scalar"(%_to_copy, %_k1_i2) : (tensor<1x4x32xf32>, !torch.int) -> tensor<1x4x32xf32>
    %_list_3029808431184 = "torch.prim.ListConstruct"(%_k1_i2) : (!torch.int) -> !torch.list<int>
    %_k2_i1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %mean = "torch.aten.mean.dim"(%pow_1, %_list_3029808431184, %_k2_i1) : (tensor<1x4x32xf32>, !torch.list<int>, !torch.int) -> tensor<1x4x1xf32>
    %_k3_f = "torch.constant.float"() {value = 1.000000e-06 : f64} : () -> !torch.float
    %add = "torch.aten.add.Scalar"(%mean, %_k3_f) : (tensor<1x4x1xf32>, !torch.float) -> tensor<1x4x1xf32>
    %rsqrt = "torch.aten.rsqrt"(%add) : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>
    %mul = "torch.aten.mul.Tensor"(%_to_copy, %rsqrt) : (tensor<1x4x32xf32>, tensor<1x4x1xf32>) -> tensor<1x4x32xf32>
    %mul_1 = "torch.aten.mul.Tensor"(%mul, %arg0) : (tensor<1x4x32xf32>, tensor<32xf16>) -> tensor<1x4x32xf32>
    %_to_copy_1 = "torch.aten._to_copy"(%mul_1) : (tensor<1x4x32xf32>) -> tensor<1x4x32xf16>
    %_k4_i4 = "torch.constant.int"() {value = 4 : i64} : () -> !torch.int
    %_k5_i32 = "torch.constant.int"() {value = 32 : i64} : () -> !torch.int
    %_list_3029811762496 = "torch.prim.ListConstruct"(%_k4_i4, %_k5_i32) : (!torch.int, !torch.int) -> !torch.list<int>
    %view = "torch.aten.view"(%_to_copy_1, %_list_3029811762496) : (tensor<1x4x32xf16>, !torch.list<int>) -> tensor<4x32xf16>
    %mm = "torch.aten.mm"(%view, %arg5) : (tensor<4x32xf16>, tensor<32x32xf16>) -> tensor<4x32xf16>
    %_k6_i1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %_list_3029808435424 = "torch.prim.ListConstruct"(%_k6_i1, %_k4_i4, %_k5_i32) : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %view_1 = "torch.aten.view"(%mm, %_list_3029808435424) : (tensor<4x32xf16>, !torch.list<int>) -> tensor<1x4x32xf16>
    %_to_copy_2 = "torch.aten._to_copy"(%view_1) : (tensor<1x4x32xf16>) -> tensor<1x4x32xf32>
    %sigmoid = "torch.aten.sigmoid"(%_to_copy_2) : (tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %mul_2 = "torch.aten.mul.Tensor"(%_to_copy_2, %sigmoid) : (tensor<1x4x32xf32>, tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %_to_copy_3 = "torch.aten._to_copy"(%mul_2) : (tensor<1x4x32xf32>) -> tensor<1x4x32xf16>
    %_list_3029811760656 = "torch.prim.ListConstruct"(%_k4_i4, %_k5_i32) : (!torch.int, !torch.int) -> !torch.list<int>
    %view_2 = "torch.aten.view"(%_to_copy_1, %_list_3029811760656) : (tensor<1x4x32xf16>, !torch.list<int>) -> tensor<4x32xf16>
    %mm_1 = "torch.aten.mm"(%view_2, %arg6) : (tensor<4x32xf16>, tensor<32x32xf16>) -> tensor<4x32xf16>
    %_list_3029811766016 = "torch.prim.ListConstruct"(%_k6_i1, %_k4_i4, %_k5_i32) : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %view_3 = "torch.aten.view"(%mm_1, %_list_3029811766016) : (tensor<4x32xf16>, !torch.list<int>) -> tensor<1x4x32xf16>
    %mul_3 = "torch.aten.mul.Tensor"(%_to_copy_3, %view_3) : (tensor<1x4x32xf16>, tensor<1x4x32xf16>) -> tensor<1x4x32xf16>
    %_list_3029811767136 = "torch.prim.ListConstruct"(%_k4_i4, %_k5_i32) : (!torch.int, !torch.int) -> !torch.list<int>
    %view_4 = "torch.aten.view"(%mul_3, %_list_3029811767136) : (tensor<1x4x32xf16>, !torch.list<int>) -> tensor<4x32xf16>
    %mm_2 = "torch.aten.mm"(%view_4, %arg7) : (tensor<4x32xf16>, tensor<32x32xf16>) -> tensor<4x32xf16>
    %_list_3029808292768 = "torch.prim.ListConstruct"(%_k6_i1, %_k4_i4, %_k5_i32) : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %view_5 = "torch.aten.view"(%mm_2, %_list_3029808292768) : (tensor<4x32xf16>, !torch.list<int>) -> tensor<1x4x32xf16>
    %add_1 = "torch.aten.add.Tensor"(%arg4, %view_5, %_k6_i1) : (tensor<1x4x32xf16>, tensor<1x4x32xf16>, !torch.int) -> tensor<1x4x32xf16>
    return %add_1 : tensor<1x4x32xf16>
  }
}
