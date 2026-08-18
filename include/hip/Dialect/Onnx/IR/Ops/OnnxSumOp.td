//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SumOp
//===----------------------------------------------------------------------===//

def Onnx_SumOp : Onnx_Op<"Sum", [Pure]> {
  let summary = "ONNX Sum, schema version 13";

  let description = [{
    Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
    All inputs and outputs must have the same data type.
    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>>:$data_0
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$sum
  );
}
