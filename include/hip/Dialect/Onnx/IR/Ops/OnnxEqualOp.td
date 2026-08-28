//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_EqualOp
//===----------------------------------------------------------------------===//

def Onnx_EqualOp : Onnx_Op<"Equal", [Pure]> {
  let summary = "ONNX Equal, schema version 19";

  let description = [{
    Returns the tensor resulted from performing the `equal` logical operation
    elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>,
      TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[BF16]>
    ]>:$A,
    AnyTypeOf<[
      TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>,
      TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[BF16]>
    ]>:$B
  );

  let results = (outs
    TensorOf<[I1]>:$C
  );
}
