//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ArgMaxOp
//===----------------------------------------------------------------------===//

def Onnx_ArgMaxOp : Onnx_Op<"ArgMax", [Pure]> {
  let summary = "ONNX ArgMax, schema version 13";

  let description = [{
    Computes the indices of the max elements of the input tensor's element along the
    provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
    If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
    If select_last_index is True (default False), the index of the last occurrence of the max
    is selected if the max appears more than once in the input. Otherwise the index of the
    first occurrence is selected.
    The type of the output tensor is integer.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$data,
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    DefaultValuedAttr<SI64Attr, "1">:$keepdims,
    DefaultValuedAttr<SI64Attr, "0">:$select_last_index
  );

  let results = (outs
    TensorOf<[I64]>:$reduced
  );
}
