//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ReduceMaxOp
//===----------------------------------------------------------------------===//

def Onnx_ReduceMaxOp : Onnx_Op<"ReduceMax", [Pure]> {
  let summary = "ONNX ReduceMax, schema version 20";

  let description = [{
    Computes the max of the input tensor's elements along the provided axes. The resulting
    tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


    If the input data type is Boolean, the comparison should consider `False < True`.

    The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    to `False` instead of `True`.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>,
      TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>
    ]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    DefaultValuedAttr<SI64Attr, "1">:$keepdims,
    DefaultValuedAttr<SI64Attr, "0">:$noop_with_empty_axes
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>,
      TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>
    ]>:$reduced
  );
}
