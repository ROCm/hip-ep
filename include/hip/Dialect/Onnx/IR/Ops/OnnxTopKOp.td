//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_TopKOp
//===----------------------------------------------------------------------===//

def Onnx_TopKOp : Onnx_Op<"TopK", [Pure]> {
  let summary = "ONNX TopK, schema version 24";

  let description = [{
    Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
    shape [a_0, a_1, ..., a_{n-1} ] and integer argument k, return two outputs:

    * Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1} ]
      which contains the values of the top k elements along the specified axis
    * Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1} ] which
      contains the indices of the top k elements (original indices from the input
      tensor).

    * If "largest" is 1 (the default value) then the k largest elements are returned.
    * If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
    * If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

    Given two equivalent values, this operator uses the indices along the axis as
    a tiebreaker. That is, the element with the lower index will appear first.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$K,
    DefaultValuedAttr<SI64Attr, "-1">:$axis,
    DefaultValuedAttr<SI64Attr, "1">:$largest,
    DefaultValuedAttr<SI64Attr, "1">:$sorted
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Values,
    TensorOf<[I64]>:$Indices
  );
}
