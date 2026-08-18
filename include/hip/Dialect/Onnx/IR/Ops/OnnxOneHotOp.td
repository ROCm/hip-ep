//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_OneHotOp
//===----------------------------------------------------------------------===//

def Onnx_OneHotOp : Onnx_Op<"OneHot", [Pure]> {
  let summary = "ONNX OneHot, schema version 11";

  let description = [{
        Produces a one-hot tensor based on inputs.
        The locations represented by the index values in the 'indices' input tensor will have 'on_value'
        and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
        are specified as part of required input argument 'values', which is a two-element tensor of format
        [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
        input tensor. The additional dimension is for one-hot representation. The additional dimension will
        be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
        dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
        dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
        as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
        the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
        output tensor.

        when axis = 0:
        output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

        when axis = -1:
        output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$indices,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$depth,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$values,
    DefaultValuedAttr<SI64Attr, "-1">:$axis
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output
  );
}
