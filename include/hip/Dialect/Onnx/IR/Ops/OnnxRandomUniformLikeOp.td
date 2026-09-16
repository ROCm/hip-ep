//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_RandomUniformLikeOp
//===----------------------------------------------------------------------===//

def Onnx_RandomUniformLikeOp : Onnx_Op<"RandomUniformLike", [Pure]> {
  let summary = "ONNX RandomUniformLike, schema version 22";

  let description = [{
    Generate a tensor with random values drawn from a uniform distribution.
    The shape of the output tensor is copied from the shape of the input tensor,
    and the parameters of the uniform distribution are specified by `low` and `high`.

    The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
    The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
    TensorProto message and be valid as an output type.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    DefaultValuedAttr<F32Attr, "1.0">:$high,
    DefaultValuedAttr<F32Attr, "0.0">:$low,
    OptionalAttr<F32Attr>:$seed
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
