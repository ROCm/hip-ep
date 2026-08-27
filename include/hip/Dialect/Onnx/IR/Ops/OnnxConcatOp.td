//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ConcatOp
//===----------------------------------------------------------------------===//

def Onnx_ConcatOp : Onnx_Op<"Concat", [Pure]> {
  let summary = "ONNX Concat, schema version 13";

  let description = [{
    Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>>:$inputs,
    SI64Attr:$axis
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$concat_result
  );
}
