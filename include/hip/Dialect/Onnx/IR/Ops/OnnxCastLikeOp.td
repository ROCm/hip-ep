//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_CastLikeOp
//===----------------------------------------------------------------------===//

def Onnx_CastLikeOp : Onnx_Op<"CastLike", [Pure]> {
  let summary = "ONNX CastLike, schema version 25";

  let description = [{
    The operator casts the elements of a given input tensor (the first input) to
    the same data type as the elements of the second input tensor.
    See documentation of the Cast operator for further details.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>,
      TensorOf<[UI<2>]>, TensorOf<[I<2>]>
    ]>:$input,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>,
      TensorOf<[UI<2>]>, TensorOf<[I<2>]>
    ]>:$target_type,
    DefaultValuedStrAttr<StrAttr, "up">:$round_mode,
    DefaultValuedAttr<SI64Attr, "1">:$saturate
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>,
      TensorOf<[UI<2>]>, TensorOf<[I<2>]>
    ]>:$output
  );
}
