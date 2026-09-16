//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ExpandOp
//===----------------------------------------------------------------------===//

def Onnx_ExpandOp : Onnx_Op<"Expand", [Pure]> {
  let summary = "ONNX Expand, schema version 13";

  let description = [{
    Broadcast the input tensor following the given shape and the broadcast rule.
    The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
    Dimensions are right alignment;
    Two corresponding dimensions must have the same value, or one of them is equal to 1.
    Also, this operator is similar to numpy.broadcast_to(input, shape),
    but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
    It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
    or the shape.ndim < input.shape.ndim.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    TensorOf<[I64]>:$shape
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output
  );
}
