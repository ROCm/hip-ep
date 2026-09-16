//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_OptionalHasElementOp
//===----------------------------------------------------------------------===//

def Onnx_OptionalHasElementOp : Onnx_Op<"OptionalHasElement", [Pure]> {
  let summary = "ONNX OptionalHasElement, schema version 18";

  let description = [{
    Returns true if (1) the input is an optional-type and contains an element,
    or, (2) the input is a tensor or sequence type.
    If the input is not provided or is an empty optional-type, this op returns false.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType
    ]>:$input
  );

  let results = (outs
    TensorOf<[I1]>:$output
  );
}
