//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_CompressOp
//===----------------------------------------------------------------------===//

def Onnx_CompressOp : Onnx_Op<"Compress", [Pure]> {
  let summary = "ONNX Compress, schema version 11";

  let description = [{
        Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
        In case axis is not provided, input is flattened before elements are selected.
        Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html

  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    TensorOf<[I1]>:$condition,
    OptionalAttr<SI64Attr>:$axis
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
