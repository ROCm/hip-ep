//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_TriluOp
//===----------------------------------------------------------------------===//

def Onnx_TriluOp : Onnx_Op<"Trilu", [Pure]> {
  let summary = "ONNX Trilu, schema version 14";

  let description = [{
    Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
    The attribute "upper" determines whether the upper or lower part is retained. If set to true,
    the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
    Default value for the "upper" attribute is true.
    Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
    of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
    All other elements in the matrix are set to zero.
    If k = 0, the triangular part on and above/below the main diagonal is retained.
    If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
    A negative k value retains the main diagonal and |k| diagonals below it.
    If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
    A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$k,
    DefaultValuedAttr<SI64Attr, "1">:$upper
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
