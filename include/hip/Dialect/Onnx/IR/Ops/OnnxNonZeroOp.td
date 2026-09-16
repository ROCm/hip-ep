//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_NonZeroOp
//===----------------------------------------------------------------------===//

def Onnx_NonZeroOp : Onnx_Op<"NonZero", [Pure]> {
  let summary = "ONNX NonZero, schema version 13";

  let description = [{
        Returns the indices of the elements that are non-zero
        (in row-major order - by dimension).
        NonZero behaves similar to numpy.nonzero:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
        but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$X
  );

  let results = (outs
    TensorOf<[I64]>:$Y
  );
}
