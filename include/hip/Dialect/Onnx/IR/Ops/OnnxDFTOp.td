//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_DFTOp
//===----------------------------------------------------------------------===//

def Onnx_DFTOp : Onnx_Op<"DFT", [Pure]> {
  let summary = "ONNX DFT, schema version 20";

  let description = [{
    Computes the discrete Fourier Transform (DFT) of the input.

    Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
    DFT is computed and `M` denotes the conceptual "all other dimensions,"
    the DFT `y[m, k]` of shape `[M, N]` is defined as

    $$y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,$$

    and the inverse transform is defined as

    $$x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,$$

    where $j$ is the imaginary unit.

    The actual shape of the output is specified in the "output" section.

    Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$dft_length,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$inverse,
    DefaultValuedAttr<SI64Attr, "0">:$onesided
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
