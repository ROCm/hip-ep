//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_STFTOp
//===----------------------------------------------------------------------===//

def Onnx_STFTOp : Onnx_Op<"STFT", [Pure]> {
  let summary = "ONNX STFT, schema version 17";

  let description = [{
    Computes the Short-time Fourier Transform of the signal.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$signal,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$frame_step,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>,
      NoneType
    ]>:$window,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$frame_length,
    DefaultValuedAttr<SI64Attr, "1">:$onesided
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$output
  );
}
