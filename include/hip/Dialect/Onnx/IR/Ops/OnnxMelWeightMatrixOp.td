//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MelWeightMatrixOp
//===----------------------------------------------------------------------===//

def Onnx_MelWeightMatrixOp : Onnx_Op<"MelWeightMatrix", [Pure]> {
  let summary = "ONNX MelWeightMatrix, schema version 17";

  let description = [{
    Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
    This function defines the mel scale in terms of a frequency in hertz according to the following formula:

        mel(f) = 2595 * log10(1 + f/700)

    In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

    The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
  }];

  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$num_mel_bins,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$dft_length,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$sample_rate,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$lower_edge_hertz,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$upper_edge_hertz,
    DefaultValuedAttr<SI64Attr, "1">:$output_datatype
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$output
  );
}
