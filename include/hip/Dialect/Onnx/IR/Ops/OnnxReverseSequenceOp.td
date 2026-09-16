//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ReverseSequenceOp
//===----------------------------------------------------------------------===//

def Onnx_ReverseSequenceOp : Onnx_Op<"ReverseSequence", [Pure]> {
  let summary = "ONNX ReverseSequence, schema version 10";

  let description = [{
    Reverse batch of sequences having different lengths specified by `sequence_lens`.

    For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
    and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
    sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

    Example 1:
      input = [[0.0, 4.0, 8.0,  12.0],
               [1.0, 5.0, 9.0,  13.0],
               [2.0, 6.0, 10.0, 14.0],
               [3.0, 7.0, 11.0, 15.0]]
      sequence_lens = [4, 3, 2, 1]
      time_axis = 0
      batch_axis = 1

      output = [[3.0, 6.0, 9.0,  12.0],
                [2.0, 5.0, 8.0,  13.0],
                [1.0, 4.0, 10.0, 14.0],
                [0.0, 7.0, 11.0, 15.0]]

    Example 2:
      input = [[0.0,  1.0,  2.0,  3.0 ],
               [4.0,  5.0,  6.0,  7.0 ],
               [8.0,  9.0,  10.0, 11.0],
               [12.0, 13.0, 14.0, 15.0]]
      sequence_lens = [1, 2, 3, 4]
      time_axis = 1
      batch_axis = 0

      output = [[0.0,  1.0,  2.0,  3.0 ],
                [5.0,  4.0,  6.0,  7.0 ],
                [10.0, 9.0,  8.0,  11.0],
                [15.0, 14.0, 13.0, 12.0]]
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    TensorOf<[I64]>:$sequence_lens,
    DefaultValuedAttr<SI64Attr, "1">:$batch_axis,
    DefaultValuedAttr<SI64Attr, "0">:$time_axis
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$Y
  );
}
