//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_CenterCropPadOp
//===----------------------------------------------------------------------===//

def Onnx_CenterCropPadOp : Onnx_Op<"CenterCropPad", [Pure]> {
  let summary = "ONNX CenterCropPad, schema version 18";

  let description = [{
    Center crop or pad an input to given dimensions.

    The crop/pad dimensions can be specified for a subset of the `axes`; unspecified dimensions will remain unchanged.

    If the input dimensions are larger than the target crop dimensions, a centered cropping window will be extracted
    from the input. The starting value for the cropping window is rounded down, which means that if the difference
    between the input shape and the crop shape is odd, the cropping window will be shifted half a pixel to the left
    of the input center.

    If the input dimensions are smaller than the target crop dimensions, the input will be padded equally on both sides
    to center it in the output. In cases where the total number of padding pixels is odd, an additional pixel will be
    added to the right side.

    The padding value used is zero.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input_data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$shape,
    OptionalAttr<I64ArrayAttr>:$axes
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output_data
  );
}
