//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_DropoutOp
//===----------------------------------------------------------------------===//

def Onnx_DropoutOp : Onnx_Op<"Dropout", [Pure]> {
  let summary = "ONNX Dropout, schema version 22";

  let description = [{
    Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
    output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
    Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
    the user can simply not pass `training_mode` input or set it to false.
    ```
    output = scale * data * mask,
    ```
    where
    ```
    scale = 1. / (1. - ratio).
    ```
    This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>
    ]>:$data,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, NoneType
    ]>:$ratio,
    AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$training_mode,
    OptionalAttr<SI64Attr>:$seed
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>
    ]>:$output,
    AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$mask
  );
}
