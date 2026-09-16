//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ConstantOfShapeOp
//===----------------------------------------------------------------------===//

def Onnx_ConstantOfShapeOp : Onnx_Op<"ConstantOfShape", [Pure]> {
  let summary = "ONNX ConstantOfShape, schema version 25";

  let description = [{
    Generate a tensor with given value and shape.
  }];

  let arguments = (ins
    TensorOf<[I64]>:$input,
    OptionalAttr<AnyAttr>:$value
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>,
      TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>,
      TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>:$output
  );
}
