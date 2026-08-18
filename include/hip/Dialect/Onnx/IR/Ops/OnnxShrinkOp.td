//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ShrinkOp
//===----------------------------------------------------------------------===//

def Onnx_ShrinkOp : Onnx_Op<"Shrink", [Pure]> {
  let summary = "ONNX Shrink, schema version 9";

  let description = [{
    Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
    having same datatype and shape with input. It has two attributes, lambd and
    bias. The formula of this operator is: If x < -lambd, y = x + bias;
    If x > lambd, y = x - bias; Otherwise, y = 0.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input,
    DefaultValuedAttr<F32Attr, "0.0">:$bias,
    DefaultValuedAttr<F32Attr, "0.5">:$lambd
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
