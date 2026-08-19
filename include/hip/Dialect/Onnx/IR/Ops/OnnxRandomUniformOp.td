//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_RandomUniformOp
//===----------------------------------------------------------------------===//

def Onnx_RandomUniformOp : Onnx_Op<"RandomUniform", [Pure]> {
  let summary = "ONNX RandomUniform, schema version 22";

  let description = [{
    Generate a tensor with random values drawn from a uniform distribution. The shape
    of the tensor is specified by the `shape` argument and the range by `low` and `high`.

    The data type is specified by the 'dtype' argument. The 'dtype' argument must
    be one of the data types specified in the 'DataType' enum field in the
    TensorProto message.
  }];

  let arguments = (ins
    DefaultValuedAttr<SI64Attr, "1">:$dtype,
    DefaultValuedAttr<F32Attr, "1.0">:$high,
    DefaultValuedAttr<F32Attr, "0.0">:$low,
    OptionalAttr<F32Attr>:$seed,
    I64ArrayAttr:$shape
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
