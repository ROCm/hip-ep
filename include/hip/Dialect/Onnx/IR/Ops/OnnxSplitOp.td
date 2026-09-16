//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SplitOp
//===----------------------------------------------------------------------===//

def Onnx_SplitOp : Onnx_Op<"Split", [Pure]> {
  let summary = "ONNX Split, schema version 18";

  let description = [{
    Split a tensor into a list of tensors, along the specified 'axis'.
    Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
    If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
    If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
    If the input 'split' is specified, it indicates the sizes of each output in the split.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$split,
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    OptionalAttr<SI64Attr>:$num_outputs
  );

  let results = (outs
    Variadic<AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>>:$outputs
  );
}
