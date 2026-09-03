//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MultinomialOp
//===----------------------------------------------------------------------===//

def Onnx_MultinomialOp : Onnx_Op<"Multinomial", [Pure]> {
  let summary = "ONNX Multinomial, schema version 22";

  let description = [{
    Generate a tensor of samples from a multinomial distribution according to the probabilities
    of each of the possible outcomes.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input,
    DefaultValuedAttr<SI64Attr, "6">:$dtype,
    DefaultValuedAttr<SI64Attr, "1">:$sample_size,
    OptionalAttr<F32Attr>:$seed
  );

  let results = (outs
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$output
  );
}
