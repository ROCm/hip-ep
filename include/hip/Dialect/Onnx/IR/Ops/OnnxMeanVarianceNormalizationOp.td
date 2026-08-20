//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MeanVarianceNormalizationOp
//===----------------------------------------------------------------------===//

def Onnx_MeanVarianceNormalizationOp : Onnx_Op<"MeanVarianceNormalization", [Pure]> {
  let summary = "ONNX MeanVarianceNormalization, schema version 13";

  let description = [{
          A MeanVarianceNormalization Function: Perform mean variance normalization
          on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    DefaultValuedAttr<I64ArrayAttr, "{0, 2, 3}">:$axes
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
