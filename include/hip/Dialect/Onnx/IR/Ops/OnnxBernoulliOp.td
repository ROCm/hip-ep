//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_BernoulliOp
//===----------------------------------------------------------------------===//

def Onnx_BernoulliOp : Onnx_Op<"Bernoulli", [Pure]> {
  let summary = "ONNX Bernoulli, schema version 22";

  let description = [{
    Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
    containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
    where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

    This operator is non-deterministic and may not produce the same values in different
    implementations (even if a seed is specified).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<F32Attr>:$seed
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>
    ]>:$output
  );
}
