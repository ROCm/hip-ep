//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GemmOp
//===----------------------------------------------------------------------===//

def Onnx_GemmOp : Onnx_Op<"Gemm", [Pure]> {
  let summary = "ONNX Gemm, schema version 13";

  let description = [{
    General Matrix multiplication:
    https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

    * A' = transpose(A) if transA else A
    * B' = transpose(B) if transB else B

    Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
    input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
    and output tensor Y has shape (M, N). A will be transposed before doing the
    computation if attribute transA is non-zero, same for B and transB.
    This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
    This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>
    ]>:$A,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>
    ]>:$B,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>,
      NoneType
    ]>:$C,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$beta,
    DefaultValuedAttr<SI64Attr, "0">:$transA,
    DefaultValuedAttr<SI64Attr, "0">:$transB
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
