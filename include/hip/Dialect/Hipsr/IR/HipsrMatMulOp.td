//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.matmul op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_MatMulOp
//===----------------------------------------------------------------------===//

def Hipsr_MatMulOp : Hipsr_DpsOp<"matmul"> {
  let summary = "Matrix multiply (batched N-D x N-D), destination-passing style";
  let description = [{
    Matrix multiplication with ONNX/NumPy `matmul` semantics. For operands that
    are both >= 2-D: `result[..., M, N] = A[..., M, K] @ B[..., K, N]`, where the
    leading (batch) dims broadcast NumPy-style. A 1-D operand is promoted for the
    multiply and stripped from the result: A `(K)` acts as `(1,K)` (drops the
    leading 1), B `(K)` acts as `(K,1)` (drops the trailing 1); `(K) @ (K)` is a
    scalar. Destination-passing style: the result buffer is passed as the `init`
    operand.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$A,
    Hipsr_TensorOrDeviceMemRef:$B,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // Rejects rank-0 (scalar) A/B: matmul needs at least a contraction dim.
  let hasVerifier = 1;
}
