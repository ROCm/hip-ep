//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.matmul op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

#ifndef HIPSR_MATMUL_OP
#define HIPSR_MATMUL_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"

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

    Carries an optional shape region (region 0, see ShapeRegionInterface). The
    contraction dim K is not in the output but must match on both sides, so it
    is guarded by a `shape.cstr_eq` witness around a `shape.assuming` region
    that produces the dims.

    Example (populated region, 2-D):
    ```mlir
    %0 = hipsr.matmul(%ctx) ins(%Aop, %Bop : tensor<?x?xf16>, tensor<?x?xf16>)
                      outs(%init : tensor<?x?xf16>)
                      : tensor<?x?xf16> shape_region {
    ^bb0(%A: tensor<?x?xf16>, %B: tensor<?x?xf16>):
      %shA = shape.shape_of %A : ...   // [M, K]
      %shB = shape.shape_of %B : ...   // [K, N]
      %kA = shape.get_extent %shA, %c1 : ...   // A's K
      %kB = shape.get_extent %shB, %c0 : ...   // B's K
      %w  = shape.cstr_eq (shape %kA), (shape %kB)   // K must match
      %d:2 = shape.assuming %w -> (index, index) {
        %m = shape.get_extent %shA, %c0 : ...   // M
        %n = shape.get_extent %shB, %c1 : ...   // N
        shape.assuming_yield %m, %n : index, index
      }
      hipsr.shape_yield (%d#0, %d#1) : [f16]
    }
    ```
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
    attr-dict (`:` type($result_tensors)^)? (`shape_region` $shape_region^)?
  }];

  // Rejects rank-0 (scalar) A/B: matmul needs at least a contraction dim.
  let hasVerifier = 1;
}

#endif // HIPSR_MATMUL_OP
