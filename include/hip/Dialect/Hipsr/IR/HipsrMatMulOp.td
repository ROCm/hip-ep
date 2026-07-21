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

    The `ctx` operand (`!hipsr.context`) is the per-session execution context.
    Every hipsr op carries it as its first operand; it is added in the ONNX
    phase and threaded through unchanged.

    Carries an optional shape region (region 0). When populated it yields the
    output shape. The contraction dim K is not in the output but must match on
    both sides, so it is constrained with a `shape.cstr_eq` witness guarding a
    `shape.assuming` region that produces the output dims (a static-equal K
    folds the witness away).

    Example (populated shape region, 2-D):
    ```mlir
    %0 = hipsr.matmul(%ctx) ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
                      outs(%init : tensor<?x?xf16>)
                      : tensor<?x?xf16> shape_region {
      %shA = shape.shape_of %A : tensor<?x?xf16> -> tensor<2xindex>
      %shB = shape.shape_of %B : tensor<?x?xf16> -> tensor<2xindex>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // Contraction dim K (A dim 1 == B dim 0): constrained, not in the result.
      %kA = shape.get_extent %shA, %c1 : tensor<2xindex>, index -> index
      %kB = shape.get_extent %shB, %c0 : tensor<2xindex>, index -> index
      %sa = shape.from_extents %kA : index
      %sb = shape.from_extents %kB : index
      %w = shape.cstr_eq %sa, %sb : !shape.shape, !shape.shape
      %d:2 = shape.assuming %w -> (index, index) {
        %m = shape.get_extent %shA, %c0 : tensor<2xindex>, index -> index
        %n = shape.get_extent %shB, %c1 : tensor<2xindex>, index -> index
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

  // Matches the hip dialect DPS ops: `ctx` prints as a leading `(%ctx)` with
  // its type elided (it is a single fixed type). Result type and shape region
  // are both optional groups so a memref/no-result form and the empty-region
  // form (the one convert-onnx-to-hipsr emits) print nothing extra.
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
