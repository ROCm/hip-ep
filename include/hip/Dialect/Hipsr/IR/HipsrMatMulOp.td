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

    Carries an optional shape region (region 0). When populated it yields the
    output shape and asserts the matmul shape constraints: A and B must share the
    contraction dim K, and the batch dims must be broadcastable (equal, or one
    side is 1).

    Example (populated shape region, 2-D):
    ```mlir
    %0 = hipsr.matmul ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
                      outs(%init : tensor<?x?xf16>)
                      -> tensor<?x?xf16> shape_region {
      %shA = shape.shape_of %A : tensor<?x?xf16> -> tensor<2xindex>
      %shB = shape.shape_of %B : tensor<?x?xf16> -> tensor<2xindex>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %ka = shape.get_extent %shA, %c1 : tensor<2xindex>, index -> index
      %kb = shape.get_extent %shB, %c0 : tensor<2xindex>, index -> index
      %eq = arith.cmpi eq, %ka, %kb : index
      cf.assert %eq, "hipsr.matmul: A and B contraction dim (K) must be equal"
      %m = shape.get_extent %shA, %c0 : tensor<2xindex>, index -> index
      %n = shape.get_extent %shB, %c1 : tensor<2xindex>, index -> index
      hipsr.shape_yield (%m, %n) : [f16]
    }
    ```
  }];

  let arguments = (ins
    Hipsr_TensorOrDeviceMemRef:$A,
    Hipsr_TensorOrDeviceMemRef:$B,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  // The shape region is optional: an empty (zero-block) region prints nothing
  // and round-trips as empty. A non-empty region prints as
  // `shape_region { ... }`.
  let assemblyFormat = [{
    `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict `->` type($result_tensors) (`shape_region` $shape_region^)?
  }];

  // Rejects rank-0 (scalar) A/B: matmul needs at least a contraction dim.
  let hasVerifier = 1;
}

#endif // HIPSR_MATMUL_OP
