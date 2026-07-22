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

    Carries an optional shape region (region 0) that yields the output shape.
    The contraction dim K is not in the output but must match on both sides, so
    it is guarded by a `shape.cstr_eq` witness around a `shape.assuming` region
    that produces the dims (a static-equal K folds the witness away). The region
    is isolated (see ShapeRegionInterface), so its body reads A and B as the
    entry-block args `%A`/`%B` below. Args mirror the DPS `ins` in order: arg 0
    is the (shape-unused) `ctx`, arg 1 A, arg 2 B.

    Example (populated region, 2-D; `^bb0(%ctxarg, %A, %B)` mirrors the ctx/A/B
    operands, `%ctxarg` unused):
    ```mlir
    %0 = hipsr.matmul(%ctx) ins(%Aop, %Bop : tensor<?x?xf16>, tensor<?x?xf16>)
                      outs(%init : tensor<?x?xf16>)
                      : tensor<?x?xf16> shape_region {
    ^bb0(%ctxarg: !hipsr.context, %A: tensor<?x?xf16>, %B: tensor<?x?xf16>):
      %shA = shape.shape_of %A : ...   // [M, K]
      %shB = shape.shape_of %B : ...   // [K, N]
      // Contraction dim K (A dim 1 == B dim 0): constrained, not in the result.
      %kA = shape.get_extent %shA, %c1 : ...   // A's K
      %kB = shape.get_extent %shB, %c0 : ...   // B's K
      %w  = shape.cstr_eq (shape %kA), (shape %kB)   // witness: K matches
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

  let extraClassDeclaration = [{
    /// Named view of the shape region's entry-block args: arg 0 = ctx (unused
    /// by the shape body), 1 = A, 2 = B. getA()/getB() return the block args,
    /// not the same-named operand accessors -- only the block args are legal
    /// inside the isolated region.
    struct ShapeRegionArgs {
      ::mlir::Block &block;
      /// Assert so a too-short arg list fails clearly, not as an out-of-range
      /// access deeper in.
      ::mlir::Value getA() const {
        assert(block.getNumArguments() > 1 &&
               "shape region entry block is missing the `A` argument "
               "(expected >= 2 block arguments: ctx, A, ...)");
        return block.getArgument(1);
      }
      ::mlir::Value getB() const {
        assert(block.getNumArguments() > 2 &&
               "shape region entry block is missing the `B` argument "
               "(expected >= 3 block arguments: ctx, A, B)");
        return block.getArgument(2);
      }
    };
  }];

  // `ctx` prints as a leading `(%ctx)` with its (single fixed) type elided,
  // matching the hip dialect DPS ops. Result type and shape region are optional
  // groups, so the memref/no-result and empty-region forms print nothing extra.
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
