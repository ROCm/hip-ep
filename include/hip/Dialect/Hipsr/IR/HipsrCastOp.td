//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.cast op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

#ifndef HIPSR_CAST_OP
#define HIPSR_CAST_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"

//===----------------------------------------------------------------------===//
// Hipsr_CastOp
//===----------------------------------------------------------------------===//

def Hipsr_CastOp : Hipsr_DpsOp<"cast"> {
  let summary = "Elementwise type cast, destination-passing style";
  let description = [{
    Casts `input` to the element type of the result (ONNX Cast). The output
    has the same shape as the input; only the element type changes.
    Destination-passing style: the result buffer is passed as the `init`
    operand.

    The `ctx` operand (`!hipsr.context`) is the per-session execution context.
    Every hipsr op carries it as its first operand; it is added in the ONNX
    phase and threaded through unchanged.

    Carries an optional shape region (region 0) that yields the output shape:
    the input shape, with the output element type. The region is isolated (see
    ShapeRegionInterface), so its body reads the input as the entry-block arg
    `%in` below (arg 0 is the shape-unused `ctx`), not the op's `input` operand.

    Example (populated region; `^bb0(%ctxarg, %in)` mirrors the ctx/input
    operands, `%ctxarg` unused):
    ```mlir
    %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>) outs(%init : tensor<?x8xf16>)
                          : tensor<?x8xf16> shape_region {
    ^bb0(%ctxarg: !hipsr.context, %in: tensor<?x8xf32>):
      %shape = shape.shape_of %in : tensor<?x8xf32> -> tensor<2xindex>
      %d0 = shape.get_extent %shape, %c0 : ...   // dim 0
      %d1 = shape.get_extent %shape, %c1 : ...   // dim 1
      hipsr.shape_yield (%d0, %d1) : [f16]
    }
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$input,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let extraClassDeclaration = [{
    /// Named view of the shape region's entry-block args: arg 0 = ctx (unused
    /// by the shape body), 1 = input. getInput() returns the block arg, not the
    /// same-named operand accessor -- only the block arg is legal inside the
    /// isolated region.
    struct ShapeRegionArgs {
      ::mlir::Block &block;
      /// Assert so a too-short arg list fails clearly, not as an out-of-range
      /// access deeper in.
      ::mlir::Value getInput() const {
        assert(block.getNumArguments() > 1 &&
               "shape region entry block is missing the `input` argument "
               "(expected >= 2 block arguments: ctx, input)");
        return block.getArgument(1);
      }
    };
  }];

  // `ctx` prints as a leading `(%ctx)` with its (single fixed) type elided,
  // matching the hip dialect DPS ops. Result type and shape region are optional
  // groups, so the memref/no-result and empty-region forms print nothing extra.
  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)? (`shape_region` $shape_region^)?
  }];
}

#endif // HIPSR_CAST_OP
