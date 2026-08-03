//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// hipsr.placeholder gives DPS ops typed init values before computing output
// shapes.
//

//===----------------------------------------------------------------------===//
// Hipsr_PlaceholderOp
//===----------------------------------------------------------------------===//

def Hipsr_PlaceholderOp : Hipsr_Op<
    "placeholder",
    [IsolatedFromAbove, SingleBlockImplicitTerminator<"ShapeYieldOp">]> {
  let summary = "Typed placeholder for hipsr DPS init operands";
  let description = [{
    Provides `outs` init values before a hipsr DPS op computes its output
    shape. Each DPS init must have the same type as the matching op result, so
    the placeholder uses the op's result types.

    `!hipsr.context` is operand 0. The remaining inputs provide values to the
    shape graph and must be block arguments or results of `hipsr.placeholder`,
    `arith.constant`, or `hipsr.constant`. `type` marks the placeholder as
    `normal` or `barrier`.

    The shape graph is independent of DPS data inputs: downstream placeholders
    use upstream placeholder results rather than DPS data results.

    The optional `shape_region` defines the placeholder result shapes. It has
    one block when present and ends with `hipsr.shape_yield`.

    One placeholder can have several results for one multi-output DPS op. Each
    result initializes exactly one DPS result and may also feed other
    placeholders. All DPS init uses must go to the same hipsr op.

    Each placeholder is tied to one DPS op. It is not `Pure`, so CSE cannot
    merge placeholders used by different ops.

    This op is temporary. It has no bufferization interface and must be replaced
    before bufferization.

    Example:
    ```mlir
    %init = hipsr.placeholder(%ctx)
        ins(%input : tensor<?x?xf16>)
        {type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<AnyRankedTensor>:$inputs,
    Hipsr_PlaceholderTypeAttr:$type
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region MaxSizedRegion<1>:$shape_region);

  let assemblyFormat = [{
    `(` $ctx `)` (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    attr-dict `:` type($results) (`shape_region` $shape_region^)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns whether a value may be used as a placeholder shape-graph input.
    static bool isAllowedShapeGraphInput(::mlir::Value value);

    /// Returns the hipsr DPS operation initialized by this placeholder, or
    /// null if no such use exists.
    ::mlir::Operation *getDpsConsumer();
  }];

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::Value":$ctx,
                   "::mlir::ValueRange":$inputs,
                   "::mlir::hipsr::PlaceholderType":$type), [{
      $_state.addOperands(ctx);
      $_state.addOperands(inputs);
      $_state.addAttribute(
          "type", ::mlir::hipsr::PlaceholderTypeAttr::get(
                      $_builder.getContext(), type));
      $_state.addTypes(resultTypes);
      $_state.addRegion();
    }]>
  ];
}
