//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// hipsr.placeholder gives hipsr ops typed `outs` values before computing
// output shapes.
//

//===----------------------------------------------------------------------===//
// Hipsr_PlaceholderOp
//===----------------------------------------------------------------------===//

def Hipsr_PlaceholderOp : Hipsr_Op<
    "placeholder",
    [IsolatedFromAbove, SingleBlockImplicitTerminator<"ShapeYieldOp">]> {
  let summary = "Typed placeholder for hipsr `outs` operands";
  let description = [{
    Supplies a hipsr op's `outs` values before its output shape is known. The
    result type follows the consumer:

    - A DPS op ties each init to the result in the same position, so the
      placeholder carries the op's result types.
    - `hipsr.compute` is not DPS. Its `outs` entry is only the destination the
      body writes into, so the placeholder carries that destination's shape,
      which may differ from the result held there.

    Inputs feed the shape graph and must be block arguments or results of
    `hipsr.placeholder`, `arith.constant`, or `hipsr.constant` — never data
    results, which keeps the shape graph parallel to the data graph.
    `placeholder_type` is `normal` or `barrier`.

    A placeholder and its consumer read the same values, the placeholder on the
    shape-graph side and the consumer on the data side, even when one of them
    ignores a value. Matching the two lists is what keeps the pair in one pool
    domain. Values with no placeholder behind them, such as block arguments and
    constants, are outside the rule.

    The optional `shape_region` computes the result shapes and ends with
    `hipsr.shape_yield`. A `normal` region omits `ctx` and takes one extent
    tensor per input, with one extent per input dimension; a `barrier` region
    takes `ctx` followed by its inputs.

    All results must fill `outs` slots of one op, one slot each, and may also
    feed other placeholders or leave a pool domain through
    `hipsr.pool_domain_yield`. Being tied to a single op, it is not `Pure`, so
    CSE cannot merge placeholders.

    The pool-domain pass requires a placeholder and its consumer to sit
    directly in the function body. Both share one destination buffer, so the
    input rule above keeps them in one pool domain. A `barrier` puts that pair
    in the domain after the values its shape region reads; a `normal`
    placeholder sets no boundary. A result that a later domain needs is returned
    from the domain holding it, which keeps the shape graph connected across the
    boundary.

    This op is temporary: it has no bufferization interface and must be
    replaced before bufferization.

    Example:
    ```mlir
    %init = hipsr.placeholder(%ctx)
        ins(%input : tensor<?x?xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x?xf16>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<AnyRankedTensor>:$inputs,
    Hipsr_PlaceholderTypeAttr:$placeholder_type
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

    /// Returns the hipsr operation whose `outs` this placeholder fills, or
    /// null if no such use exists.
    ::mlir::Operation *getConsumer();

    /// Returns this placeholder's shape-region block argument types.
    ::llvm::SmallVector<::mlir::Type> getShapeRegionArgumentTypes();
  }];
}
