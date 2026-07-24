//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.placeholder op: a typed stand-in that mirrors a hipsr DPS op's
// result type(s) so the DPS op can be built without first computing its output
// shape. The shaped DPS op base class lives in HipsrOps.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_PlaceholderOp
//===----------------------------------------------------------------------===//

def Hipsr_PlaceholderOp : Hipsr_Op<"placeholder", [Pure]> {
  let summary = "Typed placeholder for a hipsr DPS op's init operand(s)";
  let description = [{
    Stands in for the `outs` init operand(s) of a hipsr DPS op. It takes the DPS
    op's result type(s) and re-emits the same type(s) as its own results, which
    are then passed as the DPS op's init operands. This lets a DPS op be built
    without computing its output shape first: the shape region stays the source
    of truth for output dimensions.

    The DPS verifier requires each init type to equal the matching result type,
    and mirroring the result types makes that hold by construction. The op is
    variadic, so one placeholder can back one multi-output DPS op (one result
    per output). At the pool-domain partitioning stage, each result must have
    exactly one use, every result must initialize the same top-level hipsr DPS
    op, and every top-level hipsr DPS tensor init must come from such a
    placeholder. The partitioning pass verifies this transient staging
    contract and must run before CSE, which may merge same-typed pure
    placeholders.

    `hipsr.placeholder` is transient: it has no bufferization interface, so it
    must be replaced before bufferization runs.

    Example:
    ```mlir
    %inits:2 = hipsr.placeholder : tensor<?x?xf16>, tensor<?xi64>
    ```
  }];

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = "attr-dict `:` type($results)";

  let builders = [
    // Pass the DPS op's result types so init and result types match.
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes), [{
      $_state.addTypes(resultTypes);
    }]>
  ];
}
