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

def Hipsr_PlaceholderOp : Hipsr_Op<"placeholder"> {
  let summary = "Typed placeholder for hipsr DPS init operands";
  let description = [{
    Provides `outs` init values before a hipsr DPS op computes its output
    shape. Each DPS init must have the same type as the matching op result, so
    the placeholder uses the op's result types. The shape region still computes
    the output dimensions.

    One placeholder can have several results for one multi-output DPS op. Each
    result must be used once as a DPS init. All results must go to the same
    hipsr op.

    Each placeholder is tied to one DPS op. It is not `Pure`, so CSE cannot
    merge placeholders used by different ops. The pool-domain pass moves it
    next to the op that uses it.

    This op is temporary. It has no bufferization interface and must be replaced
    before bufferization.

    Example:
    ```mlir
    %inits:2 = hipsr.placeholder : tensor<?x?xf16>, tensor<?xi64>
    ```
  }];

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = "attr-dict `:` type($results)";

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes), [{
      $_state.addTypes(resultTypes);
    }]>
  ];
}
