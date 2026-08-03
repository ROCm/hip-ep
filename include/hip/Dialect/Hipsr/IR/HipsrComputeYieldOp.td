//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.compute_yield op: the terminator of a hipsr.compute region. The
// hipsr.compute op itself lives in HipsrComputeOp.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_ComputeYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_ComputeYieldOp : Hipsr_Op<"compute_yield",
    [Pure, ReturnLike, Terminator, HasParent<"ComputeOp">]> {
  let summary = "Yield the computed values out of a hipsr.compute region";
  let description = [{
    Returns values from a compute body to the parent op.

    `ReturnLike` lets the parent's `RegionBranchOpInterface` verify these
    outputs. Because the op only forwards values, `Pure` lets effect analyses
    ignore it. Operands are ranked tensors, like the results of
    `hipsr.compute`; a bufferized compute has no results and yields nothing.

    Example:
    ```mlir
    hipsr.compute_yield %out, %count : tensor<2x?xi64>, tensor<i32>
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [
    // SingleBlockImplicitTerminator requires a no-argument builder.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}
