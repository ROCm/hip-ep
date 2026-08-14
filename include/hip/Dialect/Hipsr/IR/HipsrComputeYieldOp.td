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
    Terminates a `hipsr.compute` body and gives that op its results.

    Operands are tensors before bufferization and memrefs after, in any
    `#hipsr.mem` space, and their types need not match the parent's `outs`
    types. A bufferized compute has no results, so its yield carries no
    operands.

    Example:
    ```mlir
    hipsr.compute_yield %out, %count : tensor<2x?xi64>, tensor<i32>
    ```
  }];

  let arguments = (ins Variadic<Hipsr_TensorOrAnySpaceMemRef>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [
    // SingleBlockImplicitTerminator requires a no-argument builder.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}
