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

    Operands give `hipsr.compute` its result types; because that op is not
    destination-passing style, those types need not match its `outs` types.
    Bufferization turns the operands into the device memrefs that back them,
    which is how a result whose type differs from its `outs` gets its buffer
    out of the isolated body. A compute with no results yields nothing.

    Example:
    ```mlir
    hipsr.compute_yield %out, %count : tensor<2x?xi64>, tensor<i32>
    ```
  }];

  let arguments = (ins Variadic<Hipsr_TensorOrDeviceMemRef>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [
    // SingleBlockImplicitTerminator requires a no-argument builder.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}
