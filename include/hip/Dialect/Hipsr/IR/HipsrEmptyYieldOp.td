//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.empty_yield op: the terminator of a hipsr.empty region. The
// hipsr.empty op itself lives in HipsrEmptyOp.td.
//

#ifndef HIPSR_EMPTY_YIELD_OP
#define HIPSR_EMPTY_YIELD_OP

include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Hipsr/IR/HipsrOps.td"

//===----------------------------------------------------------------------===//
// Hipsr_EmptyYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_EmptyYieldOp : Hipsr_Op<"empty_yield",
    [Pure, Terminator, HasParent<"EmptyOp">]> {
  let summary = "Yield the init tensor(s) from a hipsr.empty region";
  let description = [{
    Ends the body of a `hipsr.empty`. Its operands (each a `tensor.empty`
    result) become the parent op's results, matched by the parent's verifier.
    Operands are `AnyRankedTensor`: `hipsr.empty` is a tensor-stage op, so it
    can only yield tensors before bufferization.

    Example:
    ```mlir
    hipsr.empty_yield %t0, %t1 : tensor<?x?xf16>, tensor<?xi64>
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$tensors);

  // Operands optional so the implicit terminator prints as just the mnemonic.
  let assemblyFormat = "($tensors^ `:` type($tensors))? attr-dict";

  let builders = [
    // No-arg build for the implicit terminator the parent auto-inserts.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // HIPSR_EMPTY_YIELD_OP
