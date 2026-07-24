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
    Terminates a hipsr.empty region and yields the computed init tensors.
    The yielded tensors become the results of the hipsr.empty operation.
    Supports variadic operands for multi-result operations.

    Example (single result):
    ```mlir
    hipsr.empty_yield %tensor : tensor<?x512xf16>
    ```
    Example (multi-result):
    ```mlir
    hipsr.empty_yield %tensor0, %tensor1 : tensor<?x?xf16>, tensor<?x?xi64>
    ```

  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$tensors);

  let assemblyFormat = "($tensors^ `:` type($tensors))? attr-dict";

  let builders = [
    // No-arg build for the implicit terminator the parent auto-inserts.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // HIPSR_EMPTY_YIELD_OP
