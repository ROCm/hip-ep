//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_OPS
#define HIPSR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Hipsr/IR/HipsrDialect.td"

//===----------------------------------------------------------------------===//
// Base op class
//===----------------------------------------------------------------------===//

// Common base for all ops in the hipsr dialect. Downstream PRs define concrete
// ops (shape-region terminators, shaped compute ops, ...) on top of this.
class Hipsr_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hipsr_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Hipsr_ShapeYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_ShapeYieldOp : Hipsr_Op<"shape_yield", [Pure, Terminator]> {
  let summary = "Terminator of a hipsr shape region";
  let description = [{
    Terminates a shape region and yields the computed output dimensions as
    `index` values. Every shape region ends with exactly one `shape_yield`.

    In Phase 1 a shape region is attached to a shaped op and computes that
    op's output shape; `shape_yield` hands the resulting per-dimension SSA
    values back to the enclosing op. A zero-operand `shape_yield` denotes a
    rank-0 (scalar) result.

    Example:
    ```mlir
    hipsr.shape_yield %d0, %d1 : index, index
    ```
  }];

  let arguments = (ins Variadic<Index>:$dims);

  let assemblyFormat = "attr-dict ($dims^ `:` type($dims))?";
}

#endif // HIPSR_OPS
