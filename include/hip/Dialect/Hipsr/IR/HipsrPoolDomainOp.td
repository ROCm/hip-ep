//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.pool_domain op: a region that groups the ops sharing one memory
// pool. Its terminator, hipsr.pool_domain_yield, lives in
// HipsrPoolDomainYieldOp.td.
//

#ifndef HIPSR_POOL_DOMAIN_OP
#define HIPSR_POOL_DOMAIN_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Hipsr_PoolDomainOp
//===----------------------------------------------------------------------===//

def Hipsr_PoolDomainOp : Hipsr_Op<"pool_domain", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"PoolDomainYieldOp">
  ]> {
  let summary = "Group operations that share one memory pool";
  let description = [{
    Groups a sequence of operations so they can share one memory pool.

    The body has one block and ends with `hipsr.pool_domain_yield`.
    `IsolatedFromAbove` prevents direct use of values from the parent scope.
    Each external value must be an op operand with a matching block argument.

    `RegionBranchOpInterface` tells MLIR how values enter and leave the body.
    MLIR's verifier checks that operand types match block argument types and
    yielded types match result types. `RecursiveMemoryEffects` reports effects
    from operations inside the body. Operands and results are `AnyType`, so the
    same op works with tensors before bufferization and memrefs after it.

    Example:
    ```mlir
    %out = hipsr.pool_domain(%input : tensor<?xf32>) {
    ^bb0(%domain_input: tensor<?xf32>):
      %c0 = arith.constant 0 : index
      %size = tensor.dim %domain_input, %c0 : tensor<?xf32>
      %buffer = tensor.empty(%size) : tensor<?xf32>
      hipsr.pool_domain_yield %buffer : tensor<?xf32>
    } -> tensor<?xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  // Operands in parens, results after `->`; both optional so an empty domain
  // prints cleanly.
  let assemblyFormat = [{
    `(` ($operands^ `:` type($operands))? `)` $body
    (`->` type($results)^)? attr-dict
  }];
}

#endif // HIPSR_POOL_DOMAIN_OP
