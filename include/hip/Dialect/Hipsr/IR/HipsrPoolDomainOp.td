//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.pool_domain op: a region that groups the ops sharing one memory
// pool. Its terminator, hipsr.pool_domain_yield, lives in
// HipsrPoolDomainYieldOp.td.
//

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
    Groups operations that share one memory pool.

    Pass `!hipsr.context` as the first operand. Hipsr ops in the body use the
    matching first block argument.

    `IsolatedFromAbove` makes every value crossing the boundary explicit.
    `RegionBranchOpInterface` checks the inputs and outputs.
    `RecursiveMemoryEffects` includes effects from the body.
    `AnyType` lets the domain stay in place during bufferization.

    Example:
    ```mlir
    %out = hipsr.pool_domain(%ctx, %input
        : !hipsr.context, tensor<?xf32>) {
    ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<?xf32>):
      %init = hipsr.placeholder : tensor<?xf16>
      %result = hipsr.cast(%domain_ctx)
          ins(%domain_input : tensor<?xf32>)
          outs(%init : tensor<?xf16>) : tensor<?xf16>
      hipsr.pool_domain_yield %result : tensor<?xf16>
    } -> tensor<?xf16>
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
