//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.compute op: a DPS op whose region groups a multi-operation
// computation. Its terminator, hipsr.compute_yield, lives in
// HipsrComputeYieldOp.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_ComputeOp
//===----------------------------------------------------------------------===//

def Hipsr_ComputeOp : Hipsr_Op<"compute", [
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                              ["getDpsInitsMutable"]>,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ComputeYieldOp">
  ]> {
  let summary = "Group a multi-operation computation as one DPS op";
  let description = [{
    Groups the operations that implement one high-level computation, so an ONNX
    op that needs several tensor or shape operations still appears as a single
    hipsr op. Without the grouping those intermediates stay in the enclosing
    block, where a later op's `hipsr.placeholder` cannot use one: a placeholder
    input must be a block argument or the result of a constant or another
    placeholder.

    Each tensor init is tied to the result in the same position. The inits come
    from `hipsr.placeholder`, which owns the output-shape computation, so this
    op has no shape region of its own.

    `IsolatedFromAbove` and `RegionBranchOpInterface` make every value crossing
    the boundary explicit and verifiable, so `ctx`, the inputs, and the inits
    enter the body as entry-block arguments, in that order.
    `RecursiveMemoryEffects` keeps nested effects visible to analyses, and
    `AutomaticAllocationScope` frees allocations made in the body when it exits.

    Example:
    ```mlir
    %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                               outs(%init : tensor<6xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>, %dest: tensor<6xf16>):
      %flat = tensor.collapse_shape %in [[0, 1]]
          : tensor<2x3xf16> into tensor<6xf16>
      %filled = tensor.insert_slice %flat into %dest[0] [6] [1]
          : tensor<6xf16> into tensor<6xf16>
      hipsr.compute_yield %filled : tensor<6xf16>
    } : tensor<6xf16>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<AnyType>:$inputs,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$inits
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` ($inputs^ `:` type($inputs))? `)`
    `outs` `(` ($inits^ `:` type($inits))? `)`
    $body (`:` type($result_tensors)^)? attr-dict
  }];
}
