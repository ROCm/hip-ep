//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.compute op: a non-DPS op whose region groups a multi-operation
// computation. Its terminator, hipsr.compute_yield, lives in
// HipsrComputeYieldOp.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_ComputeOp
//===----------------------------------------------------------------------===//

def Hipsr_ComputeOp : Hipsr_Op<"compute", [
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    RecursivelySpeculatable,
    SingleBlockImplicitTerminator<"ComputeYieldOp">
  ]> {
  let summary = "Group a multi-operation computation as one non-DPS op";
  let description = [{
    Groups the ops of one high-level computation into a single hipsr op, so an
    ONNX op that lowers to several tensor or shape ops stays one op here. The
    isolated body takes `ctx`, then the inputs, then the outputs as its block
    arguments, and `hipsr.compute_yield` returns the results.

    An `outs` entry names where the result in the same position is stored. Its
    type need not match that result's, so the body may yield a reshaped or cast
    view of it.

    Operands and results are tensors before bufferization and device memrefs
    (`#hipsr.mem<device>`) after. A bufferized compute has no results and
    yields nothing.

    Example, where the body fills its output and then flattens it, so the
    result type differs:
    ```mlir
    %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                               outs(%init : tensor<2x3xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>,
         %dest: tensor<2x3xf16>):
      %written = tensor.insert_slice %in into %dest[0, 0] [2, 3] [1, 1]
          : tensor<2x3xf16> into tensor<2x3xf16>
      %flat = tensor.collapse_shape %written [[0, 1]]
          : tensor<2x3xf16> into tensor<6xf16>
      hipsr.compute_yield %flat : tensor<6xf16>
    } : tensor<6xf16>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$inputs,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$outputs
  );
  // Named after linalg's $result_tensors, which is also the pre-bufferization
  // form; a bufferized compute holds memrefs here.
  let results = (outs Variadic<Hipsr_TensorOrDeviceMemRef>:$result_tensors);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` ($inputs^ `:` type($inputs))? `)`
    `outs` `(` ($outputs^ `:` type($outputs))? `)`
    $body (`:` type($result_tensors)^)? attr-dict
  }];

  let hasVerifier = 1;
}
