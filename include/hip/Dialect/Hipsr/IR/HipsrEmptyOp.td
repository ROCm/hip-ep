//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.empty op: an init-tensor carrier whose region computes the output
// shape. Its terminator, hipsr.empty_yield, lives in HipsrEmptyYieldOp.td.
//

#ifndef HIPSR_EMPTY_OP
#define HIPSR_EMPTY_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Hipsr_EmptyOp
//===----------------------------------------------------------------------===//

def Hipsr_EmptyOp : Hipsr_Op<"empty",
    [Pure, DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"EmptyYieldOp">]> {
  let summary = "Create init tensor(s) with shape computation in a region";
  let description = [{
    Creates init tensor(s) for destination-passing style operations. Shape
    computation is encapsulated in the attached region, which executes to
    produce the init tensor(s).

    The region contains operations that compute each tensor's shape (dimensions
    and element type), creates a tensor.empty() per result with those
    dimensions, and yields them via hipsr.empty_yield.

    `RegionBranchOpInterface` models the region's yielded values as this
    operation's results and verifies their counts and types.

    Example (single-output):
    ```mlir
    %init = hipsr.empty() : tensor<?x8xf16> {
      %shape = shape.shape_of %input : tensor<?x8xf32> -> tensor<2xindex>
      %c0 = arith.constant 0 : index
      %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
      %t = tensor.empty(%d0) : tensor<?x8xf16>
      hipsr.empty_yield %t : tensor<?x8xf16>
    }
    ```

    Example (multi-result):
    ```mlir
    %init:2 = hipsr.empty() : tensor<?x?xf16>, tensor<?xi64> {
      %t0 = tensor.empty(%n, %m) : tensor<?x?xf16>
      %t1 = tensor.empty(%n) : tensor<?xi64>
      hipsr.empty_yield %t0, %t1 : tensor<?x?xf16>, tensor<?xi64>
    }
    ```
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = "`(` `)` attr-dict `:` type($results) $region";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Mixed static/dynamic sizes per result, preserving dimension position:
    /// [[sizes for result 0], ...]. Each entry is an Attribute (static extent)
    /// or a Value (runtime extent), mirroring tensor::EmptyOp::getMixedSizes().
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::OpFoldResult>>
    getMixedSizes();

    ::llvm::SmallVector<::mlir::RankedTensorType> getTensorTypes();
  }];
}

#endif // HIPSR_EMPTY_OP
