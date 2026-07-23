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

//===----------------------------------------------------------------------===//
// Hipsr_EmptyOp
//===----------------------------------------------------------------------===//

// Each hipsr.empty is a fresh init-tensor allocation (like
// bufferization.alloc_tensor), so it is not memory-effect-free: distinct ops
// yield distinct buffers even when their shapes match.
def Hipsr_EmptyOp : Hipsr_Op<"empty",
    [SingleBlockImplicitTerminator<"EmptyYieldOp">]> {
  let summary = "Create init tensor(s) with shape computation in a region";
  let description = [{
    Produces the `outs` init tensor(s) for a DPS op. The attached region
    computes each result's shape (typically `shape.shape_of` + `shape.get_extent`
    for a regular op, or `hipsr.readback_scalar` for a data-dependent one),
    builds a `tensor.empty()` per result, and yields them via `hipsr.empty_yield`.

    Unlike pool_domain / DPS compute ops, the region is NOT isolated: it may read
    parent-scope values (e.g. `shape.shape_of %input`, where %input is a domain
    operand). It is variadic so one hipsr.empty can back a multi-output DPS op.

    Bufferizes to one `memref.alloc()` per yielded tensor.

    Example:
    ```mlir
    %init = hipsr.empty() : tensor<?x8xf16> {
      %shape = shape.shape_of %input : tensor<?x8xf32> -> tensor<2xindex>
      %c0 = arith.constant 0 : index
      %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
      %t = tensor.empty(%d0) : tensor<?x8xf16>
      hipsr.empty_yield %t : tensor<?x8xf16>
    }
    ```
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  // Result types after `:`, then the region. Matches the pool_domain layout.
  let assemblyFormat = "`(` `)` attr-dict `:` type($results) $region";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Runtime dim SSA values per result: [[dims for result 0], ...].
    /// Reads each yielded tensor.empty's dynamic sizes. Region must be populated.
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>> getShapes();

    /// Compile-time tensor types of all results.
    ::llvm::SmallVector<::mlir::RankedTensorType> getTensorTypes();
  }];
}

#endif // HIPSR_EMPTY_OP
