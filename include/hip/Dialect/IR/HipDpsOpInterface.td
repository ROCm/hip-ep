//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DPS_OP_INTERFACE
#define HIP_DPS_OP_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

def HipDpsOpInterface : OpInterface<"HipDpsOp", [DestinationStyleOpInterface]> {
  let description = [{
    Marker interface for HIP destination-passing-style compute ops. Carries
    shared `reifyResultShapes` and `reifyDimOfResult` bodies. In tensor mode
    whole-shape reification walks
    `DestinationStyleOpInterface::getDpsInits()` and lifts each init operand's
    runtime shape via `tensor::getMixedSizes`, producing exactly one vector per
    SSA result. In memref mode the operation has no SSA results, so reification
    succeeds with an empty list. Ops with a tighter shape contract -- e.g.
    matmul where the result shape can be computed from `A.shape[-2]` /
    `B.shape[-1]` rather than from the outs operand -- select a manual-reify
    family and provide a per-op override.

    Single-dimension reification follows the exact tied DPS destination through
    `DestinationStyleOpInterface::getTiedOpOperand`. This avoids rebuilding
    every result dimension when an upstream dim-resolution pattern requests
    only one. The interface owns both default bodies in a sibling `.cpp` file;
    operation behavior families emit the standard-interface dispatchers.
  }];

  let cppNamespace = "::mlir::hip";

  let extraClassDeclaration = [{
    /// Default reify implementation. In tensor mode, returns a per-result
    /// vector of `OpFoldResult` -- static dims as `IntegerAttr`, dynamic dims
    /// as `tensor.dim` of the corresponding outs operand. In memref mode,
    /// returns success with no vectors because the op has no SSA results.
    /// Folds through `tensor.cast` because `tensor::DimOp::fold` peels
    /// casts (via `foldTensorCast`) and the `DimOfCastOp` canonicalizer
    /// rewrites `dim(cast(x), i)` into `dim(x, i)`. As a result, chained
    /// refinement still recovers already-narrowed dims even across the
    /// per-op refinement barrier (a `tensor.cast` on non-DPS-init uses)
    /// inserted by `--hip-infer-shapes`. The cast peel is performed by
    /// the folder/canonicalizer, not by `--hip-infer-shapes` itself.
    ::mlir::LogicalResult reifyResultShapes(
        ::mlir::OpBuilder &b,
        ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    /// Return one result dimension from its tied DPS destination. All result,
    /// destination, and dimension checks complete before this method emits IR.
    ::llvm::FailureOr<::mlir::OpFoldResult> reifyDimOfResult(
        ::mlir::OpBuilder &b, int resultIndex, int dim);
  }];
}

#endif // HIP_DPS_OP_INTERFACE
