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
    a single shared default `reifyResultShapes` body that walks
    `DestinationStyleOpInterface::getDpsInits()` and lifts each init
    operand's runtime shape via `tensor::getMixedSizes` (tensor mode) or
    `memref::getMixedSizes` (memref mode). Ops with a tighter shape
    contract -- e.g. matmul where the result shape can be computed from
    `A.shape[-2]` / `B.shape[-1]` rather than from the outs operand -- opt
    out by setting `autoReify=0` on `Hip_DpsOp` and providing a per-op
    override.

    The interface owns the default body in a sibling `.cpp` file, and
    the per-op `ReifyRankedShapedTypeOpInterface::reifyResultShapes` is
    a 3-line dispatcher auto-emitted from `Hip_DpsOp`'s
    `extraClassDefinition`.
  }];

  let cppNamespace = "::mlir::hip";

  let extraClassDeclaration = [{
    /// Default reify implementation. Returns a per-result vector of
    /// `OpFoldResult` -- static dims as `IntegerAttr`, dynamic dims as
    /// `tensor.dim` / `memref.dim` of the corresponding outs operand.
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
  }];
}

#endif // HIP_DPS_OP_INTERFACE
