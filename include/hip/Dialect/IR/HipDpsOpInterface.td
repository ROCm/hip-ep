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
    a single shared default `reifyResultShapes` body. In tensor mode it walks
    `DestinationStyleOpInterface::getDpsInits()` and lifts each init operand's
    runtime shape via `tensor::getMixedSizes`, producing exactly one vector per
    SSA result. In memref mode the operation has no SSA results, so reification
    succeeds with an empty list. Ops with a tighter shape contract -- e.g.
    matmul where the result shape can be computed from `A.shape[-2]` /
    `B.shape[-1]` rather than from the outs operand -- select a manual-reify
    family and provide a per-op override.

    The interface owns the default body in a sibling `.cpp` file, and
    the per-op `ReifyRankedShapedTypeOpInterface::reifyResultShapes` is
    a three-line dispatcher emitted by the operation's behavior family.
  }];

  let cppNamespace = "::mlir::hip";

  let extraClassDeclaration = [{
    /// Return one shape vector per tensor result, using its DPS destination.
    /// Keep static extents as attributes. Query dynamic extents with tensor.dim.
    /// Memref operations have no SSA results, so return an empty list.
    /// Tensor folding and canonicalization can resolve dimensions through casts.
    /// The shape-refinement pass does not remove these casts itself.
    ::mlir::LogicalResult reifyResultShapes(
        ::mlir::OpBuilder &b,
        ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  }];
}

#endif // HIP_DPS_OP_INTERFACE
