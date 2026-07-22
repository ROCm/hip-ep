//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ShapeRegionInterface. The pool-domain barriers are separate interfaces:
// StartBarrierInterface (HipsrStartBarrierInterface.td) and EndBarrierInterface
// (HipsrEndBarrierInterface.td).
//

#ifndef HIPSR_SHAPE_REGION_INTERFACE
#define HIPSR_SHAPE_REGION_INTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ShapeRegionInterface
//===----------------------------------------------------------------------===//

def ShapeRegionInterface : OpInterface<"ShapeRegionInterface"> {
  let cppNamespace = "::mlir::hipsr";

  let description = [{
    Interface for ops that compute their own output shape in an attached region.

    An implementing op carries an optional shape region (region 0): either
    empty (no shape computation yet) or a single block that computes the output
    dimensions and ends with `hipsr.shape_yield`.

    The region is `IsolatedFromAbove`, so it may not read values from the
    enclosing scope (not even the op's own operands). The op's DPS `ins`
    operands are instead passed in as entry-block arguments, one per input in
    order (block arg `i` mirrors DPS input `i`). Isolation is what lets stock
    MLIR passes (-cse, -canonicalize, LICM) treat the region correctly; without
    it they would hoist region ops out or merge them with identical outer ops.
    The `hipsr-populate-shape-region` pass creates the entry block with those
    arguments, then calls `populateShapeRegion` to fill the body.

    An implementing op MUST also carry the `IsolatedFromAbove` and `SingleBlock`
    traits (an interface cannot auto-attach traits), e.g.

      def Hipsr_MatMulOp : Hipsr_Op<"matmul", [
          ShapeRegionInterface, IsolatedFromAbove, SingleBlock]> { ... }

    `SingleBlock` keeps a populated region to one block while still permitting
    the empty (unpopulated) state. The interface verifier does the rest: region
    structure, the trailing `hipsr.shape_yield`, and that `IsolatedFromAbove`
    is present.

    Op categories (barrier markers are separate interfaces):
    - Regular (MatMul, Add): one shape, from the input shapes.
    - StartBarrier (Expand): one shape, from the input data.
    - EndBarrier (NonZero): two shapes (capacity + real) over two regions.
  }];

  let methods = [
    InterfaceMethod<
      [{Fill the (already-created, empty) entry block with the ops computing the
        output shape. Reach the block via `getShapeBlock()`, then read inputs as
        block arguments -- the region is isolated, so the op's operands are off
        limits.
      }],
      "void", "populateShapeRegion",
      (ins "::mlir::OpBuilder &":$builder)
    >,
    InterfaceMethod<
      [{Fill the capacity shape region (region 1). Only EndBarrier ops override
        this. Same block-arg contract as `populateShapeRegion`, reached via
        `getCapacityShapeBlock()`; confirm the op is an EndBarrier first.
      }],
      "void", "populateCapacityShapeRegion",
      (ins "::mlir::OpBuilder &":$builder),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        // Must stay non-empty: an empty defaultImplementation makes TableGen
        // treat the method as a mandatory override, so every op that omits it
        // fails to link. Regular/StartBarrier ops have no capacity region.
        return;
      }]
    >,
  ];

  // Non-virtual on purpose: only the populate* methods vary per op. These
  // accessors are fixed by the region layout (region 0 = shape, 1 = capacity)
  // or read the yield terminator, so one shared body serves every op. (Mirrors
  // DestinationStyleOpInterface, where only getDpsInitsMutable is virtual.)
  let extraSharedClassDeclaration = [{
    /// Shape region (region 0).
    ::mlir::Region &getShapeRegion() {
      return $_op->getRegion(0);
    }

    /// Capacity shape region (region 1), or nullptr if the op has none.
    ::mlir::Region *getCapacityShapeRegion() {
      return $_op->getNumRegions() > 1 ? &$_op->getRegion(1) : nullptr;
    }

    /// Entry block of the shape region, or nullptr if not created yet. Pure
    /// query, so it is safe to call defensively. A populator's null result is a
    /// pass-precondition bug (the block is created before populate runs), unlike
    /// getCapacityShapeBlock() where nullptr is an expected answer.
    ::mlir::Block *getShapeBlock() {
      ::mlir::Region &region = getShapeRegion();
      return region.empty() ? nullptr : &region.front();
    }

    /// Entry block of the capacity region, or nullptr. Region 1 is optional
    /// (only EndBarrier ops have it), so nullptr is a normal state to branch on.
    ::mlir::Block *getCapacityShapeBlock() {
      ::mlir::Region *r = getCapacityShapeRegion();
      return (r && !r->empty()) ? &r->front() : nullptr;
    }

    /// Each result's dim values, grouped per result (handy for `tensor.empty`).
    /// Requires the shape body already emitted (reads its `hipsr.shape_yield`).
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
    getShapeRegionResultShapes() {
      return ::mlir::hipsr::detail::getShapeRegionResultShapes(getShapeRegion());
    }

    /// Each result's tensor type (all extents dynamic; constant dims fold back
    /// to static later via the tensor.empty canonicalizer). Requires the shape
    /// body already emitted.
    ::llvm::SmallVector<::mlir::RankedTensorType>
    getShapeRegionResultTypes() {
      return ::mlir::hipsr::detail::getShapeRegionResultTypes(getShapeRegion());
    }

    /// getShapeRegionResultShapes for the capacity region (EndBarrier ops only).
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
    getCapacityShapeRegionResultShapes() {
      ::mlir::Region *region = getCapacityShapeRegion();
      assert(region && "op has no capacity shape region (region 1)");
      return ::mlir::hipsr::detail::getShapeRegionResultShapes(*region);
    }

    /// getShapeRegionResultTypes for the capacity region (EndBarrier ops only).
    ::llvm::SmallVector<::mlir::RankedTensorType>
    getCapacityShapeRegionResultTypes() {
      ::mlir::Region *region = getCapacityShapeRegion();
      assert(region && "op has no capacity shape region (region 1)");
      return ::mlir::hipsr::detail::getShapeRegionResultTypes(*region);
    }
  }];

  let verify = [{
    if (::mlir::failed(::mlir::hipsr::verifyShapeRegionStructure($_op)))
      return ::mlir::failure();
    // IsolatedFromAbove's own verifier does the isolation check; just require
    // the trait is present.
    if (!$_op->hasTrait<::mlir::OpTrait::IsIsolatedFromAbove>())
      return $_op->emitOpError(
          "ShapeRegionInterface requires the IsolatedFromAbove trait");
    return ::mlir::success();
  }];
}

#endif // HIPSR_SHAPE_REGION_INTERFACE
