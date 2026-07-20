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
    Base interface for shape computation.
    Provides populateShapeRegion() method - single source of truth.

    An implementing op carries an optional shape region (region 0): either
    empty (zero blocks, no shape computation yet) or a single block whose body
    computes the op's output dimensions and ends with a `hipsr.shape_yield`.
    The populate* methods emit that region; the get* accessors retrieve it.

    REQUIREMENT: an op implementing this interface MUST also add two traits
    (MLIR interfaces cannot auto-attach traits), e.g.

      def Hipsr_MatMulOp : Hipsr_Op<"matmul", [
          ShapeRegionInterface,
          IsolatedFromAboveButAllowOperands,
          SingleBlock
      ]> { ... }

    `SingleBlock` keeps a non-empty region to a single block without inserting
    or eliding a terminator (so an empty region stays empty);
    `IsolatedFromAboveButAllowOperands` restricts what the shape region may use.
    The interface verifier checks the region structure -- including that a
    non-empty region ends with `hipsr.shape_yield` -- and that the
    IsolatedFromAboveButAllowOperands trait is present (whose own verifier does
    the scoping check).

    Operation categories (barrier markers are separate interfaces):
    - Regular (MatMul, Add): no block arguments, one shape, depends on input shapes.
    - StartBarrier (Expand): no block arguments, one shape, depends on input data.
    - EndBarrier (NonZero): has block arguments, two shapes (capacity + real), depends on output data.
  }];

  let methods = [
    InterfaceMethod<
      "Populate the shape region with the ops that compute the output shape.",
      "void", "populateShapeRegion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Region &":$shapeRegion)
    >,
    InterfaceMethod<
      [{Populate the capacity shape region for EndBarrier operations.

        Only EndBarrier operations override this method. Regular and
        StartBarrier operations use the default no-op implementation.

        PRECONDITION: the caller must verify the operation is an EndBarrier
        (i.e. it has a capacity shape region, region 1) before calling this
        method. The `Region &` parameter (reference, not pointer) assumes the
        region is valid.
      }],
      "void", "populateCapacityShapeRegion",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Region &":$capacityShapeRegion),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        // Default: no capacity shape region.
      }]
    >,
    InterfaceMethod<
      "Return the shape region (region 0).",
      "::mlir::Region &", "getShapeRegion", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getRegion(0);
      }]
    >,
    InterfaceMethod<
      "Return the capacity shape region (region 1) or nullptr if absent.",
      "::mlir::Region *", "getCapacityShapeRegion", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getNumRegions() > 1 ? &$_op->getRegion(1) : nullptr;
      }]
    >,
    InterfaceMethod<
      [{Return each result's dim values, grouped per result (handy for building
        `tensor.empty`). The shape region must be populated.}],
      "::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>",
      "getShapeRegionResultShapes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::hipsr::getShapeRegionResultShapes($_op.getShapeRegion());
      }]
    >,
    InterfaceMethod<
      [{Return each result's tensor type. Every extent is dynamic (an empty dim
        group gives a rank-0 tensor); constant dims fold back to static extents
        later via the tensor.empty canonicalizer. The shape region must be
        populated.}],
      "::llvm::SmallVector<::mlir::RankedTensorType>",
      "getShapeRegionResultTypes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::hipsr::getShapeRegionResultTypes($_op.getShapeRegion());
      }]
    >,
    InterfaceMethod<
      [{Like getShapeRegionResultShapes, but for the capacity region (region 1).
        Only valid on EndBarrier ops; the capacity region must be populated.}],
      "::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>",
      "getCapacityShapeRegionResultShapes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ::mlir::Region *region = $_op.getCapacityShapeRegion();
        assert(region && "op has no capacity shape region (region 1)");
        return ::mlir::hipsr::getShapeRegionResultShapes(*region);
      }]
    >,
    InterfaceMethod<
      [{Like getShapeRegionResultTypes, but for the capacity region (region 1).
        Only valid on EndBarrier ops; the capacity region must be populated.}],
      "::llvm::SmallVector<::mlir::RankedTensorType>",
      "getCapacityShapeRegionResultTypes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ::mlir::Region *region = $_op.getCapacityShapeRegion();
        assert(region && "op has no capacity shape region (region 1)");
        return ::mlir::hipsr::getShapeRegionResultTypes(*region);
      }]
    >,
  ];

  let verify = [{
    if (::mlir::failed(::mlir::hipsr::verifyShapeRegionStructure($_op)))
      return ::mlir::failure();
    // The trait's verifier does the scoping check; just require it be present.
    if (!$_op->hasTrait<
            ::mlir::hipsr::OpTrait::IsolatedFromAboveButAllowOperands>())
      return $_op->emitOpError(
          "ShapeRegionInterface requires the "
          "IsolatedFromAboveButAllowOperands trait");
    return ::mlir::success();
  }];
}

#endif // HIPSR_SHAPE_REGION_INTERFACE
