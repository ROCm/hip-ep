//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ShapeRegionInterface. Other hipsr interfaces (barrier markers, ...) live in
// their own per-interface files.
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

    An implementing op carries a shape region (region 0) whose body computes
    the op's output dimensions and ends with a `hipsr.shape_yield`. The
    populate* methods emit that region; the get* accessors retrieve it.

    REQUIREMENT: an op implementing this interface MUST also add the
    `SingleBlockImplicitTerminator<"ShapeYieldOp">` trait, e.g.

      def Hipsr_MatMulOp : Hipsr_Op<"matmul", [
          ShapeRegionInterface,
          SingleBlockImplicitTerminator<"ShapeYieldOp">
      ]> { ... }

    The trait provides automatic terminator insertion; MLIR interfaces cannot
    auto-attach traits, so the interface verifier defensively checks the shape
    region's single-block/terminator structure (verifyShapeRegionStructure)
    before enforcing the scoping rules (verifyShapeRegionScoping), giving a
    clear error instead of a downstream crash if the trait is forgotten.

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
  ];

  let verify = [{
    if (::mlir::failed(::mlir::hipsr::verifyShapeRegionStructure($_op)))
      return ::mlir::failure();
    return ::mlir::hipsr::verifyShapeRegionScoping($_op);
  }];
}

#endif // HIPSR_SHAPE_REGION_INTERFACE
