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

    REQUIREMENT: an op implementing this interface MUST also add two traits
    (MLIR interfaces cannot auto-attach traits), e.g.

      def Hipsr_MatMulOp : Hipsr_Op<"matmul", [
          ShapeRegionInterface,
          IsolatedFromAboveButAllowOperands,
          SingleBlockExplicitTerminator<"ShapeYieldOp">
      ]> { ... }

    `SingleBlockExplicitTerminator<"ShapeYieldOp">` requires each region to end
    with a `hipsr.shape_yield` (never added automatically, since it carries the
    computed shape). `IsolatedFromAboveButAllowOperands` limits what the shape
    region may read. The interface verifier only checks that both traits are
    present so a missing one gives a clear error; each trait's own verifier does
    the real check.

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
    // Each trait's own verifier does the real check (explicit shape_yield,
    // scoping); here we only require both traits so a missing one is a clear
    // error instead of a silent skip.
    if (!$_op->hasTrait<::mlir::hipsr::OpTrait::SingleBlockExplicitTerminator<
            ::mlir::hipsr::ShapeYieldOp>::Impl>())
      return $_op->emitOpError(
          "ShapeRegionInterface requires the "
          "SingleBlockExplicitTerminator<ShapeYieldOp> trait");
    if (!$_op->hasTrait<
            ::mlir::hipsr::OpTrait::IsolatedFromAboveButAllowOperands>())
      return $_op->emitOpError(
          "ShapeRegionInterface requires the "
          "IsolatedFromAboveButAllowOperands trait");
    return ::mlir::success();
  }];
}

#endif // HIPSR_SHAPE_REGION_INTERFACE
