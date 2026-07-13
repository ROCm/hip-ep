//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_INTERFACES
#define HIPSR_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ShapeRegionInterface
//===----------------------------------------------------------------------===//

def ShapeRegionInterface : OpInterface<"ShapeRegionInterface"> {
  let cppNamespace = "::mlir::hipsr";

  let description = [{
    Single source of truth for an op's output-shape computation.

    An implementing op carries a shape region (region 0) whose body computes
    the op's output dimensions and ends with a `hipsr.shape_yield`. The
    populate* methods emit that region; the get* accessors retrieve it.

    Operation categories (barrier markers are separate interfaces):
    - Regular: no block arguments, one shape, depends on input shapes.
    - StartBarrier: no block arguments, one shape, depends on input data.
    - EndBarrier: has block arguments, two shapes (capacity + real), depends
      on output data.
  }];

  let methods = [
    InterfaceMethod<
      "Populate the shape region with the ops that compute the output shape.",
      "void", "populateShapeRegion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Region &":$shapeRegion)
    >,
    InterfaceMethod<
      "Populate the capacity shape region (only EndBarrier ops override this).",
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
    return ::mlir::hipsr::verifyShapeRegionScoping($_op);
  }];
}

#endif // HIPSR_INTERFACES
