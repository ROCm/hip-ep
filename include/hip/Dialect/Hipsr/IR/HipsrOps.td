//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_OPS
#define HIPSR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "hip/Dialect/Hipsr/IR/HipsrDialect.td"
include "hip/Dialect/Hipsr/IR/HipsrTypes.td"
include "hip/Dialect/Hipsr/IR/HipsrTraits.td"
include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.td"
include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.td"
include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.td"

//===----------------------------------------------------------------------===//
// Base op class
//===----------------------------------------------------------------------===//

// Common base for all ops in the hipsr dialect. Downstream PRs define concrete
// ops (shape-region terminators, shaped compute ops, ...) on top of this.
class Hipsr_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hipsr_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Shaped DPS op base class
//===----------------------------------------------------------------------===//

// Base for shaped compute ops (MatMul, Add, ...). Every such op:
//   - carries an optional shape region (region 0), empty or one block ending
//     in `hipsr.shape_yield`;
//   - implements ShapeRegionInterface (each op provides populateShapeRegion);
//   - is IsolatedFromAbove, so the region reads the DPS `ins` as entry-block
//     args, not the op's operands;
//   - is destination-passing style: the `outs` init operands are the outputs.
//
// EndBarrier ops additionally carry a capacity region; they extend this base
// separately.
class Hipsr_DpsOp<string mnemonic, list<Trait> traits = []> :
    Hipsr_Op<mnemonic, !listconcat([
        DeclareOpInterfaceMethods<ShapeRegionInterface, ["populateShapeRegion"]>,
        DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                                  ["getDpsInitsMutable"]>,
        IsolatedFromAbove,
        SingleBlock
      ], traits)> {
  let regions = (region MaxSizedRegion<1>:$shape_region);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
}

#endif // HIPSR_OPS
