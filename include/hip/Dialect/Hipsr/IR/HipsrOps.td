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
//   - carries an optional shape region (region 0): zero blocks, or one block
//     computing the output shape and ending in `hipsr.shape_yield`.
//     MaxSizedRegion<1> allows the 0-or-1-block range; SingleBlock caps a
//     non-empty region at one block (SingleBlockImplicitTerminator is avoided
//     because it would inject a terminator into the empty form);
//   - implements ShapeRegionInterface (each op provides populateShapeRegion);
//   - carries IsolatedFromAbove, so the shape region reads the DPS `ins` as
//     entry-block args rather than the op's operands;
//   - uses destination-passing style (DestinationStyleOpInterface): the `outs`
//     init operands are the output buffers.
//
// Regular and StartBarrier ops use this one shape region; EndBarrier ops
// (which also need a capacity region) extend it separately.
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
