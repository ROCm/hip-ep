/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
// Supplies the builtin IsIsolatedFromAbove trait the interface verifier checks.
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace hipsr {

/// Verifies region 0: empty, or a single block whose args match
/// `getShapeRegionArgOperands` and that ends in `hipsr.shape_yield`.
::mlir::LogicalResult verifyShapeRegionStructure(::mlir::Operation *op);

class ShapeRegionInterface;

/// The op operands that become the shape region's entry-block arguments, in
/// order. The isolated region reads these as block args. Keyed on the op's
/// barrier category:
///   Regular        -> data ins            (ctx dropped: shape is input-driven)
///   StartBarrier   -> ctx + data ins      (reads input data at runtime)
///   EndBarrier     -> data ins + outs     (shape comes from output data)
::llvm::SmallVector<::mlir::Value>
getShapeRegionArgOperands(ShapeRegionInterface op);

/// Region 0, the shape region; present on every ShapeRegionInterface op.
::mlir::Region &getShapeRegion(ShapeRegionInterface op);

/// Region 1, present only on EndBarrier ops; a fatal error on any other op.
::mlir::Region &getCapacityShapeRegion(ShapeRegionInterface op);

/// Each result's dim values, grouped per result. Region must be populated.
::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
getShapeRegionResultShapes(ShapeRegionInterface op);

/// Each result's tensor type, all extents dynamic. Region must be populated.
::llvm::SmallVector<::mlir::RankedTensorType>
getShapeRegionResultTypes(ShapeRegionInterface op);

/// The two above over the capacity region (region 1).
::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
getCapacityShapeRegionResultShapes(ShapeRegionInterface op);

::llvm::SmallVector<::mlir::RankedTensorType>
getCapacityShapeRegionResultTypes(ShapeRegionInterface op);

/// Bounds-checked accessor over the getShapeRegionArgOperands layout above;
/// ctx()/in(i)/out(j) index the block args, gated by OpTy's barrier category.
template <typename OpTy> struct ShapeRegionArgs {
  explicit ShapeRegionArgs(Block &b) : block(b) {}

public:
  static constexpr bool kIsStartBarrier =
      OpTy::template hasTrait<StartBarrierInterface::Trait>();
  static constexpr bool kIsEndBarrier =
      OpTy::template hasTrait<EndBarrierInterface::Trait>();

  Value ctx() const {
    static_assert(kIsStartBarrier,
                  "ctx() is only valid on start-barrier shape regions");
    return arg(0);
  }

  Value in(unsigned i) const { return arg(numCtxArgs() + i); }

  Value out(unsigned j) const {
    static_assert(kIsEndBarrier,
                  "out() is only valid on end-barrier shape regions");
    return arg(numDataInputs() + j);
  }

protected:
  Block &block;

  static constexpr unsigned numCtxArgs() { return kIsStartBarrier ? 1u : 0u; }

  unsigned numDataInputs() const {
    auto dps = cast<DestinationStyleOpInterface>(block.getParentOp());
    unsigned numIns = dps.getDpsInputs().size();
    return numIns == 0 ? 0 : numIns - 1; // drop ctx
  }

  Value arg(unsigned index) const {
    if (index >= block.getNumArguments()) {
      std::string msg;
      llvm::raw_string_ostream(msg)
          << OpTy::getOperationName() << " shape region is missing block arg "
          << index << " (block has " << block.getNumArguments() << ")";
      llvm::report_fatal_error(llvm::StringRef(msg));
    }
    return block.getArgument(index);
  }
};

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
