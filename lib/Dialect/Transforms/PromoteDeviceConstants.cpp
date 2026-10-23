/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PromoteDeviceConstants.cpp - Device copies of kernel-read constants ===//
//
// A tensor `arith.constant` bufferizes to a `memref.global` with an initial
// value, which lowers to data in the compiled module: host memory. A hip op
// handed that buffer passes a host pointer to its kernel, which faults. Only a
// `hip.constant` reaches device memory, through hip-externalize-constants and
// the runtime's constant upload.
//
// The arith form is still the right one for the host: shape folds, reification
// and `m_Constant` matching all read it, and none of them sees through a
// `hip.constant`. So the constant is split per use rather than replaced. Each
// use read from device memory moves to one `hip.constant` with the same value;
// every other use keeps the `arith.constant`. Device reads are a hip op's
// operands, a DPS init included, and tensor.insert_slice's source and
// destination: bufferization copies those with a device-to-device memcpy.
//
// Views do not move data, so a hip op reading a view of the constant reads the
// constant. Each view on the way to a device read is cloned onto the device
// constant, and only that use moves to the clone; host uses of the same view
// stay where they are.
//
// Before:
//
//   %c = arith.constant dense<1> : tensor<8xi64>
//   %v = tensor.expand_shape %c [[0, 1]] output_shape [1, 8]
//          : tensor<8xi64> into tensor<1x8xi64>
//   %g = hip.gather(%ctx) ins(%table, %v : ...) outs(%e : ...)
//   %x = tensor.extract %c[%i] : tensor<8xi64>
//
// After:
//
//   %c  = arith.constant dense<1> : tensor<8xi64>
//   %d  = hip.constant {value = dense<1> : tensor<8xi64>} : tensor<8xi64>
//   %v  = tensor.expand_shape %c ...           // no users left; DCE'd later
//   %dv = tensor.expand_shape %d [[0, 1]] output_shape [1, 8]
//          : tensor<8xi64> into tensor<1x8xi64>
//   %g = hip.gather(%ctx) ins(%table, %dv : ...) outs(%e : ...)
//   %x = tensor.extract %c[%i] : tensor<8xi64>
//
// The pass runs last before hip-externalize-constants. Everything that reads
// the arith form has run by then: hip-infer-shapes' reification, and the
// folding in hip-fusion-transform's rewrite driver, which turns a view of an
// arith.constant into a new constant (a view of a hip.constant cannot fold, so
// promoting earlier would leave it a view). The readback folds, which run in
// the canonicalizer after externalization, are applied by the pass itself just
// before promoting, for the same reason. Constants that reach the device any
// other way -- through a call or a loop/if capture, or as a graph result --
// are not handled here.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "hip-promote-device-constants"

STATISTIC(NumConstantsPromoted,
          "Number of tensor arith.constant ops given a hip.constant");
STATISTIC(NumUsesRedirected,
          "Number of device reads moved to a device constant");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PROMOTEDEVICECONSTANTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// `use` is the data operand of an op whose result aliases it, so whoever
/// reads the result reads the constant.
bool isViewSource(OpOperand &use) {
  return isa<tensor::CastOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::ReshapeOp>(use.getOwner()) &&
         use.getOperandNumber() == 0;
}

/// Whether `cst` can become an inline hip.constant: a DenseElementsAttr over a
/// static shape, of an element type the carrier supports. Anything else is
/// left as it is rather than turned into an op that fails verification.
bool isPromotable(arith::ConstantOp cst) {
  auto type = dyn_cast<RankedTensorType>(cst.getType());
  return type && type.hasStaticShape() &&
         isa<DenseElementsAttr>(cst.getValue()) &&
         ConstantOp::isSupportedElementType(type.getElementType());
}

/// `use` reads the value from device memory:
///   * any operand of a hip op. A DPS init counts too: bufferization copies a
///     constant into a fresh buffer before the op writes it, and that copy
///     lowers to a device-to-device memcpy.
///   * the source or destination of tensor.insert_slice, which bufferizes to
///     the same memcpy (into the destination's buffer, or out of a constant
///     destination into a writable one).
bool isDeviceRead(OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<HipDialect>(user->getDialect()))
    return true;
  if (auto insert = dyn_cast<tensor::InsertSliceOp>(user))
    return &use == &insert.getSourceMutable() ||
           &use == &insert.getDestMutable();
  return false;
}

class DeviceConstantPromoter {
public:
  /// Move every device read of `value` -- the arith.constant or a view of it
  /// -- directly or through further views, to its device constant.
  void redirect(Value value) {
    for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
      if (isViewSource(use)) {
        redirect(use.getOwner()->getResult(0));
        continue;
      }
      if (!isDeviceRead(use))
        continue;
      use.set(getDeviceConstant(value));
      ++NumUsesRedirected;
    }
  }

private:
  /// The device constant for `value`, the arith.constant or a view of it: the
  /// same data, backed by a hip.constant. For the arith.constant that is a new
  /// hip.constant; for a view, a clone of the view reading its source's device
  /// constant. Created once, right after `value`, so it dominates every use of
  /// `value`.
  Value getDeviceConstant(Value value) {
    if (Value deviceConstant = deviceConstantOf.lookup(value))
      return deviceConstant;

    Operation *def = value.getDefiningOp();
    OpBuilder builder(def->getContext());
    builder.setInsertionPointAfter(def);
    Value deviceConstant;
    if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
      deviceConstant = ConstantOp::create(
          builder, cst.getLoc(), cast<RankedTensorType>(cst.getType()),
          cast<DenseElementsAttr>(cst.getValue()));
      ++NumConstantsPromoted;
    } else {
      IRMapping mapping;
      mapping.map(def->getOperand(0), getDeviceConstant(def->getOperand(0)));
      deviceConstant = builder.clone(*def, mapping)->getResult(0);
    }
    deviceConstantOf[value] = deviceConstant;
    return deviceConstant;
  }

  llvm::DenseMap<Value, Value> deviceConstantOf;
};

struct PromoteDeviceConstantsPass
    : public impl::PromoteDeviceConstantsPassBase<PromoteDeviceConstantsPass> {
  void runOnOperation() override {
    // A readback of a hip.cast of a constant folds to a host constant, through
    // m_Constant on the cast's input. Once that input is a hip.constant the
    // fold cannot match, and the program pays for the cast, a sync and a D2H
    // copy at runtime. So apply the readback folds first; a cast left with no
    // reader is erased and no longer counts as a device read.
    RewritePatternSet readbackFolds(&getContext());
    ReadbackScalarOp::getCanonicalizationPatterns(readbackFolds, &getContext());
    // Only these patterns: generic folding would also rewrite unrelated IR.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(readbackFolds),
                                     config)))
      return signalPassFailure();

    SmallVector<arith::ConstantOp> constants;
    getOperation().walk([&](arith::ConstantOp cst) {
      if (isPromotable(cst))
        constants.push_back(cst);
    });

    DeviceConstantPromoter promoter;
    for (arith::ConstantOp cst : constants)
      promoter.redirect(cst.getResult());
  }
};

} // namespace
} // namespace hip
} // namespace mlir
