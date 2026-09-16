/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange ScatterNDOp::getDpsInitsMutable() {
  return getInitMutable();
}

LogicalResult ScatterNDOp::verify() {
  auto dataType = cast<ShapedType>(getData().getType());
  auto indicesType = cast<ShapedType>(getIndices().getType());
  auto updatesType = cast<ShapedType>(getUpdates().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  // The runtime reads the indices by their byte width, which `index` lacks.
  if (!indicesType.getElementType().isInteger()) {
    return emitOpError("indices element type must be an integer");
  }
  if (failed(
          verifyCompatibleShape(dataType.getShape(), outputType.getShape()))) {
    return emitOpError("output shape must match the data shape");
  }

  // The trailing extent says how many leading data axes a row addresses. The
  // updates shape follows from it, so it cannot wait for run time.
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  if (indicesShape.empty() || ShapedType::isDynamic(indicesShape.back())) {
    return emitOpError("indices must end in a static extent, which says how "
                       "many data axes a row addresses");
  }
  int64_t indexDepth = indicesShape.back();
  if (indexDepth < 1 || indexDepth > dataType.getRank()) {
    return emitOpError("the trailing indices extent must be in [1, data rank]; "
                       "data rank is ")
           << dataType.getRank() << ", got " << indexDepth;
  }

  SmallVector<int64_t> expectedUpdates = llvm::to_vector(
      llvm::concat<const int64_t>(indicesShape.drop_back(),
                                  dataType.getShape().drop_front(indexDepth)));
  if (failed(verifyCompatibleShape(expectedUpdates, updatesType.getShape()))) {
    return emitOpError("updates shape must be the leading indices extents "
                       "followed by the data extents no row addresses");
  }
  return success();
}

namespace mlir {
namespace hipsr {

LogicalResult populateScatterNDShapeRegion(OpBuilder &builder,
                                           Block &shapeBlock, ScatterNDOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  // The scatter only overwrites slices, so the output keeps the data's shape.
  Value dataShape = PlaceholderShapeRegionArgs{shapeBlock}.in(0);
  ShapeYieldOp::create(builder, op.getLoc(), ValueRange{dataShape});
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kWrapScatterND = "wrap_scatter_nd";

// The runtime's overwriting mode; the reducing ones have no hipsr operation.
constexpr int64_t kReductionNone = 0;

// The kernel reads the data and indices shapes into arrays this long.
constexpr int64_t kMaxRank = 8;

// `wrap_scatter_nd` copies the data into the output and then writes the
// updates, reading all four shapes from host arrays.
struct ScatterNDLowering : ConvertOpToLLVMPattern<ScatterNDOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ScatterNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto dataType = dyn_cast<MemRefType>(op.getData().getType());
    auto indicesType = dyn_cast<MemRefType>(op.getIndices().getType());
    auto updatesType = dyn_cast<MemRefType>(op.getUpdates().getType());
    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!dataType || !indicesType || !updatesType || !outputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }
    if (dataType.getRank() > kMaxRank || indicesType.getRank() > kMaxRank) {
      return rewriter.notifyMatchFailure(
          op, "data and indices rank must be at most 8");
    }
    int64_t elementType = getHipdnnDataType(dataType.getElementType());
    if (elementType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    Type i64Type = rewriter.getI64Type();
    auto shapeArray = [&](MemRefType type, Value descriptor) -> Value {
      return emitHostI64Array(
          extractShape(type, descriptor, rewriter, loc, i64Type), rewriter,
          loc);
    };
    // Named rather than built in the call below, where the order the arguments
    // are evaluated in is up to the compiler.
    Value dataShape = shapeArray(dataType, adaptor.getData());
    Value indicesShape = shapeArray(indicesType, adaptor.getIndices());
    Value updatesShape = shapeArray(updatesType, adaptor.getUpdates());
    Value outputShape = shapeArray(outputType, adaptor.getInit());
    // The call takes a device count that trims padded index rows. This op has
    // no such operand, so every row is written.
    Value noRowCount;

    using ScatterNDCall =
        RuntimeFunc<i32, hostPtr, devicePtr, devicePtr, devicePtr, devicePtr,
                    devicePtr, hostPtr, i64, hostPtr, i64, hostPtr, i64,
                    hostPtr, i64, i64, i64>;
    auto scatterFunc =
        ScatterNDCall::lookupOrCreateFn(rewriter, loc, module, kWrapScatterND);
    if (failed(scatterFunc)) {
      return failure();
    }
    if (failed(scatterFunc->call(
            adaptor.getCtx(), adaptor.getData(), adaptor.getIndices(),
            adaptor.getUpdates(), adaptor.getInit(), noRowCount, dataShape,
            dataType.getRank(), indicesShape, indicesType.getRank(),
            updatesShape, updatesType.getRank(), outputShape,
            outputType.getRank(), kReductionNone, elementType))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrScatterNDLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ScatterNDLowering>(converter);
}
