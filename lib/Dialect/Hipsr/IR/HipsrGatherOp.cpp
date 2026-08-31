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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct GatherPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getData() const { return in(0); }
  Value getIndices() const { return in(1); }
};
} // namespace

namespace mlir {
namespace hipsr {

// The data shape with `axis` replaced by the whole indices shape. The ONNX
// conversion builds its result type from the same rule.
SmallVector<int64_t> inferGatherResultShape(ArrayRef<int64_t> dataShape,
                                            ArrayRef<int64_t> indicesShape,
                                            int64_t axis) {
  return llvm::to_vector(
      llvm::concat<const int64_t>(dataShape.take_front(axis), indicesShape,
                                  dataShape.drop_front(axis + 1)));
}

// The same rule on symbolic shapes, one extent per output dimension.
LogicalResult populateGatherShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                        GatherOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = op.getLoc();
  GatherPlaceholderShapeArgs args{shapeBlock};
  int64_t dataRank = cast<ShapedType>(op.getData().getType()).getRank();
  int64_t indicesRank = cast<ShapedType>(op.getIndices().getType()).getRank();
  int64_t axis = op.getAxis();

  auto extent = [&](Value shape, int64_t index) -> Value {
    return shape::GetExtentOp::create(builder, loc, shape, index);
  };
  auto appendExtents = [&](SmallVectorImpl<Value> &extents, Value shape,
                           int64_t begin, int64_t end) {
    for (int64_t index : llvm::seq(begin, end)) {
      extents.push_back(extent(shape, index));
    }
  };

  SmallVector<Value> outputExtents;
  outputExtents.reserve(dataRank - 1 + indicesRank);
  appendExtents(outputExtents, args.getData(), 0, axis);
  appendExtents(outputExtents, args.getIndices(), 0, indicesRank);
  appendExtents(outputExtents, args.getData(), axis + 1, dataRank);

  Value outputShape = createExtentTensor(builder, loc, outputExtents);
  ShapeYieldOp::create(builder, loc, ValueRange{outputShape});
  return success();
}

} // namespace hipsr
} // namespace mlir

MutableOperandRange GatherOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult GatherOp::verify() {
  auto dataType = cast<ShapedType>(getData().getType());
  auto indicesType = cast<ShapedType>(getIndices().getType());
  int64_t axis = getAxis();

  if (axis < 0 || axis >= dataType.getRank()) {
    return emitOpError("axis must be in [0, data rank); data rank is ")
           << dataType.getRank() << ", got " << axis;
  }
  // The runtime steps the indices by their byte width, which `index` lacks.
  if (!indicesType.getElementType().isInteger()) {
    return emitOpError("indices element type must be an integer");
  }

  SmallVector<int64_t> expectedShape =
      inferGatherResultShape(dataType.getShape(), indicesType.getShape(), axis);
  auto outputType = cast<ShapedType>(getInit().getType());
  if (failed(verifyCompatibleShape(expectedShape, outputType.getShape()))) {
    return emitOpError("output shape must be the data shape with axis replaced "
                       "by the indices shape");
  }
  return success();
}

namespace {

constexpr const char *kWrapGather = "wrap_gather";

// `wrap_gather` takes element counts rather than shapes. It reconstructs the
// data layout as `[outer, axis_size, inner]` from `data_num_elements`,
// `axis_size` and `inner_size`, and reads both operands by their byte width.
struct GatherLowering : ConvertOpToLLVMPattern<GatherOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto dataType = dyn_cast<MemRefType>(op.getData().getType());
    auto indicesType = dyn_cast<MemRefType>(op.getIndices().getType());
    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!dataType || !indicesType || !outputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }
    // The runtime moves whole elements and steps the indices by their width.
    auto wholeBytes = [](Type elementType) {
      return elementType.isIntOrFloat() &&
             elementType.getIntOrFloatBitWidth() % 8 == 0;
    };
    if (!wholeBytes(dataType.getElementType()) ||
        !wholeBytes(indicesType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "element types must be whole bytes");
    }

    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    // Seeded with the first dimension so a shape needs one multiply less.
    auto numElements = [&](ValueRange dims) -> Value {
      if (dims.empty()) {
        return createI64Const(1);
      }
      Value product = dims.front();
      for (Value dim : dims.drop_front()) {
        product = LLVM::MulOp::create(rewriter, loc, product, dim);
      }
      return product;
    };

    SmallVector<Value> dataDims =
        extractShape(dataType, adaptor.getData(), rewriter, loc, i64Type);
    SmallVector<Value> indicesDims =
        extractShape(indicesType, adaptor.getIndices(), rewriter, loc, i64Type);
    SmallVector<Value> outputDims =
        extractShape(outputType, adaptor.getInit(), rewriter, loc, i64Type);

    int64_t axis = op.getAxis();
    Value axisSize = dataDims[axis];
    Value innerSize = numElements(ValueRange(dataDims).drop_front(axis + 1));
    // Named rather than built in the call below, where the order the arguments
    // are evaluated in is up to the compiler.
    Value dataCount = numElements(dataDims);
    Value outputCount = numElements(outputDims);
    Value indicesCount = numElements(indicesDims);
    int64_t elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t indicesElementSizeBytes =
        indicesType.getElementType().getIntOrFloatBitWidth() / 8;

    using GatherCall =
        RuntimeFunc<i32, hostPtr, devicePtr, devicePtr, devicePtr, i64, i64,
                    i64, i64, i64, i64, i64, i64>;
    auto gatherFunc =
        GatherCall::lookupOrCreateFn(rewriter, loc, module, kWrapGather);
    if (failed(gatherFunc)) {
      return failure();
    }
    if (failed(gatherFunc->call(
            adaptor.getCtx(), adaptor.getData(), adaptor.getIndices(),
            adaptor.getInit(), axis, dataCount, indicesCount, outputCount,
            axisSize, innerSize, elementSizeBytes, indicesElementSizeBytes))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrGatherLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GatherLowering>(converter);
}
