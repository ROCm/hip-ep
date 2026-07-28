/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct ExpandShapeArgs : ShapeRegionArgs<ExpandOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getInput() const { return in(0); }
  Value getRequestedShape() const { return in(1); }
};

struct CanonicalizeConstantShape : OpRewritePattern<ExpandOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getShape() || op.getShapeAttrAttr() ||
        !op.getShapeRegion().empty()) {
      return failure();
    }

    DenseIntElementsAttr denseShape;
    if (!matchPattern(op.getShape(), m_Constant(&denseShape))) {
      return failure();
    }

    auto shapeType = cast<ShapedType>(op.getShape().getType());
    if (denseShape.getNumElements() != shapeType.getDimSize(0)) {
      return failure();
    }

    SmallVector<int64_t> extents;
    extents.reserve(denseShape.getNumElements());
    for (APInt extent : denseShape.getValues<APInt>()) {
      extents.push_back(extent.getSExtValue());
    }

    rewriter.modifyOpInPlace(op, [&] {
      op.getShapeMutable().clear();
      op.setShapeAttrAttr(rewriter.getDenseI64ArrayAttr(extents));
    });
    return success();
  }
};
} // namespace

MutableOperandRange ExpandOp::getDpsInitsMutable() { return getInitMutable(); }

void ExpandOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<CanonicalizeConstantShape>(context);
}

LogicalResult ExpandOp::verify() {
  bool hasShapeOperand = static_cast<bool>(getShape());
  bool hasShapeAttr = static_cast<bool>(getShapeAttrAttr());
  if (hasShapeOperand && hasShapeAttr) {
    return emitOpError(
        "cannot have both shape operand and shape_attr attribute");
  }
  if (!hasShapeOperand && !hasShapeAttr) {
    return emitOpError(
        "must have either shape operand or shape_attr attribute");
  }

  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  int64_t shapeLength;
  if (hasShapeOperand) {
    auto shapeType = cast<ShapedType>(getShape().getType());
    if (shapeType.getRank() != 1) {
      return emitOpError("shape must be rank-1");
    }
    if (!shapeType.getElementType().isInteger(64)) {
      return emitOpError("shape element type must be i64");
    }
    if (shapeType.isDynamicDim(0)) {
      return emitOpError("shape length must be static");
    }
    shapeLength = shapeType.getDimSize(0);
  } else {
    shapeLength = static_cast<int64_t>(getShapeAttrAttr().asArrayRef().size());
  }

  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }

  int64_t expectedRank = std::max(inputType.getRank(), shapeLength);
  if (outputType.getRank() != expectedRank) {
    return emitOpError("output rank must equal max(input rank, shape length); "
                       "expected ")
           << expectedRank << ", got " << outputType.getRank();
  }
  return success();
}

bool ExpandOp::isStartBarrier() { return getShapeAttrAttr() == nullptr; }

void ExpandOp::populateShapeRegion(OpBuilder &builder, Block &shapeBlock) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = getLoc();
  MLIRContext *ctx = builder.getContext();
  ExpandShapeArgs args{shapeBlock};
  Value input = args.getInput();
  auto inputType = cast<ShapedType>(input.getType());

  Value inputShape = shape::ShapeOfOp::create(builder, loc, input);
  SmallVector<Value> requestedExtents;
  int64_t requestedRank;
  if (DenseI64ArrayAttr shapeAttr = getShapeAttrAttr()) {
    requestedRank = static_cast<int64_t>(shapeAttr.asArrayRef().size());
    requestedExtents.reserve(requestedRank);
    for (int64_t extent : shapeAttr.asArrayRef()) {
      requestedExtents.push_back(
          arith::ConstantIndexOp::create(builder, loc, extent));
    }
  } else {
    Value requestedShapeArg = args.getRequestedShape();
    auto requestedShapeType = cast<ShapedType>(requestedShapeArg.getType());
    requestedRank = requestedShapeType.getDimSize(0);
    requestedExtents.reserve(requestedRank);
    for (int64_t i : llvm::seq<int64_t>(0, requestedRank)) {
      Value index = arith::ConstantIndexOp::create(builder, loc, i);
      Value extent;
      if (isa<RankedTensorType>(requestedShapeArg.getType())) {
        extent = tensor::ExtractOp::create(builder, loc, requestedShapeArg,
                                           ValueRange{index});
      } else {
        extent = memref::LoadOp::create(builder, loc, requestedShapeArg,
                                        ValueRange{index});
      }
      requestedExtents.push_back(arith::IndexCastOp::create(
          builder, loc, builder.getIndexType(), extent));
    }
  }

  int64_t resultRank = std::max(inputType.getRank(), requestedRank);
  auto shapeType = shape::ShapeType::get(ctx);
  Value requestedShape = shape::FromExtentsOp::create(
      builder, loc, shapeType, ValueRange{requestedExtents});

  // ONNX Expand uses right-aligned multidirectional broadcasting.
  Value witness = shape::CstrBroadcastableOp::create(builder, loc, inputShape,
                                                     requestedShape);
  auto assuming = shape::AssumingOp::create(
      builder, loc, witness,
      [&](OpBuilder &b, Location) -> SmallVector<Value, 2> {
        Value broadcastShape = shape::BroadcastOp::create(
            b, loc, shapeType, inputShape, requestedShape, StringAttr{});
        SmallVector<Value, 2> resultExtents;
        resultExtents.reserve(resultRank);
        for (int64_t i : llvm::seq<int64_t>(0, resultRank)) {
          Value extent = shape::GetExtentOp::create(b, loc, broadcastShape, i);
          resultExtents.push_back(shape::SizeToIndexOp::create(b, loc, extent));
        }
        return resultExtents;
      });

  ShapeYieldOp::create(builder, loc,
                       ArrayRef<ValueRange>{assuming.getResults()},
                       TypeRange{inputType.getElementType()});
}

namespace {

constexpr const char *kWrapExpand = "wrap_expand";

struct ExpandLowering : ConvertOpToLLVMPattern<ExpandOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type hostPtrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type devicePtrType = LLVM::LLVMPointerType::get(ctx, 1);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!inputType || !outputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    auto createI64Const = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value one = createI64Const(1);
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int64_t rank = type.getRank();
      auto arrayType =
          LLVM::LLVMArrayType::get(i64Type, std::max<int64_t>(rank, 1));
      Value array = LLVM::AllocaOp::create(rewriter, loc, hostPtrType,
                                           arrayType, one, /*alignment=*/8);
      MemRefDescriptor desc(descriptor);
      for (int64_t i : llvm::seq<int64_t>(0, rank)) {
        Value dim = type.isDynamicDim(i) ? desc.size(rewriter, loc, i)
                                         : createI64Const(type.getDimSize(i));
        Value index = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                               rewriter.getI32IntegerAttr(i));
        Value elementPtr = LLVM::GEPOp::create(rewriter, loc, hostPtrType,
                                               i64Type, array, index);
        LLVM::StoreOp::create(rewriter, loc, dim, elementPtr);
      }
      return array;
    };

    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value shapePtr =
        op.getShape()
            ? extractContiguousMemRefPtr(adaptor.getShape(), rewriter, loc)
            : LLVM::ZeroOp::create(rewriter, loc, hostPtrType).getResult();
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getInit(), rewriter, loc);
    Value inputShape = emitShapeArray(inputType, adaptor.getInput());
    Value outputShape = emitShapeArray(outputType, adaptor.getInit());

    SmallVector<Type, 9> paramTypes = {
        hostPtrType,   // state
        devicePtrType, // input
        hostPtrType,   // requested shape
        devicePtrType, // output
        hostPtrType,   // input shape
        i64Type,       // input rank
        hostPtrType,   // output shape
        i64Type,       // output rank
        i64Type        // data type
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapExpand, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 9> args = {adaptor.getCtx(),
                                  inputPtr,
                                  shapePtr,
                                  outputPtr,
                                  inputShape,
                                  createI64Const(inputType.getRank()),
                                  outputShape,
                                  createI64Const(outputType.getRank()),
                                  createI64Const(dataType)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrExpandLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ExpandLowering>(converter);
}
