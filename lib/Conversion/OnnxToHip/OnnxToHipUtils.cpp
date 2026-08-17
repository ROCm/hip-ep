/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipUtils.cpp - ONNX-specific conversion helpers ------------===//

#include "OnnxToHipUtils.h"

namespace mlir::hip {
namespace {

bool areCompatibleStaticExtents(int64_t lhs, int64_t rhs) {
  return mlir::ShapedType::isDynamic(lhs) || mlir::ShapedType::isDynamic(rhs) ||
         lhs == rhs;
}

mlir::Value indexDivByConstant(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value extent,
                               int64_t divisor) {
  mlir::Value divisorValue =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, divisor);
  return mlir::arith::DivUIOp::create(rewriter, loc, extent, divisorValue);
}

} // namespace

DenseElementsAttr getCompileTimeConstantTensor(Value value) {
  if (DenseElementsAttr dense = matchHipCompileTimeConstantTensor(value))
    return dense;
  if (!value)
    return {};

  Operation *defOp = value.getDefiningOp();
  // Semantic pre-lowering rewrites intentionally run before hip.constant
  // carrier creation. Recognize only the exact generic ONNX constant form.
  if (!defOp || defOp->getName().getStringRef() != "onnx.Constant" ||
      defOp->getNumResults() != 1 || defOp->getResult(0) != value)
    return {};
  auto dense = defOp->getAttrOfType<DenseElementsAttr>("value");
  return dense && dense.getType() == value.getType() ? dense
                                                     : DenseElementsAttr();
}

bool extractConstantIntTensor(Value value, llvm::SmallVectorImpl<int64_t> &out,
                              std::optional<int64_t> expectedRank) {
  return parseDenseIntElements(getCompileTimeConstantTensor(value), out,
                               expectedRank);
}

bool extractConstantIntVector(Value value,
                              llvm::SmallVectorImpl<int64_t> &out) {
  return extractConstantIntTensor(value, out, /*expectedRank=*/1);
}

bool isResultTypeCompatibleWithPayloadShape(
    mlir::RankedTensorType resultType, llvm::ArrayRef<int64_t> inferredShape) {
  return isResultTypeCompatibleWithInferredShape(resultType, inferredShape);
}

static std::optional<llvm::ArrayRef<int64_t>>
resolveReductionAxes(Operation *op, Value data, int64_t noopWithEmptyAxes,
                     llvm::SmallVectorImpl<int64_t> &storage) {
  storage.clear();
  bool hasAxesOperand =
      op->getNumOperands() > 1 && !isa<NoneType>(op->getOperand(1).getType());
  if (hasAxesOperand) {
    auto axesType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!axesType || (axesType.getRank() != 0 && axesType.getRank() != 1) ||
        !extractConstantIntTensor(op->getOperand(1), storage))
      return std::nullopt;
  } else if (auto axesAttr = op->getAttrOfType<ArrayAttr>("axes")) {
    for (Attribute entry : axesAttr)
      storage.push_back(cast<IntegerAttr>(entry).getValue().getSExtValue());
  }

  if (storage.empty() && noopWithEmptyAxes == 0) {
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    if (!dataType)
      return std::nullopt;
    llvm::append_range(storage, llvm::seq<int64_t>(0, dataType.getRank()));
  }
  return llvm::ArrayRef<int64_t>(storage);
}

mlir::FailureOr<GqaSequenceExtents>
resolveGqaSequenceExtents(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Operation *op, mlir::Value totalSeqLen,
                          mlir::Value pastKey, mlir::Value pastValue,
                          mlir::RankedTensorType presentKeyType,
                          mlir::RankedTensorType presentValueType,
                          mlir::RankedTensorType outputQkType) {
  if (static_cast<bool>(pastKey) != static_cast<bool>(pastValue)) {
    (void)rewriter.notifyMatchFailure(
        op, "past_key and past_value must both be present or both be absent");
    return mlir::failure();
  }
  if (!presentKeyType || !presentValueType || presentKeyType.getRank() != 4 ||
      presentValueType.getRank() != 4) {
    (void)rewriter.notifyMatchFailure(
        op, "present_key and present_value must be rank-4 BNSH tensors");
    return mlir::failure();
  }
  if (outputQkType && outputQkType.getRank() != 4) {
    (void)rewriter.notifyMatchFailure(op, "output_qk must be a rank-4 tensor");
    return mlir::failure();
  }

  if (pastKey) {
    auto pastKeyType =
        mlir::dyn_cast<mlir::RankedTensorType>(pastKey.getType());
    auto pastValueType =
        mlir::dyn_cast<mlir::RankedTensorType>(pastValue.getType());
    if (!pastKeyType || !pastValueType || pastKeyType.getRank() != 4 ||
        pastValueType.getRank() != 4) {
      (void)rewriter.notifyMatchFailure(
          op, "past_key and past_value must be rank-4 BNSH tensors");
      return mlir::failure();
    }
  }

  GqaSequenceExtents extents;
  bool needsLogical = presentKeyType.isDynamicDim(2) ||
                      presentValueType.isDynamicDim(2) ||
                      (outputQkType && outputQkType.isDynamicDim(3));
  if (!needsLogical)
    return extents;

  auto totalSeqLenType =
      totalSeqLen
          ? mlir::dyn_cast<mlir::RankedTensorType>(totalSeqLen.getType())
          : mlir::RankedTensorType();
  if (!totalSeqLenType || totalSeqLenType.getRank() != 0 ||
      !mlir::isa<mlir::IntegerType, mlir::IndexType>(
          totalSeqLenType.getElementType())) {
    (void)rewriter.notifyMatchFailure(
        op, "total_sequence_length must be a scalar integer tensor");
    return mlir::failure();
  }

  extents.logical =
      readbackScalarToIndexOrExtract(rewriter, loc, op, totalSeqLen);
  auto resolvePresent = [&](mlir::RankedTensorType resultType,
                            mlir::Value past) -> mlir::Value {
    if (!resultType.isDynamicDim(2))
      return {};
    if (!past)
      return extents.logical;
    mlir::Value pastExtent =
        mlir::tensor::DimOp::create(rewriter, loc, past, 2);
    return mlir::arith::MaxUIOp::create(rewriter, loc, pastExtent,
                                        extents.logical);
  };
  extents.presentKey = resolvePresent(presentKeyType, pastKey);
  extents.presentValue = resolvePresent(presentValueType, pastValue);
  return extents;
}

mlir::FailureOr<mlir::Value>
createGqaPresentEmpty(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::RankedTensorType resultType, mlir::Value query,
                      mlir::Value key, mlir::Value totalSeqExtent,
                      int64_t numHeads, int64_t kvNumHeads) {
  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  auto keyType = key ? mlir::dyn_cast<mlir::RankedTensorType>(key.getType())
                     : mlir::RankedTensorType();
  if (!queryType || queryType.getRank() != 3 || resultType.getRank() != 4 ||
      (totalSeqExtent && !totalSeqExtent.getType().isIndex()) ||
      (resultType.isDynamicDim(2) && !totalSeqExtent) || numHeads <= 0 ||
      kvNumHeads <= 0 || (key && !keyType) ||
      (keyType && keyType.getRank() != 3 && keyType.getRank() != 4))
    return mlir::failure();

  if (!areCompatibleStaticExtents(resultType.getDimSize(0),
                                  queryType.getDimSize(0)) ||
      (resultType.getDimSize(1) != mlir::ShapedType::kDynamic &&
       resultType.getDimSize(1) != kvNumHeads) ||
      (!keyType && resultType.isDynamicDim(3)))
    return mlir::failure();
  if (resultType.isDynamicDim(3) && keyType.getRank() == 3) {
    int64_t hidden = keyType.getDimSize(2);
    if (!mlir::ShapedType::isDynamic(hidden) && hidden % kvNumHeads != 0)
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> dynamicSizes;
  dynamicSizes.reserve(resultType.getNumDynamicDims());
  for (int64_t dim : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(dim))
      continue;
    switch (dim) {
    case 0:
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, query, 0));
      break;
    case 1:
      dynamicSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, kvNumHeads));
      break;
    case 2:
      dynamicSizes.push_back(totalSeqExtent);
      break;
    case 3:
      if (keyType && keyType.getRank() == 4) {
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, key, 3));
      } else {
        mlir::Value source = key;
        mlir::Value hidden =
            mlir::tensor::DimOp::create(rewriter, loc, source, 2);
        dynamicSizes.push_back(
            indexDivByConstant(rewriter, loc, hidden, kvNumHeads));
      }
      break;
    default:
      llvm_unreachable("rank-4 GQA present shape has an invalid dimension");
    }
  }

  return mlir::Value(
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynamicSizes));
}

mlir::FailureOr<mlir::Value>
createGqaQkEmpty(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::RankedTensorType resultType, mlir::Value query,
                 mlir::Value totalSeqExtent, int64_t numHeads) {
  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  if (!queryType || queryType.getRank() != 3 || resultType.getRank() != 4 ||
      (totalSeqExtent && !totalSeqExtent.getType().isIndex()) ||
      (resultType.isDynamicDim(3) && !totalSeqExtent) || numHeads <= 0 ||
      !areCompatibleStaticExtents(resultType.getDimSize(0),
                                  queryType.getDimSize(0)) ||
      (resultType.getDimSize(1) != mlir::ShapedType::kDynamic &&
       resultType.getDimSize(1) != numHeads) ||
      !areCompatibleStaticExtents(resultType.getDimSize(2),
                                  queryType.getDimSize(1)))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> dynamicSizes;
  dynamicSizes.reserve(resultType.getNumDynamicDims());
  for (int64_t dim : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(dim))
      continue;
    switch (dim) {
    case 0:
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, query, 0));
      break;
    case 1:
      dynamicSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, numHeads));
      break;
    case 2:
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, query, 1));
      break;
    case 3:
      dynamicSizes.push_back(totalSeqExtent);
      break;
    default:
      llvm_unreachable("rank-4 GQA QK shape has an invalid dimension");
    }
  }

  return mlir::Value(
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynamicSizes));
}

static mlir::FailureOr<mlir::RankedTensorType>
inferReduceResultType(mlir::Operation *op, mlir::Value data,
                      llvm::ArrayRef<int64_t> reducedAxes, int64_t keepdims) {
  auto ranked = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(data.getType());
  if (!inputType)
    return failure();
  FailureOr<llvm::SmallVector<int64_t>> outShape =
      inferReductionShape(inputType.getShape(), reducedAxes, keepdims);
  if (failed(outShape))
    return failure();
  if (ranked) {
    if (!isResultTypeCompatibleWithInferredShape(ranked, *outShape))
      return failure();
    return ranked;
  }
  return RankedTensorType::get(*outShape, inputType.getElementType());
}

namespace {

FailureOr<Value> createReductionEmptyTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType resultType,
                                            Value data,
                                            llvm::ArrayRef<int64_t> reducedAxes,
                                            int64_t keepdims) {
  FailureOr<llvm::SmallVector<OpFoldResult>> shape =
      reifyReductionResultShape(builder, loc, data, reducedAxes, keepdims);
  if (failed(shape))
    return failure();
  return createEmptyTensorFromReifiedShape(builder, loc, resultType, *shape);
}

Value materializeReductionAxes(OpBuilder &builder, Location loc,
                               llvm::ArrayRef<int64_t> resolvedAxes) {
  auto axesType = RankedTensorType::get(
      {static_cast<int64_t>(resolvedAxes.size())}, builder.getI64Type());
  auto axesAttr = DenseIntElementsAttr::get(axesType, resolvedAxes);
  return arith::ConstantOp::create(builder, loc, axesType, axesAttr);
}

/// Shared ONNX reduction conversion skeleton. All supported reductions differ
/// only by their HIP op type; axes resolution, destination construction, and
/// attributes stay private to this translation unit.
template <typename HipOpTy>
class OnnxReductionToHip final : public RewritePattern {
public:
  OnnxReductionToHip(MLIRContext *ctx, llvm::StringRef onnxOpName)
      : RewritePattern(onnxOpName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduction expects at least one input and one output");

    auto context = getContextArg(op, rewriter);
    if (failed(context))
      return failure();

    Value data = op->getOperand(0);
    int64_t noopWithEmptyAxes = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("noop_with_empty_axes"))
      noopWithEmptyAxes = attr.getSInt();
    int64_t keepdims = 1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("keepdims"))
      keepdims = attr.getSInt();
    if (keepdims != 0 && keepdims != 1)
      return op->emitError("keepdims must be 0 or 1");
    if (noopWithEmptyAxes != 0 && noopWithEmptyAxes != 1)
      return op->emitError("noop_with_empty_axes must be 0 or 1");

    llvm::SmallVector<int64_t> axesStorage;
    std::optional<llvm::ArrayRef<int64_t>> reducedAxes =
        resolveReductionAxes(op, data, noopWithEmptyAxes, axesStorage);
    if (!reducedAxes)
      return op->emitError("reduction axes must be known at compile time");
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    if (!dataType)
      return op->emitError("reduction data must be a ranked tensor");
    auto declaredResultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (declaredResultType &&
        declaredResultType.getElementType() != dataType.getElementType())
      return op->emitError(
          "reduction data and result must have the same element type");
    llvm::StringRef hipOpName = HipOpTy::getOperationName();
    if (!isSupportedReductionElementType(hipOpName, dataType.getElementType()))
      return op->emitError()
             << "unsupported reduction element type "
             << dataType.getElementType() << "; supported types: "
             << getSupportedReductionElementTypes(hipOpName);
    auto normalizedAxes =
        normalizeReductionAxes(dataType.getRank(), *reducedAxes);
    if (failed(normalizedAxes))
      return op->emitError(
          "reduction axes must be unique, in range, and form one contiguous "
          "span");
    axesStorage.assign(normalizedAxes->begin(), normalizedAxes->end());
    llvm::ArrayRef<int64_t> axesView(axesStorage);

    auto resultType = inferReduceResultType(op, data, axesView, keepdims);
    if (failed(resultType))
      return op->emitError(
          "result type is incompatible with the reduction data shape and axes");

    Location loc = op->getLoc();
    auto init = createReductionEmptyTensor(rewriter, loc, *resultType, data,
                                           axesView, keepdims);
    if (failed(init))
      return rewriter.notifyMatchFailure(
          op, "result type is incompatible with the reduction shape");

    Value axes = materializeReductionAxes(rewriter, loc, axesView);
    auto hipOp = HipOpTy::create(rewriter, loc, *context, data, axes, *init,
                                 rewriter.getI64IntegerAttr(keepdims),
                                 rewriter.getI64IntegerAttr(noopWithEmptyAxes),
                                 rewriter.getDenseI64ArrayAttr(axesView));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void populateReductionConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<OnnxReductionToHip<ReduceSumOp>>(ctx, "onnx.ReduceSum");
  patterns.add<OnnxReductionToHip<ReduceMeanOp>>(ctx, "onnx.ReduceMean");
  patterns.add<OnnxReductionToHip<ReduceL2Op>>(ctx, "onnx.ReduceL2");
  patterns.add<OnnxReductionToHip<ReduceMaxOp>>(ctx, "onnx.ReduceMax");
  patterns.add<OnnxReductionToHip<ReduceMinOp>>(ctx, "onnx.ReduceMin");
  patterns.add<OnnxReductionToHip<ReduceProdOp>>(ctx, "onnx.ReduceProd");
}

} // namespace mlir::hip
