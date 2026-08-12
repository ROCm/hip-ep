/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AttentionScorePrepOptimizations.cpp - Swin-style attn prep folds
//----===//
//
// Pre-lowering patterns for window-attention score preparation:
//
//   1. FoldMatMulScaleMul
//        MatMul(A, B) -> Mul(scores, scale)   [scale broadcasts, e.g. Hx1x1]
//      =>
//        Mul(A, scale) -> MatMul(A', B)
//      Removes the broadcast Mul on the large [W,H,M,N] score tensor. The fold
//      is valid only when scale is constant along MatMul's contraction (K) and
//      output column (N) axes; e.g. Swin per-head Hx1x1 scales pass, column
//      scales like [10,100] on a 2x2 MatMul are rejected.
//
//   2. ExpandConstantBroadcastAdd
//        Add(activation, Constant[c])   [Constant broadcasts to output]
//      =>
//        Add(activation, Constant[c'] )  [c' pre-expanded to output shape]
//      Removes runtime broadcast Add; lowers to same-shape hip.add instead of
//      hip_elementwise_binary_bcast on large attention maps. Skipped when the
//      expanded constant would exceed kMaxConstantExpandElements (compile/memory
//      guard for tiny biases on huge static tensors).
//
// Must run BEFORE lowerOnnxConstants so inline and ORT mem-addr constants are
// still readable (value attribute or host pointer in location/offset).
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cstring>

STATISTIC(
    NumMatMulScaleFolded,
    "Number of MatMul->Mul(scale) chains folded into Mul(A,scale)->MatMul");
STATISTIC(
    NumConstantBiasExpanded,
    "Number of broadcast Add(constant bias) ops converted to same-shape Add");

namespace mlir {
namespace hip {

namespace {

static constexpr llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

/// Max output elements when materializing a broadcast constant at compile time
/// (~1 Mi elems; ~2 MiB fp16). Avoids multi-GB DenseElementsAttr on huge maps.
static constexpr int64_t kMaxConstantExpandElements = 1 << 20;

static RankedTensorType signlessForRawBuffer(RankedTensorType tensorType) {
  Type elem = tensorType.getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elem)) {
    if (!intTy.isSignless())
      return RankedTensorType::get(
          tensorType.getShape(),
          IntegerType::get(tensorType.getContext(), intTy.getWidth()));
  }
  return tensorType;
}

static bool isStaticShape(RankedTensorType type) {
  return type && type.hasStaticShape();
}

static int64_t numElements(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0)
      return -1;
    n *= d;
  }
  return n;
}

/// NumPy-style broadcast output shape for two ranked shapes (right-aligned).
static FailureOr<SmallVector<int64_t>> broadcastShape(ArrayRef<int64_t> a,
                                                      ArrayRef<int64_t> b) {
  size_t rank = std::max(a.size(), b.size());
  SmallVector<int64_t> out(rank);
  for (size_t i = 0; i < rank; ++i) {
    int64_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
    int64_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
    if (da == db || da == 1 || db == 1) {
      out[i] = da == 1 ? db : da;
      continue;
    }
    return failure();
  }
  return out;
}

static bool shapesEqual(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  for (auto [x, y] : llvm::zip(a, b))
    if (x != y)
      return false;
  return true;
}

static bool broadcastsTo(ArrayRef<int64_t> smallShape,
                         ArrayRef<int64_t> outShape) {
  if (shapesEqual(smallShape, outShape))
    return false;
  auto bc = broadcastShape(smallShape, outShape);
  return succeeded(bc) && shapesEqual(*bc, outShape);
}

/// Scale extent aligned to tensorShape[tensorDim] (NumPy right-aligned broadcast).
static int64_t alignedScaleExtent(ArrayRef<int64_t> tensorShape,
                                  ArrayRef<int64_t> scaleShape,
                                  size_t tensorDim) {
  size_t tensorRank = tensorShape.size();
  size_t scaleRank = scaleShape.size();
  if (tensorDim >= tensorRank)
    return 1;
  if (tensorRank - tensorDim > scaleRank)
    return 1;
  size_t scaleDim = tensorDim - (tensorRank - scaleRank);
  return scaleShape[scaleDim];
}

/// (A * scale) @ B == (A @ B) * scale when scale does not vary on K or N.
static bool scaleCommutesWithMatMulOnA(ArrayRef<int64_t> aShape,
                                       ArrayRef<int64_t> resultShape,
                                       ArrayRef<int64_t> scaleShape) {
  if (aShape.size() < 2 || resultShape.size() < 2)
    return false;
  int64_t kExtent =
      alignedScaleExtent(aShape, scaleShape, aShape.size() - 1);
  int64_t nExtent = alignedScaleExtent(resultShape, scaleShape,
                                       resultShape.size() - 1);
  return kExtent == 1 && nExtent == 1;
}

/// Read dense data for an inline-value or ORT mem-addr onnx.Constant.
static FailureOr<DenseElementsAttr>
getConstantDenseElements(mlir::Operation *constOp) {
  if (!constOp || constOp->getName().getStringRef() != "onnx.Constant")
    return failure();

  if (auto valueAttr = constOp->getAttrOfType<DenseElementsAttr>("value"))
    return valueAttr;

  auto locAttr = constOp->getAttrOfType<StringAttr>("location");
  auto offsetAttr = constOp->getAttrOfType<IntegerAttr>("offset");
  auto sizeAttr = constOp->getAttrOfType<IntegerAttr>("size");
  if (!locAttr || !offsetAttr || !sizeAttr)
    return failure();
  if (locAttr.getValue() != kOrtMemAddrTag)
    return failure();

  int64_t offsetVal = offsetAttr.getInt();
  int64_t dataSize = sizeAttr.getInt();
  if (offsetVal == 0 || dataSize <= 0)
    return failure();

  auto tensorType = dyn_cast<RankedTensorType>(constOp->getResult(0).getType());
  if (!tensorType)
    return failure();

  auto denseType = signlessForRawBuffer(tensorType);
  const void *dataPtr =
      reinterpret_cast<const void *>(static_cast<uintptr_t>(offsetVal));
  auto rawData =
      llvm::ArrayRef<char>(static_cast<const char *>(dataPtr), dataSize);
  return DenseElementsAttr::getFromRawBuffer(denseType, rawData);
}

static FailureOr<DenseElementsAttr>
expandBroadcastConstant(DenseElementsAttr src, ArrayRef<int64_t> srcShape,
                        ArrayRef<int64_t> outShape) {
  if (!src.isSplat() &&
      src.getNumElements() != static_cast<uint64_t>(numElements(srcShape)))
    return failure();

  int64_t outNumel = numElements(outShape);
  if (outNumel <= 0)
    return failure();

  Type elemTy = src.getElementType();
  int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  if (elemBytes <= 0)
    return failure();

  SmallVector<char> outRaw(static_cast<size_t>(outNumel * elemBytes));

  auto copyElem = [&](const char *from, char *to) {
    std::memcpy(to, from, static_cast<size_t>(elemBytes));
  };

  if (src.isSplat()) {
    auto splatRaw = src.getRawData();
    for (int64_t i = 0; i < outNumel; ++i)
      copyElem(splatRaw.data(), outRaw.data() + i * elemBytes);
  } else {
    auto srcRaw = src.getRawData();
    size_t outRank = outShape.size();
    size_t srcRank = srcShape.size();
    SmallVector<int64_t> outCoords(outRank);
    SmallVector<int64_t> srcStrides(srcRank, 1);
    for (int i = static_cast<int>(srcRank) - 2; i >= 0; --i)
      srcStrides[i] = srcStrides[i + 1] * srcShape[i + 1];

    for (int64_t linear = 0; linear < outNumel; ++linear) {
      int64_t rem = linear;
      for (int d = static_cast<int>(outRank) - 1; d >= 0; --d) {
        outCoords[d] = rem % outShape[d];
        rem /= outShape[d];
      }
      int64_t srcLinear = 0;
      for (size_t d = 0; d < outRank; ++d) {
        size_t srcDim = d - (outRank - srcRank);
        int64_t srcExtent = (srcDim < srcRank) ? srcShape[srcDim] : 1;
        int64_t coord = (srcExtent == 1) ? 0 : outCoords[d];
        if (srcDim < srcRank)
          srcLinear += coord * srcStrides[srcDim];
      }
      copyElem(srcRaw.data() + srcLinear * elemBytes,
               outRaw.data() + linear * elemBytes);
    }
  }

  auto outType = RankedTensorType::get(outShape, elemTy);
  outType = signlessForRawBuffer(outType);
  return DenseElementsAttr::getFromRawBuffer(outType, outRaw);
}

static Operation *getDefiningConstantOp(mlir::Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return nullptr;
  return def;
}

/// MatMul(A,B) -> Mul(scores, scale)  =>  Mul(A,scale) -> MatMul(A',B)
struct FoldMatMulScaleMul : public RewritePattern {
  FoldMatMulScaleMul(MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *mulOp,
                                PatternRewriter &rewriter) const override {
    if (mulOp->getNumOperands() != 2)
      return failure();

    Operation *matmulOp = nullptr;
    Value scaleVal;
    if (auto *def = mulOp->getOperand(0).getDefiningOp()) {
      if (def->getName().getStringRef() == "onnx.MatMul") {
        matmulOp = def;
        scaleVal = mulOp->getOperand(1);
      }
    }
    if (!matmulOp) {
      if (auto *def = mulOp->getOperand(1).getDefiningOp()) {
        if (def->getName().getStringRef() == "onnx.MatMul") {
          matmulOp = def;
          scaleVal = mulOp->getOperand(0);
        }
      }
    }
    if (!matmulOp || matmulOp->getNumOperands() != 2)
      return failure();
    if (!matmulOp->hasOneUse())
      return failure();
    if (!getDefiningConstantOp(scaleVal))
      return failure();

    auto mulResultTy =
        dyn_cast<RankedTensorType>(mulOp->getResult(0).getType());
    auto matmulResultTy =
        dyn_cast<RankedTensorType>(matmulOp->getResult(0).getType());
    if (!isStaticShape(mulResultTy) || !isStaticShape(matmulResultTy))
      return failure();
    if (mulResultTy.getShape() != matmulResultTy.getShape())
      return failure();

    Value a = matmulOp->getOperand(0);
    Value b = matmulOp->getOperand(1);
    auto aTy = dyn_cast<RankedTensorType>(a.getType());
    auto scaleTy = dyn_cast<RankedTensorType>(scaleVal.getType());
    if (!aTy || !scaleTy)
      return failure();

    auto scaledATy = broadcastShape(aTy.getShape(), scaleTy.getShape());
    if (failed(scaledATy) || !shapesEqual(*scaledATy, aTy.getShape()))
      return failure();
    if (!scaleCommutesWithMatMulOnA(aTy.getShape(), matmulResultTy.getShape(),
                                    scaleTy.getShape()))
      return failure();

    Location loc = mulOp->getLoc();
    rewriter.setInsertionPoint(matmulOp);
    OperationState mulState(loc, "onnx.Mul");
    mulState.addOperands({a, scaleVal});
    mulState.addTypes(aTy);
    Operation *aScaled = rewriter.create(mulState);
    matmulOp->setOperand(0, aScaled->getResult(0));
    rewriter.replaceOp(mulOp, matmulOp->getResult(0));
    ++NumMatMulScaleFolded;
    return success();
  }
};

/// Add(x, Constant[bias]) with broadcast  =>  Add(x, Constant[expanded])
struct ExpandConstantBroadcastAdd : public RewritePattern {
  ExpandConstantBroadcastAdd(MLIRContext *ctx)
      : RewritePattern("onnx.Add", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *addOp,
                                PatternRewriter &rewriter) const override {
    if (addOp->getNumOperands() != 2)
      return failure();

    auto outTy = dyn_cast<RankedTensorType>(addOp->getResult(0).getType());
    if (!isStaticShape(outTy))
      return failure();

    Value actVal;
    Operation *constOp = nullptr;
    for (Value v : addOp->getOperands()) {
      if (Operation *c = getDefiningConstantOp(v)) {
        if (constOp)
          return failure();
        constOp = c;
      } else {
        if (actVal)
          return failure();
        actVal = v;
      }
    }
    if (!constOp || !actVal)
      return failure();

    auto constTy = dyn_cast<RankedTensorType>(constOp->getResult(0).getType());
    if (!constTy || !isStaticShape(constTy))
      return failure();
    if (!broadcastsTo(constTy.getShape(), outTy.getShape()))
      return failure();

    int64_t outNumel = numElements(outTy.getShape());
    if (outNumel <= 0 || outNumel > kMaxConstantExpandElements)
      return failure();

    auto denseOr = getConstantDenseElements(constOp);
    if (failed(denseOr))
      return failure();

    auto expandedOr =
        expandBroadcastConstant(*denseOr, constTy.getShape(), outTy.getShape());
    if (failed(expandedOr))
      return failure();

    Location loc = addOp->getLoc();
    auto expandedTy =
        RankedTensorType::get(outTy.getShape(), constTy.getElementType());
    expandedTy = signlessForRawBuffer(expandedTy);

    OperationState cstState(loc, "onnx.Constant");
    cstState.addTypes(expandedTy);
    cstState.addAttribute("value", *expandedOr);
    Operation *newConst = rewriter.create(cstState);

    unsigned constIdx = addOp->getOperand(0) == constOp->getResult(0) ? 0 : 1;
    rewriter.modifyOpInPlace(
        addOp, [&] { addOp->setOperand(constIdx, newConst->getResult(0)); });
    ++NumConstantBiasExpanded;
    return success();
  }
};

} // namespace

void populateAttentionScorePrepOptimizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx) {
  patterns.add<FoldMatMulScaleMul, ExpandConstantBroadcastAdd>(ctx);
}

} // namespace hip
} // namespace mlir
