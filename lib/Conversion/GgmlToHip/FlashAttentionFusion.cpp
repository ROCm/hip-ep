/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- FlashAttentionFusion.cpp - GGML grouped-attention fusion ---------===//

#include "GgmlToHipUtils.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hip {
namespace {

static bool hasPermutation(ggml::PermuteOp op, ArrayRef<int64_t> expected) {
  ArrayAttr axes = op.getAxes();
  if (axes.size() != expected.size())
    return false;
  for (auto [attribute, value] : llvm::zip_equal(axes, expected)) {
    auto axis = dyn_cast<IntegerAttr>(attribute);
    if (!axis || axis.getInt() != value)
      return false;
  }
  return true;
}

/// Returns the destination rows written by the cache update that produced
/// `keyCache`, or a null value when the chain cannot be proved.
///
/// The rows let the kernel traverse only the live part of the cache. Anything
/// this function cannot prove leaves the operand absent, which keeps the
/// previous whole-cache traversal.
static Value cacheUpdateRows(Value keyCache) {
  // The key operand reaches attention through layout-only views of the
  // updated cache. Walk a bounded chain of them; a longer or unexpected chain
  // is not proof that the rows describe this cache.
  constexpr int kMaximumViewDepth = 4;
  Value current = keyCache;
  for (int depth = 0; depth <= kMaximumViewDepth; ++depth) {
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return Value();
    if (auto setRows = dyn_cast<ggml::SetRowsOp>(producer)) {
      Value indices = setRows.getIndices();
      auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
      if (!indicesType || indicesType.getRank() != 4 ||
          !indicesType.getElementType().isSignlessInteger(64)) {
        return Value();
      }
      for (int64_t dimension = 0; dimension < 3; ++dimension) {
        if (indicesType.getDimSize(dimension) != 1)
          return Value();
      }
      return indices;
    }
    if (!isa<ggml::PermuteOp, ggml::ViewOp, ggml::ReshapeOp, tensor::ReshapeOp>(
            producer))
      return Value();
    current = producer->getOperand(0);
  }
  return Value();
}

/// Prove the result extents of a candidate layout-only permutation. The
/// BSHD/BNSH permutation used here is self-inverse, so the GGML axes attribute
/// and MLIR output-to-input dimension map are both [0, 2, 1, 3].
static bool hasPermutationExtents(ggml::PermuteOp op,
                                  ArrayRef<int64_t> outputToInput) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!inputType || !resultType ||
      inputType.getRank() != static_cast<int64_t>(outputToInput.size()) ||
      resultType.getRank() != static_cast<int64_t>(outputToInput.size()) ||
      inputType.getElementType() != resultType.getElementType()) {
    return false;
  }

  for (auto [outputDimension, inputDimension] :
       llvm::enumerate(outputToInput)) {
    if (!haveEquivalentGgmlDimensionExtent(op.getInput(), inputDimension,
                                           op.getResult(), outputDimension)) {
      return false;
    }
  }
  return true;
}

static bool isKnownContiguous(Value value) {
  return hasDenseInputEvidence(value);
}

/// Recognize the layout boundary around GGML FLASH_ATTN_EXT and retain its
/// explicit-mask semantics in one HIP operation. This is deliberately a
/// pre-conversion fusion: erasing the now-dead source permutes is safe in the
/// ordinary greedy rewrite driver and prevents them from becoming observable
/// device copies during the subsequent full dialect conversion.
struct FuseFlashAttentionLayout final
    : OpRewritePattern<ggml::FlashAttentionExtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::FlashAttentionExtOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getMask() || op.getSinks() ||
        op.getScale().convertToDouble() <= 0.0 ||
        op.getMaxBias().convertToDouble() != 0.0 ||
        op.getLogitSoftcap().convertToDouble() != 0.0 ||
        op.getPrecision() != ggml::ComputePrecision::F32) {
      return rewriter.notifyMatchFailure(
          op, "requires explicit-mask f32 GGML flash-attention semantics");
    }

    auto queryPermute = op.getQuery().getDefiningOp<ggml::PermuteOp>();
    auto keyPermute = op.getKey().getDefiningOp<ggml::PermuteOp>();
    auto valuePermute = op.getValue().getDefiningOp<ggml::PermuteOp>();
    constexpr int64_t kTokenMajorToHeadMajor[] = {0, 2, 1, 3};
    if (!queryPermute || !keyPermute || !valuePermute ||
        !queryPermute->hasOneUse() || !keyPermute->hasOneUse() ||
        !valuePermute->hasOneUse() ||
        !hasPermutation(queryPermute, kTokenMajorToHeadMajor) ||
        !hasPermutation(keyPermute, kTokenMajorToHeadMajor) ||
        !hasPermutation(valuePermute, kTokenMajorToHeadMajor)) {
      return rewriter.notifyMatchFailure(
          op, "requires single-use BSHD-to-BNSH input permutations");
    }
    if (!hasPermutationExtents(queryPermute, kTokenMajorToHeadMajor) ||
        !hasPermutationExtents(keyPermute, kTokenMajorToHeadMajor) ||
        !hasPermutationExtents(valuePermute, kTokenMajorToHeadMajor)) {
      return rewriter.notifyMatchFailure(
          op, "requires each input permutation to preserve every mapped "
              "dimension");
    }

    Value query = queryPermute.getInput();
    Value key = keyPermute.getInput();
    Value value = valuePermute.getInput();
    Value mask = op.getMask();
    auto queryType = dyn_cast<RankedTensorType>(query.getType());
    auto keyType = dyn_cast<RankedTensorType>(key.getType());
    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!queryType || !keyType || !valueType || !maskType || !resultType ||
        queryType.getRank() != 4 || keyType.getRank() != 4 ||
        valueType.getRank() != 4 || maskType.getRank() != 4 ||
        resultType.getRank() != 4 || !queryType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        !keyType.getElementType().isF16() ||
        !valueType.getElementType().isF16() ||
        !maskType.getElementType().isF16() ||
        resultType.getShape().drop_back() != queryType.getShape().drop_back() ||
        resultType.getDimSize(3) != queryType.getDimSize(3) ||
        valueType != keyType) {
      return rewriter.notifyMatchFailure(
          op, "requires F32 BSHD query/result and matching F16 BSGD caches");
    }
    if (!isKnownContiguous(query) || !isKnownContiguous(key) ||
        !isKnownContiguous(value)) {
      return rewriter.notifyMatchFailure(
          op, "requires contiguous token-major query and cache inputs");
    }

    const int64_t batch = queryType.getDimSize(0);
    const int64_t queryHeads = queryType.getDimSize(2);
    const int64_t headDimension = queryType.getDimSize(3);
    // The cache's slots may be the session's (dynamic): the kernel takes
    // them at run time, and the mask must cover the same slots.
    const int64_t cacheCapacity = keyType.getDimSize(1);
    const bool sessionCapacity = ShapedType::isDynamic(cacheCapacity);
    const int64_t kvHeads = keyType.getDimSize(2);
    if (ShapedType::isDynamic(batch) || batch <= 0 ||
        ShapedType::isDynamic(queryHeads) || queryHeads <= 0 ||
        ShapedType::isDynamic(kvHeads) || kvHeads <= 0 ||
        (!sessionCapacity && cacheCapacity <= 0) ||
        ShapedType::isDynamic(headDimension) ||
        !isTokenMajorGqaHeadDimension(headDimension) ||
        queryHeads % kvHeads != 0 || keyType.getDimSize(0) != batch ||
        keyType.getDimSize(3) != headDimension ||
        (maskType.getDimSize(0) != 1 && maskType.getDimSize(0) != batch) ||
        maskType.getDimSize(1) != 1 ||
        !haveEquivalentGgmlDimensionExtent(key, 1, mask, 3) ||
        !haveEquivalentGgmlDimensionExtent(key, 1, value, 1)) {
      return rewriter.notifyMatchFailure(
          op, "requires compatible grouped-attention geometry");
    }
    if (!haveEquivalentGgmlDimensionExtent(query, 1, mask, 2)) {
      return op.emitOpError(
          "cannot prove matching query and mask token dimensions for fused "
          "HIP attention");
    }
    // The result has the query's extents.
    for (int64_t dimension = 0; dimension < queryType.getRank(); ++dimension) {
      if (!haveEquivalentGgmlDimensionExtent(query, dimension, op.getResult(),
                                             dimension)) {
        return op.emitOpError(
                   "cannot prove shape-preserving fused attention dimension ")
               << dimension;
      }
    }

    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context))
      return failure();
    Value init =
        createGgmlEmptyTensor(rewriter, op.getLoc(), resultType, query);
    if (!init)
      return failure();

    auto fused = TokenMajorGqaOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, *context, query, key,
        value, mask, cacheUpdateRows(key), init, op.getScaleAttr());
    copyGgmlResultProvenanceAttrs(op, fused);

    rewriter.replaceOp(op, fused.getResultTensors());
    rewriter.eraseOp(queryPermute);
    rewriter.eraseOp(keyPermute);
    rewriter.eraseOp(valuePermute);
    return success();
  }
};

} // namespace

void populateGgmlFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseFlashAttentionLayout>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
