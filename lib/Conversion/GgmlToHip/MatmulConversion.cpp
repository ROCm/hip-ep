/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- MatmulConversion.cpp - Lower GGML matrix multiplication ------------===//

#include "GgmlToHipUtils.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"
#include "hip/Dialect/Ggml/IR/GgmlSymbols.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace mlir {
namespace hip {
namespace {

/// Return one explicit Q8_1 preparation for an activation SSA value and K.
///
/// The operation is placed immediately after the activation producer (or at
/// the start of the block for a block argument), so it dominates every use of
/// that SSA value. Looking through the activation's users makes the result
/// independent of the order in which dialect-conversion visits its MUL_MAT
/// consumers. Tensor SSA identity is the cache key deliberately: bufferized
/// pointers are reused for different logical values and are not safe keys.
static Value findQ8Activation(Value activation, int64_t k, int64_t sumMode,
                              int64_t rowsPerToken, Value context,
                              Operation *consumer) {
  auto activationType = dyn_cast<RankedTensorType>(activation.getType());
  if (!activationType)
    return {};
  int64_t q8RowBytes = 0;
  if (llvm::MulOverflow(k / kquant::kQ8_1BlockElements, kquant::kQ8_1BlockBytes,
                        q8RowBytes))
    return {};
  SmallVector<int64_t> q8Shape(activationType.getShape());
  q8Shape.back() = q8RowBytes;

  DominanceInfo dominance(consumer->getParentOfType<func::FuncOp>());
  for (Operation *user : activation.getUsers()) {
    auto quantize = dyn_cast<QuantizeQ8_1Op>(user);
    if (!quantize || quantize.getCtx() != context ||
        quantize.getKAttr().getInt() != k || quantize.getSumMode() != sumMode ||
        quantize.getRowsPerToken() != rowsPerToken ||
        quantize.getResultTensors().size() != 1)
      continue;
    auto resultType = dyn_cast<RankedTensorType>(
        quantize.getResultTensors().front().getType());
    if (resultType && resultType.getShape() == ArrayRef<int64_t>(q8Shape) &&
        resultType.getElementType().isUnsignedInteger(8) &&
        dominance.dominates(quantize.getOperation(), consumer))
      return quantize.getResultTensors().front();
  }
  return {};
}

static bool isEligibleNativeKQuantWeight(Value weight, int64_t k) {
  // A packed runtime weight behind its cast (lowerPackedRuntimeWeights).
  Operation *source = nullptr;
  StringAttr encoding;
  if (auto cast = weight.getDefiningOp<UnrealizedConversionCastOp>()) {
    source = cast;
    encoding = cast->getAttrOfType<StringAttr>("ggml.source_encoding");
  }
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  const std::optional<GgmlKQuantFormatInfo> packedFormat =
      encoding ? getGgmlKQuantFormatInfo(encoding.getValue()) : std::nullopt;
  if (!source || !source->hasOneUse() || !weightType ||
      weightType.getRank() != 4 || weightType.getDimSize(0) != 1 ||
      weightType.getDimSize(3) != k ||
      (!weightType.getElementType().isF16() &&
       !weightType.getElementType().isF32()) ||
      !packedFormat) {
    return false;
  }

  const int64_t n = weightType.getDimSize(2);
  if (ShapedType::isDynamic(n) || n <= 0)
    return false;
  return weightType.getDimSize(1) == 1;
}

/// Return true for a packed projection that consumes `activation` and can
/// share an explicit Q8_1 preparation with another packed projection.
static bool isEligibleNativeKQuantConsumer(Operation *user, Value activation,
                                           int64_t k) {
  auto candidate = dyn_cast<ggml::MulMatOp>(user);
  return candidate && candidate.getInput() == activation &&
         candidate.getPrecision() == ggml::ComputePrecision::Default &&
         candidate.getHint() == 0 &&
         isEligibleNativeKQuantWeight(candidate.getWeight(), k);
}

static bool hasMultipleNativeKQuantConsumers(Value activation, int64_t k) {
  unsigned count = 0;
  for (Operation *user : activation.getUsers()) {
    if (isEligibleNativeKQuantConsumer(user, activation, k) && ++count > 1)
      return true;
  }
  return false;
}

static FailureOr<Value> createQ8Activation(PatternRewriter &rewriter,
                                           Location loc, Value context,
                                           Value activation, int64_t k,
                                           int64_t sumMode,
                                           int64_t rowsPerToken) {

  auto activationType = cast<RankedTensorType>(activation.getType());
  int64_t q8RowBytes = 0;
  if (llvm::MulOverflow(k / kquant::kQ8_1BlockElements, kquant::kQ8_1BlockBytes,
                        q8RowBytes)) {
    emitError(loc) << "Q8_1 row byte cardinality overflows int64";
    return failure();
  }
  SmallVector<int64_t> q8Shape(activationType.getShape());
  q8Shape.back() = q8RowBytes;
  auto q8Type = RankedTensorType::get(
      q8Shape, rewriter.getIntegerType(8, /*isSigned=*/false));

  OpBuilder::InsertionGuard guard(rewriter);
  if (Operation *producer = activation.getDefiningOp())
    rewriter.setInsertionPointAfter(producer);
  else
    rewriter.setInsertionPointToStart(
        cast<BlockArgument>(activation).getOwner());

  Value init = createGgmlEmptyTensor(rewriter, loc, q8Type, activation);
  if (!init)
    return failure();
  auto quantize = QuantizeQ8_1Op::create(
      rewriter, loc, context, activation, init, rewriter.getI64IntegerAttr(k),
      rewriter.getI64IntegerAttr(sumMode),
      rewriter.getI64IntegerAttr(rowsPerToken));
  return quantize.getResultTensors().front();
}

/// `op` as the native kernels multiply by a packed weight: its bytes
/// `packed` of `format`, K by N.
static LogicalResult packedMatmul(ggml::MulMatOp op, PatternRewriter &rewriter,
                                  Value context, Value input, Value packed,
                                  const GgmlKQuantFormatInfo &format, int64_t k,
                                  int64_t n) {
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  constexpr int64_t sumMode = kquant::kSumMatrixScaledVectorInteger;
  Value quantizedActivation =
      findQ8Activation(input, k, sumMode, /*rowsPerToken=*/1, context, op);
  if (quantizedActivation || hasMultipleNativeKQuantConsumers(input, k)) {
    if (!quantizedActivation) {
      FailureOr<Value> created =
          createQ8Activation(rewriter, op.getLoc(), context, input, k, sumMode,
                             /*rowsPerToken=*/1);
      if (failed(created))
        return failure();
      quantizedActivation = *created;
    }
    Value init = createGgmlEmptyTensor(rewriter, op.getLoc(), resultType,
                                       quantizedActivation);
    if (!init)
      return failure();
    auto matmul = KQuantMatmulQ8_1Op::create(
        rewriter, op.getLoc(), context, quantizedActivation, packed, init,
        rewriter.getI64IntegerAttr(k), rewriter.getI64IntegerAttr(n),
        rewriter.getStringAttr(kquant::getFormatName(format.format)),
        rewriter.getI64IntegerAttr(sumMode),
        rewriter.getI64IntegerAttr(/*rowsPerToken=*/1));
    rewriter.replaceOp(op, matmul.getResultTensors());
    return success();
  }
  Value init = createGgmlEmptyTensor(rewriter, op.getLoc(), resultType, input);
  if (!init)
    return failure();
  auto matmul = KQuantMatmulOp::create(
      rewriter, op.getLoc(), context, input, packed, init,
      rewriter.getI64IntegerAttr(k), rewriter.getI64IntegerAttr(n),
      rewriter.getStringAttr(kquant::getFormatName(format.format)));
  rewriter.replaceOp(op, matmul.getResultTensors());
  return success();
}

struct MulMatToHip final : OpRewritePattern<ggml::MulMatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::MulMatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getPrecision() != ggml::ComputePrecision::Default ||
        op.getHint() != 0) {
      return op.emitOpError(
          "HIP lowering supports only default precision and hint");
    }
    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context)) {
      return failure();
    }

    Value input = op.getInput();
    Value weightOperand = op.getWeight();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightType = dyn_cast<RankedTensorType>(weightOperand.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !weightType || !resultType || inputType.getRank() != 4 ||
        weightType.getRank() != 4 || resultType.getRank() != 4) {
      return op.emitOpError("HIP lowering requires rank-4 tensor operands");
    }
    if (!inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32()) {
      return op.emitOpError(
          "HIP lowering requires f32 activations and an f32 result");
    }
    if (!haveEquivalentGgmlDimensionExtent(weightOperand, 3, input, 3) ||
        !haveEquivalentGgmlDimensionExtent(input, 0, op.getResult(), 0) ||
        !haveEquivalentGgmlDimensionExtent(input, 1, op.getResult(), 1) ||
        !haveEquivalentGgmlDimensionExtent(input, 2, op.getResult(), 2) ||
        !haveEquivalentGgmlDimensionExtent(weightOperand, 2, op.getResult(),
                                           3)) {
      return op.emitOpError(
          "HIP lowering cannot prove GGML matrix/result dimension identities");
    }

    // A packed weight the caller hands over (a ggml backend's split reads
    // the model's tensors where llama.cpp loaded them): its bytes, [1, 1, N,
    // row bytes], behind a cast to its type (lowerPackedRuntimeWeights).
    auto cast = weightOperand.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast || cast.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(
          op, "supports only packed K-quant weights");
    Value bytes = cast.getInputs().front();
    auto bytesType = dyn_cast<RankedTensorType>(bytes.getType());
    auto encoding = cast->getAttrOfType<StringAttr>("ggml.source_encoding");
    std::optional<GgmlKQuantFormatInfo> packedFormat =
        encoding ? getGgmlKQuantFormatInfo(encoding.getValue()) : std::nullopt;
    const int64_t n = weightType.getDimSize(2);
    const int64_t k = weightType.getDimSize(3);
    if (!bytesType || !bytesType.hasStaticShape() || !packedFormat ||
        weightType.getDimSize(0) != 1 || weightType.getDimSize(1) != 1 ||
        ShapedType::isDynamic(n) || ShapedType::isDynamic(k) ||
        k % packedFormat->blockElements != 0)
      return op.emitOpError("a packed runtime weight must be one static "
                            "matrix of whole blocks the native kernels read");
    Value flat = tensor::CollapseShapeOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get({bytesType.getNumElements()},
                              bytesType.getElementType()),
        bytes, SmallVector<ReassociationIndices>{{0, 1, 2, 3}});
    if (failed(packedMatmul(op, rewriter, *context, input, flat, *packedFormat,
                            k, n)))
      return failure();
    if (cast->use_empty())
      rewriter.eraseOp(cast);
    return success();
  }
};

} // namespace

void populateGgmlMatmulConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<MulMatToHip>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
