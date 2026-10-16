/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HoistDynamicQuant.cpp - quantize an activation once, upstream ------===//
//
// At decode every int4 projection runs the integer dot-product GEMV, which
// first quantizes its fp16 activation to int8 in a launch of its own. A
// decoder layer's normalized activation feeds several such consumers (a
// projection group and the routed experts), and each quantizes it again.
//
// This pass makes the quantized activation an SSA value that every consumer
// shares. When the activation is the output of a hip.skip_rms_norm or a
// hip.gated_rms_norm, the norm writes it alongside its fp16 output, which
// removes the launch altogether;
// otherwise one hip.dynamic_quant_i8 serves all consumers, which only pays
// when there are at least two.
//
//   Before:
//     %y:2 = hip.skip_rms_norm(%ctx) ins(%x, %r, %g : ...) outs(%e0, %e1 : ...)
//     %p:2 = hip.matmul_nbits_group(%ctx) ins(%y#0 : ...) ... {block_size = 32}
//     %m   = hip.qmoe(%ctx) ins(%y#0, ...) ... {block_size = 32}
//   After:
//     %y:4 = hip.skip_rms_norm(%ctx) ins(%x, %r, %g : ...)
//              outs(%e0, %e1, %eq, %es : ...) {quant_block_size = 32}
//     %p:2 = hip.matmul_nbits_group(%ctx) ins(%y#0 : ...)
//              a_quant(%y#2, %y#3 : ...) ...
//     %m   = hip.qmoe(%ctx) ins(%y#0, ...) input_quant(%y#2, %y#3 : ...) ...
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

#define DEBUG_TYPE "hip-hoist-dynamic-quant"

STATISTIC(NumNormFolds, "Number of norms that also write the quantization");
STATISTIC(NumQuantOps, "Number of hip.dynamic_quant_i8 ops created");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HOISTDYNAMICQUANTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

using QuantKey = std::pair<Value, int64_t>;

// The activation and quantization block size of an op that can take a
// pre-quantized activation, if it is one.
std::optional<QuantKey> quantizableInput(Operation *op) {
  std::optional<QuantKey> key;
  if (auto mm = dyn_cast<MatMulNBitsOp>(op)) {
    if (!mm.getAQ() && mm.getBits() == 4)
      key = {mm.getA(), mm.getBlockSize()};
  } else if (auto group = dyn_cast<MatMulNBitsGroupOp>(op)) {
    if (!group.getAQ() && group.getBits() == 4)
      key = {group.getA(), group.getBlockSize()};
  } else if (auto moe = dyn_cast<QMoEOp>(op)) {
    if (!moe.getInputQ() && moe.getExpertWeightBits() == 4)
      key = {moe.getInput(), moe.getBlockSize()};
  }
  if (!key)
    return std::nullopt;

  // The integer path quantizes fp16 rows whose length is a multiple of 32 in
  // groups that are a power of two of at least 32 elements.
  auto type = dyn_cast<RankedTensorType>(key->first.getType());
  if (!type || !type.getElementType().isF16() || type.getRank() < 1 ||
      type.isDynamicDim(type.getRank() - 1))
    return std::nullopt;
  const int64_t k = type.getDimSize(type.getRank() - 1);
  const int64_t blockSize = key->second;
  if (k % 32 != 0 || blockSize < 32 || !llvm::isPowerOf2_64(blockSize))
    return std::nullopt;
  return key;
}

void setQuantOperands(Operation *op, Value q, Value scale) {
  if (auto mm = dyn_cast<MatMulNBitsOp>(op)) {
    mm.getAQMutable().assign(q);
    mm.getAScaleMutable().assign(scale);
  } else if (auto group = dyn_cast<MatMulNBitsGroupOp>(op)) {
    group.getAQMutable().assign(q);
    group.getAScaleMutable().assign(scale);
  } else {
    auto moe = cast<QMoEOp>(op);
    moe.getInputQMutable().assign(q);
    moe.getInputScaleMutable().assign(scale);
  }
}

RankedTensorType quantType(RankedTensorType activation) {
  return RankedTensorType::get(
      activation.getShape(), IntegerType::get(activation.getContext(), 8));
}

RankedTensorType scaleType(RankedTensorType activation, int64_t blockSize) {
  SmallVector<int64_t> shape(activation.getShape());
  shape.back() = llvm::divideCeil(shape.back(), blockSize);
  return RankedTensorType::get(shape,
                               Float32Type::get(activation.getContext()));
}

// Destination of `type` whose dynamic extents are those of `like`.
Value createInit(OpBuilder &b, Location loc, Value like,
                 RankedTensorType type) {
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < type.getRank(); ++i)
    if (type.isDynamicDim(i))
      dynSizes.push_back(tensor::DimOp::create(b, loc, like, i));
  return tensor::EmptyOp::create(b, loc, type.getShape(),
                                 type.getElementType(), dynSizes);
}

// Rebuilds the norm producing `activation` with the quantized pair as two
// more outputs and returns them, or returns nothing when it is not a norm
// that can.
std::optional<std::pair<Value, Value>> foldIntoNorm(Value activation,
                                                    int64_t blockSize) {
  auto result = dyn_cast<OpResult>(activation);
  if (!result || result.getResultNumber() != 0)
    return std::nullopt;
  Operation *norm = result.getOwner();
  auto skipNorm = dyn_cast<SkipRmsNormOp>(norm);
  auto gatedNorm = dyn_cast<GatedRmsNormOp>(norm);
  if (skipNorm ? skipNorm.hasQuantOutputs()
               : !gatedNorm || gatedNorm.hasQuantOutputs())
    return std::nullopt;
  auto outputsOf = [&]() -> OperandRange {
    return skipNorm ? skipNorm.getOutputs() : gatedNorm.getOutputs();
  };
  if (norm->getNumResults() != outputsOf().size())
    return std::nullopt;

  auto type = cast<RankedTensorType>(activation.getType());
  OpBuilder b(norm);
  Location loc = norm->getLoc();
  Value input = skipNorm ? skipNorm.getInput() : gatedNorm.getInput();
  SmallVector<Value> outputs(outputsOf());
  SmallVector<Type> resultTypes(norm->getResultTypes());
  for (RankedTensorType t : {quantType(type), scaleType(type, blockSize)}) {
    outputs.push_back(createInit(b, loc, input, t));
    resultTypes.push_back(t);
  }
  Operation *fused;
  if (skipNorm)
    fused = SkipRmsNormOp::create(
        b, loc, resultTypes, skipNorm.getCtx(), skipNorm.getInput(),
        skipNorm.getSkip(), skipNorm.getGamma(), skipNorm.getBias(), outputs,
        skipNorm.getEpsilonAttr(), b.getI64IntegerAttr(blockSize));
  else
    fused = GatedRmsNormOp::create(
        b, loc, resultTypes, gatedNorm.getCtx(), gatedNorm.getInput(),
        gatedNorm.getGate(), gatedNorm.getWeight(), outputs,
        gatedNorm.getEpsilonAttr(), b.getI64IntegerAttr(blockSize));
  for (auto [from, to] : llvm::zip(norm->getResults(), fused->getResults()))
    from.replaceAllUsesWith(to);
  norm->erase();

  const unsigned n = fused->getNumResults();
  ++NumNormFolds;
  return std::make_pair(fused->getResult(n - 2), fused->getResult(n - 1));
}

std::pair<Value, Value> createQuant(Value ctx, Value activation,
                                    int64_t blockSize) {
  OpBuilder b(activation.getContext());
  if (Operation *def = activation.getDefiningOp())
    b.setInsertionPointAfter(def);
  else
    b.setInsertionPointToStart(cast<BlockArgument>(activation).getOwner());
  Location loc = activation.getLoc();
  auto type = cast<RankedTensorType>(activation.getType());
  SmallVector<RankedTensorType, 2> types = {quantType(type),
                                            scaleType(type, blockSize)};
  SmallVector<Value, 2> inits;
  for (RankedTensorType t : types)
    inits.push_back(createInit(b, loc, activation, t));
  auto quant = DynamicQuantI8Op::create(
      b, loc, TypeRange{types[0], types[1]}, ctx, activation, inits,
      b.getI64IntegerAttr(blockSize));
  ++NumQuantOps;
  return {quant->getResult(0), quant->getResult(1)};
}

struct HoistDynamicQuantPass
    : public impl::HoistDynamicQuantPassBase<HoistDynamicQuantPass> {
  void runOnOperation() override {
    llvm::MapVector<QuantKey, SmallVector<Operation *>> consumers;
    DenseMap<Value, unsigned> numBlockSizes;
    getOperation().walk([&](Operation *op) {
      if (std::optional<QuantKey> key = quantizableInput(op)) {
        auto &ops = consumers[*key];
        if (ops.empty())
          ++numBlockSizes[key->first];
        ops.push_back(op);
      }
    });

    // Folding rebuilds a norm, which replaces values other entries are keyed
    // on, so each entry re-reads its activation from a consumer.
    SmallVector<std::tuple<SmallVector<Operation *>, int64_t, bool>> entries;
    for (auto &[key, ops] : consumers)
      entries.emplace_back(ops, key.second, numBlockSizes[key.first] == 1);

    for (auto &[ops, blockSize, foldable] : entries) {
      Value activation = quantizableInput(ops.front())->first;
      // A norm writes a single quantization.
      std::optional<std::pair<Value, Value>> quant;
      if (foldable)
        quant = foldIntoNorm(activation, blockSize);
      // Every hip op takes the context handle as its first operand.
      if (!quant && ops.size() >= 2)
        quant = createQuant(ops.front()->getOperand(0), activation, blockSize);
      if (!quant)
        continue;
      for (Operation *op : ops)
        setQuantOperands(op, quant->first, quant->second);
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
