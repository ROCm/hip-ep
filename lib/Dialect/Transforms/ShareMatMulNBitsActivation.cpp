/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ShareMatMulNBitsActivation.cpp - Group sibling MatMulNBits ---------===//
//
// The runtime's integer-dot (dp4a) MatMulNBits path quantizes its activation
// to int8 before every GEMV. Sibling ops that read the same activation (the
// Q/K/V or gate/up projections of one input) quantize identical data once per
// sibling. This pass tags each such group so the runtime can quantize once:
//
//   hip.act_quant_group  = i64 group id, unique in the module (> 0)
//   hip.act_quant_leader = unit attribute on the first op of the group
//
// The leader runs first in program order and always refreshes (or
// invalidates) the runtime's cached copy, so a follower never reuses an
// activation quantized by an earlier inference or by an unrelated op that
// happened to get the same buffer address. That guarantee needs every member
// of a group in one block reading one SSA value, which is the grouping key.
//
// When the activation comes from a producer that can write the int8 copy in
// its own epilogue (the output of hyper-connection pre-mix or mixer, or the
// normed output of post-mix-norm), the producer leads the group instead, even
// for a single consumer, and gets
//
//   hip.act_quant_group      = the group id
//   hip.act_quant_block_size = i64 block size every consumer quantizes with
//
// so no consumer quantizes at all.
//
// Runs on tensor IR, right before bufferization: a tensor value is one
// immutable value, and bufferization keeps its buffer intact until the last
// reader. A memref value is not a safe key after pooling, where CSE merges
// identical pool views of unrelated tensors into one value.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "hip-share-matmul-nbits-activation"

STATISTIC(NumGroups, "Number of MatMulNBits activation-sharing groups");
STATISTIC(
    NumFollowers,
    "Number of MatMulNBits ops that reuse a group's quantized activation");
STATISTIC(NumProducers,
          "Number of producer ops that write a group's quantized activation");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_SHAREMATMULNBITSACTIVATIONPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// The reduction width of `input` as written by an int8-capable producer in
// `block`, i.e. the K its consumers must have; nullopt when there is none.
std::optional<int64_t> producerActivationWidth(Value input, Block *block) {
  Operation *def = input.getDefiningOp();
  if (!def || def->getBlock() != block)
    return std::nullopt;
  if (auto pre = dyn_cast<HyperConnectionPreMixOp>(def))
    return pre.getHidden();
  if (auto mixer = dyn_cast<HyperConnectionMixerOp>(def)) {
    if (input != mixer->getResult(0))
      return std::nullopt;
    return mixer.getHidden();
  }
  if (auto post = dyn_cast<HyperConnectionPostMixNormOp>(def)) {
    if (input != post->getResult(1))
      return std::nullopt;
    return post.getBranches() * post.getHidden();
  }
  return std::nullopt;
}

// Every consumer must quantize with the same block size, since the producer
// writes a single copy.
std::optional<int64_t> commonBlockSize(ArrayRef<MatMulNBitsOp> ops, int64_t k) {
  const int64_t blockSize = MatMulNBitsOp(ops.front()).getBlockSize();
  for (MatMulNBitsOp mm : ops)
    if (mm.getBits() != 4 || static_cast<int64_t>(mm.getK()) != k ||
        static_cast<int64_t>(mm.getBlockSize()) != blockSize)
      return std::nullopt;
  return blockSize;
}

struct ShareMatMulNBitsActivationPass
    : public impl::ShareMatMulNBitsActivationPassBase<
          ShareMatMulNBitsActivationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());
    int64_t nextGroup = 1;

    module.walk([&](Block *block) {
      llvm::MapVector<Value, SmallVector<MatMulNBitsOp, 4>> byInput;
      for (Operation &op : *block)
        if (auto mm = dyn_cast<MatMulNBitsOp>(op))
          byInput[mm.getA()].push_back(mm);

      for (auto &[input, ops] : byInput) {
        std::optional<int64_t> producerBlockSize;
        if (std::optional<int64_t> k = producerActivationWidth(input, block))
          producerBlockSize = commonBlockSize(ops, *k);
        if (!producerBlockSize && ops.size() < 2)
          continue;
        const int64_t group = nextGroup++;
        for (MatMulNBitsOp mm : ops)
          mm->setAttr("hip.act_quant_group", builder.getI64IntegerAttr(group));
        ++NumGroups;
        if (producerBlockSize) {
          Operation *producer = input.getDefiningOp();
          producer->setAttr("hip.act_quant_group",
                            builder.getI64IntegerAttr(group));
          producer->setAttr("hip.act_quant_block_size",
                            builder.getI64IntegerAttr(*producerBlockSize));
          ++NumProducers;
          NumFollowers += ops.size();
        } else {
          ops.front()->setAttr("hip.act_quant_leader", builder.getUnitAttr());
          NumFollowers += ops.size() - 1;
        }
      }
    });
  }
};

} // namespace
} // namespace hip
} // namespace mlir
