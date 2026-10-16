/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GroupMatMulNBits.cpp - hip.matmul_nbits -> hip.matmul_nbits_group --===//
//
// A decoder layer projects one normalized activation through several int4
// weights (q/k/v, or the linear-attention in_proj_* set, or a MoE router next
// to its shared-expert projections). At decode every projection is a GEMV
// plus a separate activation-quantization launch, and the small ones are
// dispatch-bound: two launches of ~2 us each for a few KB of weights. One
// hip.matmul_nbits_group computes all of them from a single quantization and
// a single GEMV launch.
//
// Members keep their own destinations. Concatenating the weights instead
// would make each projection a slice of one output, and every consumer that
// needs an identity-layout operand would then pay a copy.
//
//   Before:
//     %e0 = tensor.empty(%d0, %d1) : tensor<?x?x8192xf16>
//     %q  = hip.matmul_nbits(%ctx) ins(%x, %Bq, %Sq : ...) outs(%e0 : ...)
//             {K = 2048, N = 8192, ...} : tensor<?x?x8192xf16>
//     %e1 = tensor.empty(%d0, %d1) : tensor<?x?x512xf16>
//     %k  = hip.matmul_nbits(%ctx) ins(%x, %Bk, %Sk : ...) outs(%e1 : ...)
//             {K = 2048, N = 512, ...} : tensor<?x?x512xf16>
//   After:
//     %f0 = tensor.empty(%d0, %d1) : tensor<?x?x8192xf16>
//     %f1 = tensor.empty(%d0, %d1) : tensor<?x?x512xf16>
//     %q, %k = hip.matmul_nbits_group(%ctx) ins(%x : ...)
//                weights(%Bq, %Bk : ...) scales(%Sq, %Sk : ...)
//                outs(%f0, %f1 : ...) {K = 2048, Ns = array<i64: 8192, 512>}
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <tuple>

#define DEBUG_TYPE "hip-group-matmul-nbits"

STATISTIC(NumGroups, "Number of hip.matmul_nbits_group ops created");
STATISTIC(NumGrouped, "Number of hip.matmul_nbits ops folded into a group");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_GROUPMATMULNBITSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// The grouped decode kernel is the dp4a GEMV, so only ops it can run are
// worth grouping; anything else would just be computed member by member.
bool isGroupable(MatMulNBitsOp op) {
  if (op.getZeroPoints() || op.getGIdx() || op.getBias())
    return false;
  if (op.getBits() != 4 || op.getK() % 32 != 0 || op.getScaleElemSize() != 2)
    return false;
  if (op.getNumResults() != 1)
    return false;
  auto aType = dyn_cast<RankedTensorType>(op.getA().getType());
  return aType && aType.getElementType().isF16() &&
         isa<RankedTensorType>(op.getResult(0).getType());
}

using GroupKey = std::tuple<Value, int64_t, int64_t, int64_t, int64_t>;

GroupKey keyOf(MatMulNBitsOp op) {
  return {op.getA(), op.getK(), op.getBits(), op.getBlockSize(),
          op.getAccuracyLevel()};
}

// The group sits at the first member, so every later member's weight and
// scales must be available there. Constants defined further down are pure and
// operand-free, so they can move up; anything else excludes the member.
bool hoistOperandsAbove(MatMulNBitsOp member, Operation *anchor,
                        DominanceInfo &dom) {
  SmallVector<Operation *, 2> toMove;
  for (Value v : {member.getB(), member.getScales()}) {
    if (dom.properlyDominates(v, anchor))
      continue;
    Operation *def = v.getDefiningOp();
    if (!def || def->getNumOperands() != 0 || def->getNumRegions() != 0 ||
        def->getBlock() != anchor->getBlock() || !isMemoryEffectFree(def))
      return false;
    toMove.push_back(def);
  }
  for (Operation *def : toMove)
    def->moveBefore(anchor);
  return true;
}

// Fresh destination for a member at the group's position: the member's own
// init may be defined below it.
Value createInit(OpBuilder &b, Location loc, Value A,
                 RankedTensorType resultType) {
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i)
    if (resultType.isDynamicDim(i))
      dynSizes.push_back(tensor::DimOp::create(b, loc, A, i));
  return tensor::EmptyOp::create(b, loc, resultType.getShape(),
                                 resultType.getElementType(), dynSizes);
}

// Drops a replaced member's tensor.empty init and the tensor.dim ops sizing it.
void eraseDeadChain(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || !isOpTriviallyDead(def))
    return;
  SmallVector<Value> operands(def->getOperands());
  def->erase();
  for (Value operand : operands)
    eraseDeadChain(operand);
}

void emitGroup(ArrayRef<MatMulNBitsOp> members) {
  MatMulNBitsOp first = members.front();
  OpBuilder b(first);
  Location loc = b.getFusedLoc(llvm::to_vector(
      llvm::map_range(members, [](MatMulNBitsOp m) { return m.getLoc(); })));

  SmallVector<Value> weights, scales, inits;
  SmallVector<Type> resultTypes;
  SmallVector<int64_t> ns;
  for (MatMulNBitsOp m : members) {
    auto resultType = cast<RankedTensorType>(m.getResult(0).getType());
    weights.push_back(m.getB());
    scales.push_back(m.getScales());
    inits.push_back(createInit(b, loc, first.getA(), resultType));
    resultTypes.push_back(resultType);
    ns.push_back(m.getN());
  }

  auto group = MatMulNBitsGroupOp::create(
      b, loc, resultTypes, first.getHandle(), first.getA(), weights, scales,
      inits, first.getK(), ns, first.getBits(), first.getBlockSize(),
      first.getAccuracyLevel());

  for (auto [m, result] : llvm::zip_equal(members, group.getResults())) {
    MatMulNBitsOp member = m;
    SmallVector<Value> oldInits(
        cast<DestinationStyleOpInterface>(member.getOperation())
            .getDpsInits());
    member.getResult(0).replaceAllUsesWith(result);
    member.erase();
    for (Value init : oldInits)
      eraseDeadChain(init);
  }
  ++NumGroups;
  NumGrouped += members.size();
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] grouped " << members.size()
                          << " projections at " << loc << "\n");
}

struct GroupMatMulNBitsPass
    : public impl::GroupMatMulNBitsPassBase<GroupMatMulNBitsPass> {
  void runOnOperation() override {
    DominanceInfo dom(getOperation());
    for (Block &block : getOperation().getBody()) {
      llvm::MapVector<GroupKey, SmallVector<MatMulNBitsOp>> candidates;
      for (auto op : block.getOps<MatMulNBitsOp>())
        if (isGroupable(op))
          candidates[keyOf(op)].push_back(op);

      for (auto &entry : candidates) {
        ArrayRef<MatMulNBitsOp> ops = entry.second;
        while (ops.size() >= 2) {
          SmallVector<MatMulNBitsOp> members{ops.front()};
          size_t used = 1;
          for (; used < ops.size() &&
                 members.size() < MatMulNBitsGroupOp::kMaxMembers;
               ++used)
            if (hoistOperandsAbove(ops[used], members.front(), dom))
              members.push_back(ops[used]);
          if (members.size() >= 2)
            emitGroup(members);
          ops = ops.drop_front(used);
        }
      }
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
