/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- BufferUtils.cpp - Shared buffer analysis utilities -----------------===//
//
// Implements the utilities declared in BufferUtils.h.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/BufferUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;

int64_t mlir::hip::getStaticByteSize(MemRefType type) {
  if (!type.hasStaticShape())
    return 0;
  int64_t totalBits =
      type.getNumElements() * type.getElementTypeBitWidth();
  return static_cast<int64_t>(llvm::divideCeil(totalBits, 8));
}

int64_t mlir::hip::alignUp(int64_t value, int64_t alignment) {
  assert(alignment > 0 && "alignment must be positive");
  return (value + alignment - 1) / alignment * alignment;
}

Value mlir::hip::emitAlignUp(OpBuilder &builder, Location loc, Value value,
                             int64_t alignment) {
  if (alignment <= 1)
    return value;
  Value alignM1 =
      arith::ConstantIndexOp::create(builder, loc, alignment - 1);
  Value alignConst =
      arith::ConstantIndexOp::create(builder, loc, alignment);
  Value sum = builder.createOrFold<arith::AddIOp>(loc, value, alignM1);
  Value divided = builder.createOrFold<arith::DivUIOp>(loc, sum, alignConst);
  return builder.createOrFold<arith::MulIOp>(loc, divided, alignConst);
}

unsigned mlir::hip::findLastAliasedUseIndex(
    Value allocResult, const BufferViewFlowAnalysis &aliasAnalysis,
    Block &block, const DenseMap<Operation *, unsigned> &opIndex,
    unsigned blockSize) {
  unsigned lastIdx = 0;
  for (Value alias : aliasAnalysis.resolve(allocResult)) {
    for (Operation *user : alias.getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;

      auto it = opIndex.find(user);
      unsigned userIdx;
      if (it != opIndex.end()) {
        userIdx = it->second;
      } else if (auto *ancestor = block.findAncestorOpInBlock(*user)) {
        userIdx = opIndex.lookup(ancestor);
      } else {
        userIdx = blockSize - 1;
      }
      lastIdx = std::max(lastIdx, userIdx);
    }
  }
  return lastIdx;
}

Operation *mlir::hip::findLastAliasedUser(
    Value allocResult, const BufferViewFlowAnalysis &aliasAnalysis,
    Block &entryBlock) {
  Operation *lastUser = allocResult.getDefiningOp();
  for (Value alias : aliasAnalysis.resolve(allocResult)) {
    for (Operation *user : alias.getUsers()) {
      Operation *resolved = user;
      if (resolved->getBlock() != &entryBlock) {
        resolved = entryBlock.findAncestorOpInBlock(*resolved);
        if (!resolved)
          continue;
      }
      if (lastUser->isBeforeInBlock(resolved))
        lastUser = resolved;
    }
  }
  return lastUser;
}

bool mlir::hip::isAliasInSet(Value root,
                             const BufferViewFlowAnalysis &aliasAnalysis,
                             const DenseSet<Value> &valueSet) {
  for (Value alias : aliasAnalysis.resolve(root))
    if (valueSet.contains(alias))
      return true;
  return false;
}
