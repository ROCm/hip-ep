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

int64_t mlir::hip::getElementByteWidth(MemRefType type) {
  return static_cast<int64_t>(
      llvm::divideCeil(type.getElementTypeBitWidth(), 8));
}

FailureOr<int64_t> mlir::hip::getStaticByteSize(MemRefType type) {
  if (!type.hasStaticShape())
    return failure();

  int64_t byteSize = getElementByteWidth(type);
  for (int64_t dim : type.getShape())
    if (llvm::MulOverflow(byteSize, dim, byteSize))
      return failure();
  return byteSize;
}

Value mlir::hip::emitAlignUp(OpBuilder &builder, Location loc, Value value,
                             int64_t alignment) {
  if (alignment <= 1)
    return value;
  Value alignM1 = arith::ConstantIndexOp::create(builder, loc, alignment - 1);
  Value alignConst = arith::ConstantIndexOp::create(builder, loc, alignment);
  Value sum = builder.createOrFold<arith::AddIOp>(loc, value, alignM1);
  Value divided = builder.createOrFold<arith::DivUIOp>(loc, sum, alignConst);
  return builder.createOrFold<arith::MulIOp>(loc, divided, alignConst);
}

unsigned mlir::hip::findLastAliasedUseIndex(
    Value allocResult, const BufferViewFlowAnalysis &aliasAnalysis,
    Block &block, const DenseMap<Operation *, unsigned> &opIndex,
    unsigned blockSize) {
  unsigned lastIdx = 0;
  // resolve() returns all *forward* (downstream) aliases: the allocResult
  // itself plus any values derived from it via view-like ops (memref.view,
  // memref.subview, memref.cast, etc.).  We must consider users of ALL
  // aliases to get the true last-use index.
  for (Value alias : aliasAnalysis.resolve(allocResult)) {
    for (Operation *user : alias.getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;

      auto it = opIndex.find(user);
      unsigned userIdx;
      if (it != opIndex.end()) {
        // User is directly in the entry block.
        userIdx = it->second;
      } else if (auto *ancestor = block.findAncestorOpInBlock(*user)) {
        // User is inside a nested region (e.g. scf.for body); attribute
        // its index to the enclosing op in the entry block.
        userIdx = opIndex.lookup(ancestor);
      } else {
        // User is unreachable from this block; conservatively assume last.
        userIdx = blockSize - 1;
      }
      lastIdx = std::max(lastIdx, userIdx);
    }
  }
  return lastIdx;
}

Operation *
mlir::hip::findLastAliasedUser(Value allocResult,
                               const BufferViewFlowAnalysis &aliasAnalysis,
                               Block &entryBlock) {
  // Start from the defining op so we always have a valid baseline, even if
  // the alloc has no users at all (e.g., dead code not yet cleaned up).
  Operation *lastUser = allocResult.getDefiningOp();
  // resolve() returns all forward aliases -- see findLastAliasedUseIndex.
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
  // Check whether root or any of its forward aliases (views, casts, etc.)
  // appears in valueSet.  Used by LowerAllocs to skip hip.free for returned
  // buffers even when a derived view (not the alloc itself) is returned.
  for (Value alias : aliasAnalysis.resolve(root))
    if (valueSet.contains(alias))
      return true;
  return false;
}
