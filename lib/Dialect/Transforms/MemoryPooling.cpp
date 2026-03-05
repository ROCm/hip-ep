/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Memory Pooling Pass - Greedy strip-packing of memref.alloc into a single
// GPU pool obtained via hip.get_pool(%ctx).
//===----------------------------------------------------------------------===//
// Algorithm:
//   1. Collect all memref.alloc ops with static shapes in each function.
//   2. For each alloc find its last transitive use (liveness end).
//   3. Greedy best-fit strip packing (256-byte alignment):
//      - For each buffer (largest first), find the smallest gap that fits
//        among pool regions whose lifetimes don't overlap the current buffer,
//        or append beyond the current high-water mark.
//   4. Replace every memref.alloc with:
//        %pool = hip.get_pool(%ctx) : memref<?xi8, 1>   (once per function)
//        %buf  = memref.view %pool[%byte_offset][] : ... to memref<..., 1>
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::hip;

namespace {

// 256-byte alignment matches hipdnn-ep pool allocator
static constexpr uint64_t kPoolAlignment = 256;

static inline uint64_t alignUp(uint64_t v, uint64_t align) {
  return (v + align - 1) / align * align;
}

struct AllocInfo {
  memref::AllocOp op;
  uint64_t sizeBytes;
  unsigned defOrder;     // program-order index of the defining op
  unsigned lastUseOrder; // program-order index of the last transitive use
  uint64_t poolOffset;   // assigned by strip packing
};

// Return element size in bytes, or 0 for unsupported types.
static uint64_t elementSizeBytes(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return ft.getWidth() / 8;
  if (auto it = dyn_cast<IntegerType>(t))
    return it.getWidth() / 8;
  return 0;
}

// Compute static buffer size in bytes; returns 0 if dynamic or unsupported.
static uint64_t computeBufferBytes(MemRefType mrt) {
  if (!mrt.hasStaticShape())
    return 0;
  uint64_t elemSize = elementSizeBytes(mrt.getElementType());
  if (elemSize == 0)
    return 0;
  uint64_t nElems = 1;
  for (int64_t d : mrt.getShape())
    nElems *= static_cast<uint64_t>(d);
  return nElems * elemSize;
}

// Build a map: operation pointer -> program-order index within a function.
static DenseMap<Operation*, unsigned> buildOrderMap(func::FuncOp funcOp) {
  DenseMap<Operation*, unsigned> order;
  unsigned idx = 0;
  funcOp.walk([&](Operation* op) { order[op] = idx++; });
  return order;
}

// Find the highest program-order index of any operation that transitively uses
// `val` (i.e., the liveness end of the buffer).
static unsigned findLastUseOrder(Value val,
                                 const DenseMap<Operation*, unsigned>& order) {
  unsigned last = order.lookup(val.getDefiningOp());
  // BFS/DFS through use-def chains (handles memref.view, etc.)
  SmallVector<Value, 8> worklist = {val};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation* user : v.getUsers()) {
      auto it = order.find(user);
      if (it != order.end())
        last = std::max(last, it->second);
      for (Value res : user->getResults())
        worklist.push_back(res);
    }
  }
  return last;
}

// True if the live intervals of a and b overlap.
static bool intervalsOverlap(const AllocInfo& a, const AllocInfo& b) {
  return !(a.lastUseOrder < b.defOrder || b.lastUseOrder < a.defOrder);
}

// Greedy best-fit strip packing.  Assigns poolOffset for each AllocInfo.
// Returns the total pool size needed (before final alignment).
static uint64_t stripPack(SmallVector<AllocInfo>& allocs) {
  SmallVector<unsigned> sortedIdx;
  sortedIdx.reserve(allocs.size());
  for (unsigned i = 0; i < allocs.size(); ++i)
    sortedIdx.push_back(i);
  // Largest first for better packing density
  std::sort(sortedIdx.begin(), sortedIdx.end(), [&](unsigned a, unsigned b) {
    return allocs[a].sizeBytes > allocs[b].sizeBytes;
  });

  SmallVector<bool> assigned(allocs.size(), false);
  uint64_t hiWater = 0;

  for (unsigned idx : sortedIdx) {
    AllocInfo& cur = allocs[idx];

    // Candidate start offsets: boundaries of already-placed buffers + 0
    SmallVector<uint64_t> candidates = {0};
    for (unsigned j = 0; j < allocs.size(); ++j) {
      if (!assigned[j])
        continue;
      candidates.push_back(allocs[j].poolOffset);
      candidates.push_back(allocs[j].poolOffset + allocs[j].sizeBytes);
    }
    candidates.push_back(hiWater); // fallback: append
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());

    uint64_t bestOffset = alignUp(hiWater, kPoolAlignment);
    uint64_t bestGap = UINT64_MAX;
    bool found = false;

    for (uint64_t cand : candidates) {
      uint64_t aligned = alignUp(cand, kPoolAlignment);
      uint64_t end = aligned + cur.sizeBytes;

      bool conflict = false;
      for (unsigned j = 0; j < allocs.size(); ++j) {
        if (!assigned[j] || j == idx)
          continue;
        if (!intervalsOverlap(cur, allocs[j]))
          continue;
        uint64_t js = allocs[j].poolOffset;
        uint64_t je = js + allocs[j].sizeBytes;
        if (!(end <= js || je <= aligned)) {
          conflict = true;
          break;
        }
      }
      if (!conflict) {
        uint64_t gap = aligned - cand; // alignment padding wasted
        if (!found || gap < bestGap) {
          bestGap = gap;
          bestOffset = aligned;
          found = true;
        }
      }
    }

    cur.poolOffset = bestOffset;
    assigned[idx] = true;
    hiWater = std::max(hiWater, bestOffset + cur.sizeBytes);
  }

  return hiWater;
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class MemoryPoolingPass
    : public PassWrapper<MemoryPoolingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryPoolingPass)

  StringRef getArgument() const final { return "memory-pooling"; }
  StringRef getDescription() const final {
    return "Replace memref.alloc with memref.view into a GPU pool "
           "(hip.get_pool) using greedy best-fit strip packing";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (failed(processFunction(module, funcOp, builder)))
        return signalPassFailure();
    }
  }

private:
  LogicalResult processFunction(ModuleOp module, func::FuncOp funcOp,
                                OpBuilder& builder) {
    // Step 1: collect static memref.alloc ops
    SmallVector<AllocInfo> allocs;
    auto orderMap = buildOrderMap(funcOp);

    // Collect allocs that must NOT be pooled (must stay in AS 0):
    //   1. Allocs returned directly by the function.
    //   2. Allocs that are the source of a memref.copy targeting a function
    //      block argument — these are output buffers converted to out-params
    //      by buffer-results-to-out-params.  Pooling them would produce an
    //      AS 1 source for an AS 0 memcpy, which is a mixed address-space
    //      intrinsic that crashes on HIP targets without real GPU hardware.
    DenseSet<Operation*> unpoolableAllocs;
    funcOp.walk([&](func::ReturnOp retOp) {
      for (Value v : retOp.getOperands()) {
        if (auto allocOp = v.getDefiningOp<memref::AllocOp>())
          unpoolableAllocs.insert(allocOp.getOperation());
      }
    });
    funcOp.walk([&](memref::CopyOp copyOp) {
      // If the copy destination is a function block argument (an out-param),
      // the source alloc must remain AS 0.
      if (isa<BlockArgument>(copyOp.getTarget())) {
        if (auto allocOp = copyOp.getSource().getDefiningOp<memref::AllocOp>())
          unpoolableAllocs.insert(allocOp.getOperation());
      }
    });

    funcOp.walk([&](memref::AllocOp allocOp) {
      uint64_t sz = computeBufferBytes(allocOp.getType());
      if (sz == 0)
        return; // dynamic or unsupported — leave as-is

      // Skip output buffers that must remain in AS 0 (returned or out-params).
      if (unpoolableAllocs.count(allocOp.getOperation()))
        return;

      AllocInfo info;
      info.op = allocOp;
      info.sizeBytes = sz;
      info.defOrder = orderMap.lookup(allocOp.getOperation());
      info.lastUseOrder = findLastUseOrder(allocOp.getResult(), orderMap);
      info.poolOffset = 0;
      allocs.push_back(info);
    });

    if (allocs.empty())
      return success();

    // Step 2: require !hip.context as arg 0 (inserted by hip-add-context-arg)
    if (funcOp.getNumArguments() == 0 ||
        !isa<hip::ContextType>(funcOp.getArgument(0).getType())) {
      return funcOp.emitError(
          "[memory-pooling] function missing !hip.context as arg 0 — "
          "run hip-add-context-arg before memory-pooling");
    }
    Value ctxArg = funcOp.getArgument(0);

    // Step 3: assign pool offsets
    uint64_t rawPoolSize = stripPack(allocs);
    uint64_t poolSize = alignUp(rawPoolSize, kPoolAlignment);

    // Step 4: emit hip.get_pool once at function entry
    Block& entry = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entry);

    // Pool type: memref<?xi8, 1>  (1-D raw byte buffer in GPU address space)
    auto i8Ty = builder.getIntegerType(8);
    auto i64Ty = builder.getI64Type();
    auto poolType = MemRefType::get(
        {static_cast<int64_t>(ShapedType::kDynamic)}, i8Ty,
        MemRefLayoutAttrInterface{},
        IntegerAttr::get(i64Ty, 1) // address space 1 = GPU
    );
    Value poolBuf =
        builder.create<hip::GetPoolOp>(funcOp.getLoc(), poolType, ctxArg);

    // Step 5: replace each memref.alloc with memref.view of the pool
    for (AllocInfo info : allocs) {
      builder.setInsertionPoint(info.op);
      Location loc = info.op.getLoc();

      Value byteOffset = builder.create<arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(info.poolOffset));

      // View type: same shape/element as original, but GPU address space
      MemRefType origTy = info.op.getType();
      MemRefType viewTy =
          MemRefType::get(origTy.getShape(), origTy.getElementType(),
                          MemRefLayoutAttrInterface{},
                          IntegerAttr::get(i64Ty, 1));

      Value view = builder.create<memref::ViewOp>(loc, viewTy, poolBuf,
                                                  byteOffset, ValueRange{});
      info.op.getResult().replaceAllUsesWith(view);
      info.op.erase();
    }

    // Annotate module with final pool size for diagnostics
    module->setAttr("hipdnn.pool_size", builder.getI64IntegerAttr(poolSize));
    return success();
  }
};

} // namespace

namespace mlir {
namespace hip {

std::unique_ptr<Pass> createMemoryPoolingPass() {
  return std::make_unique<MemoryPoolingPass>();
}

} // namespace hip
} // namespace mlir
