/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeHostScalars.cpp - place small host-accessed buffers
//-------===//
//
// Place small static buffers in runtime-owned mapped scratch before buffer
// reuse and device pooling. Each candidate gets a separate aligned region.
// Keep all payload operations and their order.
//
// Host-written buffers must remain host-accessible. Buffer reuse can otherwise
// merge them into larger allocations that exceed this pass's size limit.
// The resulting host store would access device-pool memory.
//
// Follow aliases to find host loads, stores, and supported consumers. A small
// copy-filled shape buffer can also use scratch if it has an explicit readback
// and bounded aliases. This preserves its placement when early materialization
// prevents reuse with a later host-written buffer.
//
// Run after linalg-to-loops so host stores are visible. Run before
// OptimizeMemRefs and PoolAllocs so candidates still have separate allocations.
// This pass changes storage placement only. HIP-to-LLVM separately proves
// whether a readback and its synchronization can be removed.
//
// Limits: at most 16 elements; context in argument zero; no unknown consumers.
// Float buffers need a HIP consumer. This bounded rule does not make all host
// accesses safe. Larger and unsupported allocations need separate handling.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-materialize-host-scalars"

STATISTIC(NumAllocsMaterialized,
          "Number of memref.alloc redirected to host-mapped scratch");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_MATERIALIZEHOSTSCALARSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Round \p x up to a multiple of \p align (align must be a power of two).
static int64_t roundUp(int64_t x, int64_t align) {
  return (x + align - 1) & ~(align - 1);
}

/// Classify every transitive user of \p memrefVal and decide whether they are
/// all compatible with the backing buffer living in host-mapped scratch (see
/// the file header for the rationale behind each category). Sets \p sawHostIO
/// when a `memref.store`/`load` is found anywhere in the chain. Returns false
/// at the first user we cannot vouch for -- an op of an unknown dialect, or a
/// view whose own users escape.
///
/// View/descriptor ops are recognized generically via `ViewLikeOpInterface`
/// (reinterpret_cast, expand/collapse_shape, subview, view, cast, AND reshape)
/// instead of a hand-maintained `isa<>` list. They touch no memory themselves,
/// so the backing buffer is still a host-staged scalar; we recurse into the
/// view's result. The recursion is gated on `getViewSource() == memrefVal`:
/// this matters for `memref.reshape`, whose viewed source is the DATA operand,
/// not the shape operand -- when our buffer is the shape operand the reshape
/// result aliases the data (not us), so it is a terminal accept, not a recurse.
/// `memref.copy` is not view-like (it carries Read/Write memory effects); it is
/// a terminal host-mapping-safe use and does not itself flag host I/O.
static bool classifyHostScalarUsers(Value memrefVal, bool &sawHostIO,
                                    bool &sawHipUser) {
  for (Operation *user : memrefVal.getUsers()) {
    // Host I/O — the SEGV trigger we're staging away from the GPU pool.
    if (isa<memref::StoreOp, memref::LoadOp>(user)) {
      sawHostIO = true;
      continue;
    }
    // Metadata-only / lifetime users: harmless.
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;
    // hip.* consumers are host-mapping-safe (hipHostMallocMapped is
    // GPU-readable at the same VA on UMA targets).
    if (user->getDialect() && user->getDialect()->getNamespace() == "hip") {
      sawHipUser = true;
      continue;
    }
    // View/alias ops: recurse ONLY into the result that actually aliases this
    // buffer. getViewSource() pins which operand is the viewed one — critical
    // for memref.reshape, where our buffer may be the SHAPE operand (then the
    // result aliases the DATA, not us → terminal accept, not a recurse).
    if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
      if (view.getViewSource() == memrefVal) {
        if (!classifyHostScalarUsers(view->getResult(0), sawHostIO, sawHipUser))
          return false;
      }
      continue;
    }
    // memref.copy is not view-like; it's a terminal host-mapping-safe use
    // (Read src / Write dst) and does NOT itself flag host I/O.
    if (isa<memref::CopyOp>(user))
      continue;

    return false; // genuinely unknown user → reject candidate
  }
  return true;
}

// Keep a small copy-filled shape buffer in mapped storage when an explicit
// readback observes it. Preserve every copy and readback. This avoids a new
// device-pool slot when early placement separates it from a host-written
// buffer.
static bool isCopyReadbackCandidate(memref::AllocOp allocOp) {
  auto func = allocOp->getParentOfType<func::FuncOp>();
  auto rootType = allocOp.getType();
  if (!func || !llvm::hasSingleElement(func.getBody()) ||
      allocOp->getBlock() != &func.front() ||
      !rootType.getLayout().isIdentity() ||
      !rootType.getElementType().isIntOrIndex())
    return false;

  SmallVector<Value> pending{allocOp.getResult()};
  llvm::SmallPtrSet<Operation *, 16> visited;
  bool hasCopy = false, hasReadback = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    auto type = cast<MemRefType>(value.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    // These aliases retain the aligned base. Static contiguous layouts give
    // an exact interval in the original allocation. Reject offsets or extents
    // that could reach another scratch region.
    if (type.getElementType() != rootType.getElementType() ||
        !type.hasStaticShape() ||
        failed(type.getStridesAndOffset(strides, offset)) || offset < 0 ||
        offset > rootType.getNumElements() ||
        type.getNumElements() > rootType.getNumElements() - offset)
      return false;
    int64_t stride = 1;
    for (int64_t i = type.getRank(); i > 0; --i) {
      if (type.getDimSize(i - 1) != 1 && strides[i - 1] != stride)
        return false;
      stride *= type.getDimSize(i - 1);
    }
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != allocOp->getBlock())
        return false;
      if (isa<memref::SubViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::ReinterpretCastOp, memref::CastOp>(user)) {
        if (visited.insert(user).second)
          pending.push_back(user->getResult(0));
        continue;
      }
      if (isa<memref::DimOp>(user))
        continue;
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        // This bounded case uses scalar flat copies. The runtime accepts
        // mapped pointers and retains the descriptor offset on both sides.
        for (Value operand : copy->getOperands()) {
          auto operandType = cast<MemRefType>(operand.getType());
          SmallVector<int64_t> copyStrides;
          int64_t copyOffset;
          if (!operandType.hasStaticShape() ||
              operandType.getNumElements() != 1 ||
              failed(
                  operandType.getStridesAndOffset(copyStrides, copyOffset)) ||
              llvm::is_contained(copyStrides, ShapedType::kDynamic))
            return false;
        }
        hasCopy |= copy.getTarget() == value;
        continue;
      }
      if (isa<ReadbackScalarOp, ReadbackDimOp>(user)) {
        if (user->getOperand(0) != func.getArgument(0))
          return false;
        hasReadback = true;
        continue;
      }
      // These wrappers accept mapped input storage. Do not move kernel
      // destinations or extend this list without checking the runtime.
      if (isa<PadOp, SliceOp, AddOp, CastOp>(user) &&
          user->getOperand(0) == func.getArgument(0) &&
          type.getLayout().isIdentity()) {
        auto effects = cast<MemoryEffectOpInterface>(user);
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffectsOnValue(value, instances);
        if (llvm::all_of(instances, [](const auto &effect) {
              return isa<MemoryEffects::Read>(effect.getEffect());
            }))
          continue;
      }
      return false;
    }
  }
  return hasCopy && hasReadback;
}

// Keep the existing size limit. Host I/O or a bounded copy/readback chain
// must require the small buffer. Float buffers also need a HIP consumer.
static bool isHostScalarCandidate(memref::AllocOp allocOp) {
  MemRefType type = allocOp.getType();
  if (!type.hasStaticShape())
    return false;
  if (type.getNumElements() > 16)
    return false;
  Type elem = type.getElementType();
  bool isFloat = isa<FloatType>(elem);
  if (!elem.isIntOrIndex() && !isFloat)
    return false;

  bool hasHostIO = false;
  bool hasHipUser = false;
  if (!classifyHostScalarUsers(allocOp.getResult(), hasHostIO, hasHipUser))
    return false;
  return (hasHostIO && (!isFloat || hasHipUser)) ||
         isCopyReadbackCandidate(allocOp);
}

struct MaterializeHostScalarsPass
    : public impl::MaterializeHostScalarsPassBase<MaterializeHostScalarsPass> {

  void runOnOperation() override;
};

void MaterializeHostScalarsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  // Need !hip.context as arg 0 to call hip.get_host_scratch. If absent (e.g.
  // utility funcs or pre-context-arg passes), silently skip — we're a
  // best-effort mitigation, not a correctness requirement on every function.
  if (funcOp.getNumArguments() == 0 ||
      !isa<ContextType>(funcOp.getArgument(0).getType()))
    return;
  Value ctx = funcOp.getArgument(0);

  // Collect candidates.  The walk order here determines the per-candidate
  // scratch offsets computed below; any refactor that changes the traversal
  // (sorting, reverse-walk, parallel collection) changes the scratch layout
  // and therefore the LIT IR snapshots.  Keep `funcOp.walk` (post-order,
  // deterministic).
  SmallVector<memref::AllocOp> candidates;
  funcOp.walk([&](memref::AllocOp op) {
    if (isHostScalarCandidate(op))
      candidates.push_back(op);
  });
  if (candidates.empty())
    return;

  // Compute byte size per candidate and aligned offsets. 64-byte alignment
  // matches the runtime pool and is comfortably above the largest scalar
  // alignment; gives every candidate its own cache line.
  constexpr int64_t kAlign = 64;
  SmallVector<int64_t> offsets;
  offsets.reserve(candidates.size());
  int64_t total = 0;
  for (memref::AllocOp allocOp : candidates) {
    MemRefType ty = allocOp.getType();
    Type elemTy = ty.getElementType();
    unsigned bits = elemTy.isIndex() ? 64 : elemTy.getIntOrFloatBitWidth();
    int64_t elemBytes = static_cast<int64_t>((bits + 7) / 8);
    int64_t bytes = ty.getNumElements() * elemBytes;
    if (bytes == 0)
      bytes = 1; // rank-0 with i1 etc. — never zero-sized
    int64_t off = roundUp(total, kAlign);
    offsets.push_back(off);
    total = off + bytes;
  }
  total = roundUp(total, kAlign);

  // Emit hip.get_host_scratch(%ctx, %total) at function entry.
  Block &entry = funcOp.getBody().front();
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entry);
  Location loc = funcOp.getLoc();
  Value totalSize = arith::ConstantIndexOp::create(builder, loc, total);
  auto scratchType =
      MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(8));
  Value scratch =
      GetHostScratchOp::create(builder, loc, scratchType, ctx, totalSize);

  // Replace each candidate with a memref.view over the scratch buffer.
  // Deallocs of the candidate are erased — the scratch buffer is owned by
  // the runtime (released in hipdnn_ep_state_cleanup), like the pool.
  for (auto [allocOp, offset] : llvm::zip(candidates, offsets)) {
    builder.setInsertionPoint(allocOp);
    Value offsetVal =
        arith::ConstantIndexOp::create(builder, allocOp.getLoc(), offset);
    auto view = memref::ViewOp::create(builder, allocOp.getLoc(),
                                       allocOp.getType(), scratch, offsetVal,
                                       /*sizes=*/ValueRange{});

    // Erase deallocs first (they reference the alloc's result).
    SmallVector<memref::DeallocOp> deallocs;
    for (Operation *user : allocOp->getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        deallocs.push_back(d);
    for (auto d : deallocs)
      d.erase();

    allocOp.replaceAllUsesWith(view.getResult());
    allocOp.erase();
    ++NumAllocsMaterialized;
    LLVM_DEBUG(llvm::dbgs()
               << "  Materialized " << view << " at offset " << offset << "\n");
  }
}

} // namespace
} // namespace hip
} // namespace mlir
