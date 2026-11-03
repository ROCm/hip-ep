/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeHostScalars.cpp - Place host scalar buffers ------------===//
//
// Place small static buffers in runtime-owned mapped scratch before buffer
// reuse and device pooling. Each candidate gets a separate aligned region.
// Keep all payload operations and their order.
//
// Run after linalg-to-loops so host stores are visible. Run before
// OptimizeMemRefs: reuse can merge a candidate into a larger allocation that
// exceeds the size limit. Pooling would then put host stores in device memory.
// HIP-to-LLVM checks redundant readbacks separately.
//
// Before: %a = memref.alloc() : memref<1xi32>
//         memref.store %n, %a[%c0] : memref<1xi32>
// After:  %a = memref.view %scratch[%offset][]
//             : memref<?xi8> to memref<1xi32>
//         memref.store %n, %a[%c0] : memref<1xi32>
//
// Limits: at most 16 elements; context in argument zero; no unknown consumers.
// Float buffers need a HIP consumer. This bounded rule does not make all host
// accesses safe. Larger and unsupported allocations need separate handling.
//
// A tiny inline `memref.global` read by a hip op is the same kind of host
// storage. Canonicalization can fold shape arithmetic into `dense<1> :
// tensor<i64>` after `hip-externalize-constants`, and bufferization lowers
// that to a global in the host JIT module. A GPU kernel reading it page-faults,
// so the global is replaced by a scratch view the host fills with stores.
// Externalized globals stay device-uploaded. Scratch bytes follow the data
// layout stride, so a padded element such as i48 occupies its ABI slot.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/RuntimeAccessOpInterface.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "hip-materialize-host-scalars"

STATISTIC(NumAllocsMaterialized,
          "Number of memref.alloc redirected to host-mapped scratch");
STATISTIC(NumConstantsMaterialized,
          "Number of constant memref.get_global redirected to host scratch");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_MATERIALIZEHOSTSCALARSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Round \p x up to a multiple of \p align (align must be a power of two).
static int64_t roundUp(int64_t x, int64_t align) {
  return (x + align - 1) & ~(align - 1);
}

/// Preserve host-I/O placement through aliases and existing HIP consumers.
/// This is not a proof that arbitrary device-filled buffers can be mapped.
static bool classifyHostScalarUsers(Value memrefVal, bool &sawHostIO,
                                    bool &sawHipUser) {
  for (Operation *user : memrefVal.getUsers()) {
    if (isa<memref::StoreOp, memref::LoadOp>(user)) {
      sawHostIO = true;
      continue;
    }
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;
    // Keep the existing placement rule for HIP consumers of host-fed data.
    if (isa_and_nonnull<HipDialect>(user->getDialect())) {
      sawHipUser = true;
      continue;
    }
    // A reshape result aliases its data operand, not its shape operand.
    if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
      if (view.getViewSource() == memrefVal) {
        if (!classifyHostScalarUsers(view->getResult(0), sawHostIO, sawHipUser))
          return false;
      }
      continue;
    }
    if (isa<memref::CopyOp>(user))
      continue;

    return false;
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
    if (!isContiguousRowMajor(type.getShape(), strides))
      return false;
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
      // Query the backend contract. Memory effects alone do not establish
      // mapped-input support. Keep kernel destinations in device storage.
      auto access = dyn_cast<RuntimeAccessOpInterface>(user);
      if (access && access.supportsHostMappedInputs() &&
          user->getNumOperands() != 0 &&
          user->getOperand(0) == func.getArgument(0) &&
          type.getLayout().isIdentity()) {
        auto effects = dyn_cast<MemoryEffectOpInterface>(user);
        if (!effects)
          return false;
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

/// Return the initial value of the global \p getGlobal reads if it is a tiny
/// inline constant (an initial value, no `hip.external_data`) read by a hip op,
/// or null. Such a global lowers to storage in the host JIT module, which a GPU
/// kernel cannot dereference; see the file header.
static DenseElementsAttr getHostConstantCandidate(memref::GetGlobalOp op) {
  MemRefType type = op.getType();
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      type.getNumElements() > 16)
    return {};
  // The host fill is `arith.constant` + `memref.store`, and `arith.constant`
  // only produces signless integers.
  Type elem = type.getElementType();
  if (!elem.isSignlessIntOrIndex() && !isa<FloatType>(elem))
    return {};
  auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      op, op.getNameAttr());
  if (!global || global->hasAttr("hip.external_data"))
    return {};
  auto init = dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr());
  if (!init)
    return {};
  bool hasHostIO = false;
  bool hasHipUser = false;
  if (!classifyHostScalarUsers(op.getResult(), hasHostIO, hasHipUser) ||
      !hasHipUser)
    return {};
  return init;
}

static int64_t scratchBytes(MemRefType type) {
  // Match the default layout used by HIP-to-LLVM, including element padding.
  const DataLayout dataLayout;
  Type elemTy = type.getElementType();
  int64_t elemBytes =
      llvm::alignTo(dataLayout.getTypeSize(elemTy).getFixedValue(),
                    dataLayout.getTypeABIAlignment(elemTy));
  int64_t bytes = type.getNumElements() * elemBytes;
  return bytes == 0 ? 1 : bytes;
}

/// Appends every (user, aliased value) pair reaching \p buffer, directly or
/// through views of it.
static void
collectAliasUses(Value buffer,
                 SmallVectorImpl<std::pair<Operation *, Value>> &uses) {
  for (Operation *user : buffer.getUsers()) {
    uses.push_back({user, buffer});
    if (auto view = dyn_cast<ViewLikeOpInterface>(user))
      if (view.getViewSource() == buffer)
        collectAliasUses(view->getResult(0), uses);
  }
}

/// Host loads from \p allocOp that may run before an async copy into it lands.
/// memref.copy lowers to an async hipMemcpy on the stream, so a later
/// memref.load reads the scratch before the copy completes.
static SmallVector<memref::LoadOp>
findLoadsAfterAsyncWrites(memref::AllocOp allocOp) {
  SmallVector<std::pair<Operation *, Value>> uses;
  collectAliasUses(allocOp.getResult(), uses);
  Block *block = allocOp->getBlock();
  SmallVector<Operation *> writers;
  for (auto [user, val] : uses) {
    auto copy = dyn_cast<memref::CopyOp>(user);
    if (copy && copy.getTarget() == val)
      if (Operation *a = block->findAncestorOpInBlock(*user))
        writers.push_back(a);
  }
  SmallVector<memref::LoadOp> loads;
  for (auto [user, val] : uses) {
    auto load = dyn_cast<memref::LoadOp>(user);
    if (!load || !isa<IntegerType, FloatType>(load.getType()))
      continue;
    Operation *at = block->findAncestorOpInBlock(*load);
    if (!at)
      continue;
    if (llvm::any_of(writers, [&](Operation *w) {
          return w == at || w->isBeforeInBlock(at);
        }))
      loads.push_back(load);
  }
  return loads;
}

/// Replaces \p load with a stream-synchronized hip.readback_scalar of the same
/// element.
static void rewriteAsReadback(memref::LoadOp load, Value ctx) {
  OpBuilder b(load);
  Location loc = load.getLoc();
  Value src = load.getMemRef();
  int64_t rank = load.getMemRefType().getRank();
  if (rank > 0) {
    SmallVector<OpFoldResult> offsets(load.getIndices());
    SmallVector<OpFoldResult> ones(rank, b.getIndexAttr(1));
    Value elem = memref::SubViewOp::create(b, loc, src, offsets, ones, ones);
    src = memref::CollapseShapeOp::create(b, loc, elem,
                                          ArrayRef<ReassociationIndices>{});
  }
  Value v = ReadbackScalarOp::create(b, loc, load.getType(), ctx, src);
  load.replaceAllUsesWith(v);
  load.erase();
}

struct MaterializeHostScalarsPass
    : public impl::MaterializeHostScalarsPassBase<MaterializeHostScalarsPass> {

  void runOnOperation() override;
};

void MaterializeHostScalarsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  if (funcOp.getNumArguments() == 0 ||
      !isa<ContextType>(funcOp.getArgument(0).getType()))
    return;
  Value ctx = funcOp.getArgument(0);

  // Collect candidates before rewriting. Keep their traversal order.
  SmallVector<memref::AllocOp> candidates;
  funcOp.walk([&](memref::AllocOp op) {
    if (isHostScalarCandidate(op))
      candidates.push_back(op);
  });
  // Constant globals are laid out after the allocs so that adding them left
  // the alloc offsets unchanged.
  SmallVector<std::pair<memref::GetGlobalOp, DenseElementsAttr>> constants;
  funcOp.walk([&](memref::GetGlobalOp op) {
    if (DenseElementsAttr init = getHostConstantCandidate(op))
      constants.emplace_back(op, init);
  });
  if (candidates.empty() && constants.empty())
    return;

  constexpr int64_t kAlign = 64;
  SmallVector<int64_t> offsets;
  offsets.reserve(candidates.size());
  SmallVector<int64_t> constantOffsets;
  constantOffsets.reserve(constants.size());
  int64_t total = 0;
  auto place = [&](MemRefType ty) {
    int64_t off = roundUp(total, kAlign);
    total = off + scratchBytes(ty);
    return off;
  };
  for (memref::AllocOp allocOp : candidates)
    offsets.push_back(place(allocOp.getType()));
  for (auto &entry : constants)
    constantOffsets.push_back(place(entry.first.getType()));
  total = roundUp(total, kAlign);

  Block &entry = funcOp.getBody().front();
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entry);
  Location loc = funcOp.getLoc();
  Value totalSize = arith::ConstantIndexOp::create(builder, loc, total);
  auto scratchType =
      MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(8));
  Value scratch =
      GetHostScratchOp::create(builder, loc, scratchType, ctx, totalSize);

  // The runtime owns scratch. Replace allocations and erase their deallocs.
  for (auto [allocOp, offset] : llvm::zip(candidates, offsets)) {
    for (memref::LoadOp load : findLoadsAfterAsyncWrites(allocOp))
      rewriteAsReadback(load, ctx);

    builder.setInsertionPoint(allocOp);
    Value offsetVal =
        arith::ConstantIndexOp::create(builder, allocOp.getLoc(), offset);
    auto view = memref::ViewOp::create(builder, allocOp.getLoc(),
                                       allocOp.getType(), scratch, offsetVal,
                                       /*sizes=*/ValueRange{});

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

  // Replace each constant global read with a scratch view the host fills at
  // the same point.
  for (auto [entry, offset] : llvm::zip(constants, constantOffsets)) {
    auto [getGlobal, init] = entry;
    Location opLoc = getGlobal.getLoc();
    MemRefType type = getGlobal.getType();
    builder.setInsertionPoint(getGlobal);
    Value offsetVal = arith::ConstantIndexOp::create(builder, opLoc, offset);
    auto view = memref::ViewOp::create(builder, opLoc, type, scratch, offsetVal,
                                       /*sizes=*/ValueRange{});
    ArrayRef<int64_t> shape = type.getShape();
    auto values = init.getValues<Attribute>();
    for (int64_t linear = 0; linear < type.getNumElements(); ++linear) {
      Attribute elem = init.isSplat() ? init.getSplatValue<Attribute>()
                                      : *(values.begin() + linear);
      Value value =
          arith::ConstantOp::create(builder, opLoc, cast<TypedAttr>(elem));
      SmallVector<Value> indices(shape.size());
      int64_t rest = linear;
      for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
        indices[d] =
            arith::ConstantIndexOp::create(builder, opLoc, rest % shape[d]);
        rest /= shape[d];
      }
      memref::StoreOp::create(builder, opLoc, value, view, indices);
    }
    getGlobal.replaceAllUsesWith(view.getResult());
    getGlobal.erase();
    ++NumConstantsMaterialized;
    LLVM_DEBUG(llvm::dbgs() << "  Materialized constant " << view
                            << " at offset " << offset << "\n");
  }
}

} // namespace
} // namespace hip
} // namespace mlir
