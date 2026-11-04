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
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/RuntimeAccessOpInterface.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "hip-materialize-host-scalars"

STATISTIC(NumAllocsMaterialized,
          "Number of memref.alloc redirected to host-mapped scratch");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_MATERIALIZEHOSTSCALARSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

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

/// Small tensor constants a converter emits after hip-externalize-constants
/// (e.g. a folded class count fed to hip.mul) bufferize to a
/// `memref.get_global` of a host `.data` global, which a hip kernel cannot
/// read. Rewrite each one a hip op reads into a `memref.alloc` plus host
/// stores of its values, so it becomes an ordinary candidate below.
static void stageHipReadConstantGlobals(func::FuncOp funcOp) {
  auto module = funcOp->getParentOfType<ModuleOp>();
  if (!module)
    return;
  SmallVector<memref::GetGlobalOp> getGlobals;
  funcOp.walk([&](memref::GetGlobalOp op) { getGlobals.push_back(op); });
  for (memref::GetGlobalOp getGlobal : getGlobals) {
    MemRefType type = getGlobal.getType();
    if (!type.hasStaticShape() || type.getNumElements() > 16 ||
        !type.getLayout().isIdentity())
      continue;
    Type elem = type.getElementType();
    if (!elem.isIntOrIndex() && !isa<FloatType>(elem))
      continue;
    auto global =
        module.lookupSymbol<memref::GlobalOp>(getGlobal.getNameAttr());
    if (!global || !global.getConstant())
      continue;
    auto init =
        dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr());
    if (!init)
      continue;
    bool sawHostIO = false, sawHipUser = false;
    if (!classifyHostScalarUsers(getGlobal.getResult(), sawHostIO,
                                 sawHipUser) ||
        !sawHipUser)
      continue;

    OpBuilder builder(getGlobal);
    Location loc = getGlobal.getLoc();
    auto alloc = memref::AllocOp::create(builder, loc, type);
    ArrayRef<int64_t> shape = type.getShape();
    SmallVector<Attribute> values(init.getValues<Attribute>());
    for (int64_t linear = 0; linear < type.getNumElements(); ++linear) {
      SmallVector<Value> indices(shape.size());
      int64_t rem = linear;
      for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
        indices[d] =
            arith::ConstantIndexOp::create(builder, loc, rem % shape[d]);
        rem /= shape[d];
      }
      Attribute v =
          init.isSplat() ? init.getSplatValue<Attribute>() : values[linear];
      Value scalar =
          arith::ConstantOp::create(builder, loc, cast<TypedAttr>(v));
      memref::StoreOp::create(builder, loc, scalar, alloc, indices);
    }
    getGlobal.replaceAllUsesWith(alloc.getResult());
    getGlobal.erase();
  }
}

namespace {
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

  stageHipReadConstantGlobals(funcOp);

  // Collect candidates before rewriting. Keep their traversal order.
  SmallVector<memref::AllocOp> candidates;
  funcOp.walk([&](memref::AllocOp op) {
    if (isHostScalarCandidate(op))
      candidates.push_back(op);
  });
  if (candidates.empty())
    return;

  constexpr int64_t kAlign = 64;
  SmallVector<int64_t> offsets;
  offsets.reserve(candidates.size());
  int64_t total = 0;
  // Match the default layout used by HIP-to-LLVM, including element padding.
  const DataLayout dataLayout;
  for (memref::AllocOp allocOp : candidates) {
    MemRefType ty = allocOp.getType();
    Type elemTy = ty.getElementType();
    int64_t elemBytes =
        llvm::alignTo(dataLayout.getTypeSize(elemTy).getFixedValue(),
                      dataLayout.getTypeABIAlignment(elemTy));
    int64_t bytes = std::max(int64_t{1}, ty.getNumElements() * elemBytes);
    int64_t off = llvm::alignTo(total, kAlign);
    offsets.push_back(off);
    total = off + bytes;
  }
  total = llvm::alignTo(total, kAlign);

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
}

} // namespace
} // namespace hip
} // namespace mlir
