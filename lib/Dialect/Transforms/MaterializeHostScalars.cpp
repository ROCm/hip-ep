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
// Run after linalg-to-loops so host stores are visible. Run before
// OptimizeMemRefs: reuse can merge a candidate into a larger allocation that
// exceeds the size limit. Pooling would then put host stores in device memory.
// HIP-to-LLVM checks redundant readbacks separately.
//
// Limits: at most 16 elements; context in argument zero; no unknown consumers.
// Float buffers need a HIP consumer. This bounded rule does not make all host
// accesses safe. Larger and unsupported allocations need separate handling.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace {

/// Follow aliases and reject users that do not support mapped storage.
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
    if (user->getDialect() && user->getDialect()->getNamespace() == "hip") {
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
static bool
isCopyReadbackCandidate(memref::AllocOp allocOp,
                        const DenseSet<OperationName> &mappedInputOps) {
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
      // Query the backend contract. Memory effects alone do not establish
      // mapped-input support. Keep kernel destinations in device storage.
      if (mappedInputOps.contains(user->getName()) &&
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

// Keep the existing size limit. Host I/O or a bounded copy/readback chain
// must require the small buffer. Float buffers also need a HIP consumer.
static bool
isHostScalarCandidate(memref::AllocOp allocOp,
                      const DenseSet<OperationName> &mappedInputOps) {
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
         isCopyReadbackCandidate(allocOp, mappedInputOps);
}

struct MaterializeHostScalarsPass
    : public impl::MaterializeHostScalarsPassBase<MaterializeHostScalarsPass> {

  LogicalResult initialize(MLIRContext *context) override {
    mappedInputOps = getHostMappedInputOps(context);
    return success();
  }

  void runOnOperation() override;

private:
  DenseSet<OperationName> mappedInputOps;
};

void MaterializeHostScalarsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  // Scratch belongs to the runtime context in argument zero.
  if (funcOp.getNumArguments() == 0 ||
      !isa<ContextType>(funcOp.getArgument(0).getType()))
    return;
  Value ctx = funcOp.getArgument(0);

  // Collect candidates before rewriting. Keep their traversal order.
  SmallVector<memref::AllocOp> candidates;
  funcOp.walk([&](memref::AllocOp op) {
    if (isHostScalarCandidate(op, mappedInputOps))
      candidates.push_back(op);
  });
  if (candidates.empty())
    return;

  // Give each candidate a separate aligned region.
  constexpr int64_t kAlign = 64;
  SmallVector<int64_t> offsets;
  offsets.reserve(candidates.size());
  int64_t total = 0;
  for (memref::AllocOp allocOp : candidates) {
    MemRefType ty = allocOp.getType();
    Type elemTy = ty.getElementType();
    unsigned bits = elemTy.isIndex() ? 64 : elemTy.getIntOrFloatBitWidth();
    int64_t elemBytes = static_cast<int64_t>((bits + 7) / 8);
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
