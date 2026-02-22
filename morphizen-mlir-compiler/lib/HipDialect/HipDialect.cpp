/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hip;
using namespace mlir::bufferization;

#include "HipDialect.cpp.inc"

void HipDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "HipTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "HipOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AllocationOpInterface for Hip_AllocOp
//===----------------------------------------------------------------------===//

std::optional<Operation*> mlir::hip::AllocOp::buildDealloc(OpBuilder& builder,
                                                           Value alloc) {
  // Extract handle from the alloc operation
  auto allocOp = alloc.getDefiningOp<hip::AllocOp>();
  if (!allocOp)
    return std::nullopt;

  return builder
      .create<hip::FreeOp>(
          alloc.getLoc(),
          allocOp.getHandle(), // Same context from the alloc operation
          alloc                // Buffer to free
          )
      .getOperation();
}

std::optional<Value> mlir::hip::AllocOp::buildClone(OpBuilder& builder,
                                                    Value alloc) {
  // GPU buffer cloning is complex, let MLIR handle via explicit copies
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MemoryEffectsOpInterface Implementations
//===----------------------------------------------------------------------===//

void mlir::hip::AllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // hip.alloc allocates GPU memory
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::FreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // hip.free deallocates GPU memory
  effects.emplace_back(MemoryEffects::Free::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ConvOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Read inputs (conservative: assumes reads from memory)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  // Write output
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::GemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Read inputs and write output (result is read-write due to beta != 0)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::MaxPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::AvgPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::CopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Read from source, write to destination
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface Implementations
//===----------------------------------------------------------------------===//

MutableOperandRange mlir::hip::ConvOp::getDpsInitsMutable() {
  return getOutputMutable(); // Last operand is destination
}

MutableOperandRange mlir::hip::ReluOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::GemmOp::getDpsInitsMutable() {
  return getResultMutable();
}

MutableOperandRange mlir::hip::MaxPoolOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::AvgPoolOp::getDpsInitsMutable() {
  return getOutputMutable();
}

//===----------------------------------------------------------------------===//
// hip.copy Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Eliminate self-copy: hip.copy(%ctx, %buf, %buf) → remove
struct EliminateSelfCopy : public OpRewritePattern<hip::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hip::CopyOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getSource() == op.getDestination()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

/// Eliminate copy after single-use DPS write
/// Pattern: %temp = hip.alloc; hip.operation(..., %temp); hip.copy(%temp, %out)
/// Result: hip.operation(..., %out); (temp allocation becomes dead)
struct EliminateCopyAfterDPSWrite : public OpRewritePattern<hip::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hip::CopyOp copyOp,
                                PatternRewriter& rewriter) const override {
    Value src = copyOp.getSource();
    Value dst = copyOp.getDestination();

    // Source must come from hip.alloc
    auto allocOp = src.getDefiningOp<hip::AllocOp>();
    if (!allocOp)
      return failure();

    // Find DPS operation writing to source
    Operation* writer = nullptr;
    for (Operation* user : src.getUsers()) {
      if (user == copyOp)
        continue;

      // Check if operation implements DestinationStyleOpInterface
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
        // Check if this operation writes to src
        for (OpOperand& init : dpsOp.getDpsInitsMutable()) {
          if (init.get() == src) {
            if (writer)
              return failure(); // Multiple writers, can't optimize
            writer = user;
          }
        }
      }
    }

    if (!writer)
      return failure();

    // Check single use: only writer + copyOp use the buffer
    if (!llvm::hasSingleElement(llvm::make_filter_range(
            src.getUsers(), [&](Operation* user) { return user != copyOp; })))
      return failure();

    // Redirect writer to destination
    for (OpOperand& init :
         cast<DestinationStyleOpInterface>(writer).getDpsInitsMutable()) {
      if (init.get() == src) {
        rewriter.startOpModification(writer);
        init.set(dst);
        rewriter.finalizeOpModification(writer);
      }
    }

    // Remove copy (buffer becomes dead, cleaned by DCE)
    rewriter.eraseOp(copyOp);
    return success();
  }
};
} // namespace

void mlir::hip::CopyOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                    MLIRContext* context) {
  results.add<EliminateSelfCopy, EliminateCopyAfterDPSWrite>(context);
}

void mlir::hip::GetConstantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Allocates view to constant memory (read-only effect)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

// Type and op class implementations (parse/print/verify, TypeIDs)
#define GET_TYPEDEF_CLASSES
#include "HipTypes.cpp.inc"

#define GET_OP_CLASSES
#include "HipOps.cpp.inc"
