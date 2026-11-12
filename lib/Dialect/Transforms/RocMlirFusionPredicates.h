/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_ROCMLIRFUSIONPREDICATES_H
#define HIP_DIALECT_TRANSFORMS_ROCMLIRFUSIONPREDICATES_H

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"
#include "hip/Dialect/IR/HipDialect.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <optional>

namespace mlir::hip {

// A cast is fused only when every float it touches is f16, bf16, or f32.
// An f64 cast stays a HIP cast, so f64 pointwise ops after it are not pulled
// into the rocMLIR kernel either. rocMLIR's blockwise load rejects f64.
inline bool isRocMlirCastEndpoint(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  Type element = shaped ? shaped.getElementType() : type;
  if (!isa<FloatType>(element))
    return true;
  return element.isF16() || element.isBF16() || element.isF32();
}

// Elementwise ops HipToTosa lowers to TOSA elementwise ops, which rocMLIR
// fuses into the anchor's epilogue or prologue.
inline bool isFusablePointwise(Operation *op) {
  if (auto cast = dyn_cast_if_present<CastOp>(op))
    return llvm::all_of(cast->getOperandTypes(), isRocMlirCastEndpoint) &&
           llvm::all_of(cast->getResultTypes(), isRocMlirCastEndpoint);
  return isa_and_present<
      MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp, TanhOp, SoftplusOp, GeluOp,
      BiasGeluOp, FastGeluOp, LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
      AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp, RoundOp, AtanOp, FloorOp,
      ExpOp, LogOp, AbsOp, NegOp, SubOp, LessOp, SignOp, ModOp, WhereOp>(op);
}

// View ops that rocMLIR folds into rock.transform maps instead of running as a
// copy. The legality is HipToTosa's own, so an absorbed view always lowers.
enum class ViewKind : unsigned {
  Transpose = 1u << 0,
  Reshape = 1u << 1,
  Slice = 1u << 2,
  Broadcast = 1u << 3,
};

inline std::optional<ViewKind> getFusableViewKind(Operation *op) {
  if (isa_and_present<TransposeOp>(op))
    return ViewKind::Transpose;
  if (isTosaLowerableReshape(op))
    return ViewKind::Reshape;
  if (auto slice = dyn_cast_if_present<tensor::ExtractSliceOp>(op);
      slice && isTosaLowerableSlice(slice))
    return ViewKind::Slice;
  if (auto expand = dyn_cast_if_present<ExpandOp>(op);
      expand && isTosaLowerableExpand(expand))
    return ViewKind::Broadcast;
  return std::nullopt;
}

// Whether `value` is computed, through ops outside the kernel, from a chain
// op. Only ops between the anchor and the reader can be on such a path, so
// the walk stops at anything defined before the anchor.
inline bool dependsOnChain(Value value, Operation *anchor,
                           const SetVector<Operation *> &chain) {
  Block *block = anchor->getBlock();
  SmallVector<Value> worklist = {value};
  DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (def)
      def = block->findAncestorOpInBlock(*def);
    if (!def || def->isBeforeInBlock(anchor))
      continue;
    if (chain.contains(def))
      return true;
    if (!visited.insert(def).second)
      continue;
    // A region can read a chain value without passing it as an operand.
    def->walk([&](Operation *nested) {
      llvm::append_range(worklist, nested->getOperands());
    });
  }
  return false;
}

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_ROCMLIRFUSIONPREDICATES_H
