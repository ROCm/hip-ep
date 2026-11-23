/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_ROCMLIRFUSIONPREDICATES_H
#define HIP_DIALECT_TRANSFORMS_ROCMLIRFUSIONPREDICATES_H

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"
#include "hip/Dialect/IR/HipDialect.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/TypeUtilities.h>

#include <optional>

namespace mlir::hip {

// Elementwise ops HipToTosa lowers to TOSA elementwise ops, which rocMLIR
// fuses into the anchor's epilogue or prologue.
inline bool isFusablePointwise(Operation *op) {
  return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp, TanhOp,
                         SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                         LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                         AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                         RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                         SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp>(op);
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

// A float gemm with a few output columns and a long reduction has too few
// output tiles to fill the GPU; rocMLIR then splits the reduction across
// workgroups that atomically add partial sums (split-K). It offers split-K
// only when every epilogue op is linear and the result is a float type with
// atomic add, so such an anchor takes no quantize, clamp or other nonlinear
// epilogue op: one would cost it every split-K configuration.
inline constexpr int64_t kSplitKMaxColumns = 64;
inline constexpr int64_t kSplitKMinDepthPerColumn = 16;

inline bool reliesOnSplitK(Operation *anchor) {
  Value a;
  bool transA = false;
  if (auto matmul = dyn_cast<MatmulOp>(anchor)) {
    a = matmul.getA();
    transA = matmul.getTransA() != 0;
  } else if (auto gemm = dyn_cast<GemmOp>(anchor)) {
    a = gemm.getInputA();
    transA = gemm.getTransA() != 0;
  } else {
    return false;
  }
  auto aType = dyn_cast<ShapedType>(a.getType());
  auto result = dyn_cast<ShapedType>(anchor->getResult(0).getType());
  if (!aType || !aType.hasStaticShape() || aType.getRank() < 2 || !result ||
      !result.hasStaticShape() || !isa<FloatType>(result.getElementType()))
    return false;
  int64_t depth = aType.getDimSize(aType.getRank() - (transA ? 2 : 1));
  int64_t columns = result.getShape().back();
  return columns <= kSplitKMaxColumns &&
         depth >= kSplitKMinDepthPerColumn * columns;
}

// The epilogue ops that rocMLIR keeps under split-K, where each split applies
// them to its partial sum: an add or subtract, a negate, a float-to-float
// cast, and a multiply or divide that reads the kernel value once.
inline bool isSplitKEpilogueOp(Operation &op,
                               const DenseSet<Value> &fromKernel) {
  if (isa<AddOp, SubOp, NegOp>(op))
    return true;
  if (isa<MulOp, DivOp>(op))
    return llvm::count_if(op.getOperands(),
                          [&](Value v) { return fromKernel.contains(v); }) == 1;
  if (auto cast = dyn_cast<CastOp>(op))
    return isa<FloatType>(getElementTypeOrSelf(cast.getInput().getType())) &&
           isa<FloatType>(getElementTypeOrSelf(cast->getResult(0).getType()));
  return false;
}

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_ROCMLIRFUSIONPREDICATES_H
