/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_KERNELOUTLINE_H
#define HIP_DIALECT_TRANSFORMS_KERNELOUTLINE_H

#include "hip/Dialect/IR/HipDialect.h"

#include <llvm/ADT/SetVector.h>
#include <mlir/IR/PatternMatch.h>

#include <optional>
#include <utility>

namespace mlir::hip {

/// Rank-0 inline hip.constant used by fused pointwise ops (Relu zero, Clip
/// bounds, residual Mul scale). File- and memory-backed carriers have no dense
/// attr for in-kernel tosa.const and stay kernel arguments.
bool isInlineScalarConstant(Value v);

/// The kernel ABI rocMLIR expects for a scalar: tensor<1xT> for a rank-0
/// value, every other type unchanged.
Type promoteRankZero(Type type);

bool isStaticTensor(Value v);

/// rock.blockwise_load and rock.blockwise_store take integers of at most 32
/// bits, so a wider integer tensor cannot be a kernel argument or result.
bool isKernelIOType(Type type);
bool isKernelIOValue(Value v);

/// A quantize on the way out lets the kernel write the narrow type.
bool isFusableQuantizeEpilogue(Operation *op);

/// A dequantize on the way in lets the kernel read the narrow type, and a
/// quantize feeding it folds the round trip into the load.
bool isFusablePrologue(Operation *op);

/// Which neighbours of an anchor may join its kernel, beyond the forward
/// pointwise chain every anchor takes.
struct FusionPolicy {
  // ViewKind bits the input absorption may take.
  unsigned inputViewKinds = 0;
  // ViewKind bits the forward chain past the anchor may take.
  unsigned forwardViewKinds = 0;
  // A unary pointwise producer of an input root that allows one.
  bool activationPrologue = false;
  // A quantize on the forward chain.
  bool quantizeEpilogue = false;
  // The forward chain may pass a value that is also read outside the kernel;
  // the kernel then returns that value as an extra result.
  bool multiOutput = false;

  bool allowsEpilogue(Operation *op) const;
  bool allowsInputView(Operation *op) const;
  bool allowsForwardView(Operation *op) const;
  bool absorbsInputs() const {
    return inputViewKinds != 0 || activationPrologue;
  }
};

FusionPolicy getFusionPolicy(bool fuseViews);

/// Where input absorption starts, with whether each root may take a pointwise
/// prologue.
SmallVector<std::pair<Value, bool>> getInputRoots(Operation *anchor);

/// Anchors that own a forward chain once they are in the graph.
bool isAnchorOp(Operation *op);

/// Anchors whose result is computed in integers, so the dequantize, pointwise,
/// and quantize ops after it can run in its epilogue.
bool isQuantizedAnchor(Operation *op);

/// A rank-0 value the kernel can recompute from inline scalar constants: such
/// a constant, or a dequantize or reciprocal of them.
bool isConstantScalarExpr(Value v);

/// The ops isConstantScalarExpr walked, which the kernel clones.
void collectConstantScalarExpr(Value v, SetVector<Operation *> &ops);

/// The values an absorbed producer reads as data, or nullopt for an expand
/// whose shape is computed at run time.
std::optional<SmallVector<Value>> getDataInputs(Operation *producer,
                                                Value context);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_KERNELOUTLINE_H
