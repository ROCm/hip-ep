/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlProvenance.h - What the captured graph said about a value ----===//
//
// GGML tensors share memory: a view does not copy, it points into another
// tensor's buffer. That sharing is in the operations -- a view names its source
// and a destination-style write names what it writes -- so these accessors walk
// it rather than read a recorded copy of it. Provenance attributes are read
// from the defining operation or the function's entry argument. Layouts,
// including extents, strides, and offsets, are read through GgmlLayout.h.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLPROVENANCE_H
#define HIP_DIALECT_GGML_IR_GGMLPROVENANCE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <optional>

namespace mlir {
namespace ggml {

/// The provenance a value carries, whether it is produced by an operation or
/// enters through an entry argument.
inline Attribute getProvenanceAttr(Value value, StringRef name) {
  if (Operation *producer = value.getDefiningOp()) {
    // A result of a fused operation other than the one its attributes
    // describe may carry its own (`name#N`).
    if (producer->getNumResults() > 1)
      if (Attribute own = producer->getAttr(
              (name + "#" + Twine(cast<OpResult>(value).getResultNumber()))
                  .str()))
        return own;
    return producer->getAttr(name);
  }
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return {};
  auto function =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || function.getBody().empty() ||
      argument.getOwner() != &function.getBody().front()) {
    return {};
  }
  return function.getArgAttr(argument.getArgNumber(), name);
}

/// The value this one was handed its bytes by, or null when it owns them.
///
/// GGML tensors share memory in two ways, and both are edges in this graph. A
/// view does not copy: it reads an operand's bytes through different extents,
/// strides, or a starting offset, which is what `ViewLikeOpInterface` says.
/// `ggml.set_rows` replaces part of its destination and hands the buffer back,
/// which is what makes it destination-style.
inline Value getStorageSource(Value value) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return {};
  if (auto view = dyn_cast<ViewLikeOpInterface>(producer))
    return view.getViewSource();
  auto destination = dyn_cast<DestinationStyleOpInterface>(producer);
  if (!destination || destination.getNumDpsInits() != 1)
    return {};
  return destination.getDpsInitOperand(0)->get();
}

/// The value that owns the bytes this one occupies, or null for a cycle.
///
/// Following the edges above reaches the value that allocated them -- an entry
/// argument, or an operation that produced fresh storage -- and that value is
/// the buffer's name. Two values share a buffer exactly when this returns the
/// same non-null value for both.
inline Value getStorageRoot(Value value) {
  if (!value)
    return {};
  // Floyd's traversal detects malformed cycles without limiting valid chains.
  Value slow = value;
  Value fast = value;
  while (true) {
    for (unsigned step = 0; step < 2; ++step) {
      Value source = getStorageSource(fast);
      if (!source)
        return fast;
      fast = source;
    }
    slow = getStorageSource(slow);
    if (slow == fast)
      return {};
  }
}

/// Whether this value owns its buffer outright rather than viewing another's.
inline bool isStorageRoot(Value value) {
  return value && getStorageRoot(value) == value;
}

/// The known value whose buffer this one views. No alias root does not prove
/// ownership; use isStorageRoot for that.
inline std::optional<Value> getAliasRoot(Value value) {
  Value root = getStorageRoot(value);
  if (!root || root == value)
    return std::nullopt;
  return root;
}

/// Whether two values may share storage. Unknown roots cannot prove separation.
inline bool haveSameStorage(Value lhs, Value rhs) {
  Value lhsRoot = getStorageRoot(lhs);
  Value rhsRoot = getStorageRoot(rhs);
  return !lhsRoot || !rhsRoot || lhsRoot == rhsRoot;
}

/// The buffer an operation's result is written to.
inline Value getWrittenStorage(Operation *op) {
  if (op->getNumResults() != 1)
    return {};
  return getStorageRoot(op->getResult(0));
}

} // namespace ggml
} // namespace mlir

#endif // HIP_DIALECT_GGML_IR_GGMLPROVENANCE_H
