/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_KERNELOUTLINE_H
#define HIP_DIALECT_TRANSFORMS_KERNELOUTLINE_H

#include "hip/Dialect/IR/HipDialect.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include <optional>
#include <string>
#include <utility>

namespace mlir::hip {

/// An inline hip.constant has a dense value that HIP-to-TOSA turns into
/// tosa.const. File- and memory-backed carriers are Pure and operand-free too,
/// but they name a byte range nothing has read, so they stay kernel arguments.
bool isInlineDenseConstant(Value v);

/// Rank-0 inline hip.constant used by fused pointwise ops (Relu zero, Clip
/// bounds, residual Mul scale). A rank-0 buffer argument has no index for
/// rock.transforms_to_ptr, so only the inline form is cloned.
bool isInlineScalarConstant(Value v);

/// The kernel ABI rocMLIR expects for a scalar: tensor<1xT> for a rank-0
/// value, every other type unchanged.
Type promoteRankZero(Type type);

bool isStaticTensor(Value v);

/// rock.blockwise_load and rock.blockwise_store take integers of at most 32
/// bits, so a wider integer tensor cannot be a kernel argument or result.
bool isKernelIOType(Type type);
bool isKernelIOValue(Value v);

/// A dequantize on the way in lets the kernel read the narrow type, and a
/// quantize feeding it folds the round trip into the load.
bool isFusablePrologue(Operation *op);

/// A view that the kernel may absorb on one of its inputs.
bool allowsInputView(Operation *op);

/// A view that the kernel may absorb past the anchor.
bool allowsForwardView(Operation *op);

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

/// Symbol for one outlined rocMLIR kernel: `mlir_<ops>_<n>`. `ops` is the
/// clone-order list. A `hip.rocmlir_elementwise` contributes its body ops, and
/// bookkeeping (empty, reshape, constants, poison, return) is left out.
/// `counter` ends one past the suffix that was used. A suffix already present
/// in `module` is skipped, so two copies of the same chain stay unique.
std::string rocMlirKernelName(ModuleOp module, ArrayRef<Operation *> ops,
                              int &counter);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_KERNELOUTLINE_H
