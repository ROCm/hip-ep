/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "KernelOutline.h"
#include "QLinearPlan.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir::hip {

// Bookkeeping cloned so the kernel is IsolatedFromAbove is not part of the
// name. A return is not either: gemm-gemm walks a finished kernel body.
static bool isRocMlirKernelNameOp(Operation *op) {
  return !isa<tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
              arith::ConstantOp, ub::PoisonOp, ConstantOp, func::ReturnOp>(op);
}

std::string rocMlirKernelName(ArrayRef<Operation *> ops, int index) {
  std::string name = "mlir";
  for (Operation *op : ops) {
    if (!isRocMlirKernelNameOp(op))
      continue;
    name.push_back('_');
    name += op->getName().stripDialect().str();
  }
  name.push_back('_');
  name += std::to_string(index);
  return name;
}

bool isInlineScalarConstant(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 0)
    return false;
  auto constant = v.getDefiningOp<ConstantOp>();
  return constant && constant.getValueAttr();
}

// MIGraphX gives a scalar the shape {1}, and rock-flatten-tosa-func-args
// leaves rank-1 boundaries alone. A rank-0 argument reaches
// rock.transforms_to_ptr with no coordinate to linearize (`affine_map -> ()`),
// which fails as "Transforms are not well formed"; tensor<1xT> still carries
// the single index 0.
Type promoteRankZero(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getRank() != 0)
    return type;
  return RankedTensorType::get({1}, tensorType.getElementType(),
                               tensorType.getEncoding());
}

bool isStaticTensor(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  return type && type.hasStaticShape();
}

static constexpr unsigned kMaxKernelIntegerWidth = 32;

bool isKernelIOType(Type type) {
  auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
  return !intType || intType.getWidth() <= kMaxKernelIntegerWidth;
}

bool isKernelIOValue(Value v) { return isKernelIOType(v.getType()); }

bool isFusablePrologue(Operation *op) {
  return isFusablePointwise(op) || isTosaLowerableQdq(op);
}

// rocMLIR reads every kernel input through rock.transform maps, so a view on
// one becomes an index remap instead of a separate copy: a transpose next to
// a convolution folds into its input_layout / output_layout, and a reshape,
// slice, or broadcast on any input folds into that input's map.
//
// Past the anchor, rocMLIR has to invert the view to map each output element
// back to the GEMM tile that computes it. A transpose or reshape inverts; a
// broadcast does not ("cannot regularize"). A slice would compute output that
// the kernel then drops, so it stays outside too.
//
// A value on the chain that something outside also reads is written out as
// one more kernel result instead of ending the chain: it is stored either way,
// and the epilogue no longer has to load it back.
bool allowsInputView(Operation *op) {
  return getFusableViewKind(op).has_value();
}

bool allowsForwardView(Operation *op) {
  std::optional<ViewKind> kind = getFusableViewKind(op);
  return kind == ViewKind::Transpose || kind == ViewKind::Reshape;
}

// A conv's filter takes no pointwise prologue: an elementwise op on it would
// rerun in every workgroup that loads it. Both matmul operands may, since in
// attention both are activations. A gemm's C is added once per output
// element, so it takes views only.
SmallVector<std::pair<Value, bool>> getInputRoots(Operation *anchor) {
  SmallVector<std::pair<Value, bool>> roots;
  if (auto conv = dyn_cast<ConvOp>(anchor)) {
    roots.push_back({conv.getInput(), true});
    roots.push_back({conv.getWeights(), false});
    if (Value bias = conv.getBias())
      roots.push_back({bias, false});
  } else if (auto matmul = dyn_cast<MatmulOp>(anchor)) {
    roots.push_back({matmul.getA(), true});
    roots.push_back({matmul.getB(), true});
  } else if (auto gemm = dyn_cast<GemmOp>(anchor)) {
    roots.push_back({gemm.getInputA(), true});
    roots.push_back({gemm.getInputB(), true});
    if (Value c = gemm.getInputC())
      roots.push_back({c, false});
  } else if (auto conv = dyn_cast<QuantConvOp>(anchor)) {
    roots.push_back({conv.getInput(), true});
    roots.push_back({conv.getWeights(), false});
  } else if (auto conv = dyn_cast<QLinearConvOp>(anchor)) {
    roots.push_back({conv.getInput(), true});
    roots.push_back({conv.getWeights(), false});
  } else if (auto matmul = dyn_cast<QuantMatmulOp>(anchor)) {
    roots.push_back({matmul.getA(), true});
    roots.push_back({matmul.getB(), false});
  } else if (auto matmul = dyn_cast<QLinearMatMulOp>(anchor)) {
    roots.push_back({matmul.getA(), true});
    roots.push_back({matmul.getB(), false});
  }
  return roots;
}

bool isAnchorOp(Operation *op) {
  return isa<MatmulOp, ConvOp, GemmOp, QuantConvOp, QuantMatmulOp>(op);
}

bool isQuantizedAnchor(Operation *op) {
  return isa<QuantConvOp, QuantMatmulOp, QLinearConvOp, QLinearMatMulOp>(op) ||
         (isa<ConvOp>(op) && isPlannedQLinear(op));
}

// A QLinearAdd's constant operand is a dequantize of inline scalars, and a
// Div's divisor a reciprocal of one.
bool isConstantScalarExpr(Value v) {
  if (isInlineScalarConstant(v))
    return true;
  Operation *def = v.getDefiningOp();
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!def || !type || type.getRank() != 0)
    return false;
  bool isDequantize = isa<DequantizeLinearOp>(def) && isTosaLowerableQdq(def);
  if (!isDequantize && !isa<ReciprocalOp>(def))
    return false;
  return llvm::all_of(
      cast<DestinationStyleOpInterface>(def).getDpsInputs(), [](Value input) {
        return isa<ContextType>(input.getType()) || isConstantScalarExpr(input);
      });
}

void collectConstantScalarExpr(Value v, SetVector<Operation *> &ops) {
  Operation *def = v.getDefiningOp();
  if (!ops.insert(def) || isInlineScalarConstant(v))
    return;
  for (Value input : cast<DestinationStyleOpInterface>(def).getDpsInputs())
    if (!isa<ContextType>(input.getType()))
      collectConstantScalarExpr(input, ops);
}

// A hip.expand's target shape is not data: HipToTosa takes the broadcast from
// the result type, so the shape only has to resolve inside the kernel, and
// absorbOperandFreeProducers clones a constant one.
std::optional<SmallVector<Value>> getDataInputs(Operation *producer,
                                                Value context) {
  if (auto expand = dyn_cast<ExpandOp>(producer)) {
    Operation *shape = expand.getShape().getDefiningOp();
    if (!shape || shape->getNumOperands() != 0 || !isMemoryEffectFree(shape))
      return std::nullopt;
    return SmallVector<Value>{expand.getInput()};
  }
  auto dps = dyn_cast<DestinationStyleOpInterface>(producer);
  SmallVector<Value> inputs = dps ? SmallVector<Value>(dps.getDpsInputs())
                                  : SmallVector<Value>(producer->getOperands());
  llvm::erase(inputs, context);
  return inputs;
}

} // namespace mlir::hip
