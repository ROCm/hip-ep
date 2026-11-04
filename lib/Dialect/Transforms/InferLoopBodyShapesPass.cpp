/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InferLoopBodyShapesPass.cpp - Rank unranked tensors in loop bodies -===//
//
// Module pass that runs AFTER `onnx-loop-outline` and BEFORE
// `convert-onnx-to-hip`.
//
// Problem. The importer emits `tensor<*xT>` (unranked) for values whose
// shape it cannot infer. The canonical case is an outlined `hip.loop`
// body's loop-carried output -- e.g. an `onnx.Concat` that accumulates a
// KV-cache slice along the sequence axis. The ONNX-to-HIP converters
// require ranked result types (`ConcatConversion` bails on unranked), so
// an unranked body output (a) blocks conversion -- the op survives as
// `onnx.*` and later fails bufferization -- and (b) leaves the body
// `func.return` operand type-incompatible with the func signature.
//
// Fix. For every `hip.loop` body func, establish rank using two
// complementary sources of truth, then reconcile the signature:
//
//   1. Seed  -- copy the loop op's `$v_init` operand types onto the body's
//               loop-carried block args (already ranked post-outline; done
//               for parity with onnx-mlir `ONNXLoopOp::inferShapes`, which
//               likewise seeds body args from the loop inputs).
//   2. Infer -- forward-propagate rank onto unranked `onnx.*` results from
//               their (now ranked) operands, to a fixed point.
//   3. Backstop -- any loop-carried body output still unranked is set to the
//               matching `$v_init` type: the ONNX Loop spec mandates
//               body-output type == loop-carried-input type, so this is
//               authoritative even where no forward rule exists.
//   4. Reconcile -- rebuild the body func signature from its (seeded) args
//               and its (now ranked) terminator operands.
//
//   Main-graph inference can refine a value that is already a `$v_init`
//   (a constant Slice is the Qwen case). That happens after the first body
//   seed, so the walk repeats: copy each refined `$v_init` type onto the
//   `hip.loop` result, then seed and reconcile that body again. The loop
//   verifier requires those two types to be identical.
//
// The same forward walk also runs on every other function, not only loop
// bodies. A main-graph `onnx.Reshape` whose shape operand is `tensor<Nxi64>`
// has result rank N even when the importer left the result `tensor<*xT>`
// (BertSquad: four such Reshapes, and every op downstream of them). Ranking
// that result, then the ops that copy or broadcast its shape, lets
// `convert-onnx-to-hip` see dynamic shapes (`tensor<?x?x?xT>`) instead of an
// unknown rank. The same walk fills `Pad`, `Conv`, `InstanceNormalization`,
// and `Upsample` results (FNSCandy: the file stores no intermediate shapes,
// so the first Pad's ranked input and unranked result crash conversion).
// Static pads, kernel, stride, and scales become static extents; anything
// still unknown stays dynamic.
//
// Scope. This pass only ESTABLISHES rank so conversion can proceed; all
// `?`-dim narrowing on the resulting HIP-dialect ops remains the job of the
// post-conversion `--hip-infer-shapes`. See
// `docs/design/hip-shape-inference.md`.
//
// Before:
//   func.func private @loop_body(%ctx, %i, %c, %acc: tensor<1x?x1152xf16>, ...)
//       -> tensor<*xf16> {
//     %a = hip.multi_head_attention ... : tensor<?x?x?xf16>
//     %r = "onnx.Concat"(%acc, %a) {axis = 1} :
//            (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<*xf16>
//     return %r : tensor<*xf16>
//   }
// After:
//   func.func private @loop_body(%ctx, %i, %c, %acc: tensor<1x?x1152xf16>, ...)
//       -> tensor<1x?x1152xf16> {
//     %a = hip.multi_head_attention ... : tensor<?x?x?xf16>
//     %r = "onnx.Concat"(%acc, %a) {axis = 1} :
//            (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<1x?x1152xf16>
//     return %r : tensor<1x?x1152xf16>
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cmath>

#define DEBUG_TYPE "hip-infer-loop-body-shapes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

STATISTIC(NumOnnxResultsRanked,
          "Unranked onnx.* results rank-established by forward inference");
STATISTIC(NumLoopContractRanked,
          "Loop-carried body outputs rank-established from $v_init "
          "(loop-contract backstop)");
STATISTIC(NumBodyFuncsReconciled,
          "Loop-body func signatures reconciled from terminator operand types");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_INFERLOOPBODYSHAPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Forward shape rule for `onnx.Concat`: infer a ranked result type from the
/// operand types, or return null when the rule cannot apply (some operand is
/// unranked, ranks or element types disagree, or `axis` is missing / out of
/// range).
///
/// Follows the ONNX Concat spec (and onnx-mlir `ONNXConcatOpShapeHelper`):
/// every operand shares the result's rank and element type; a non-axis dim
/// takes any operand's static extent if one exists (else stays dynamic), and
/// the axis dim is the sum of operand extents (dynamic if any operand is
/// dynamic there). Example: `Concat((1x?x1152, ?x?x?), axis=1)` -> `1x?x1152`.
static RankedTensorType inferConcatResult(Operation *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() != 1)
    return {};
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!axisAttr)
    return {};

  // All operands must be ranked tensors of one common rank and element type.
  SmallVector<RankedTensorType> operands;
  operands.reserve(op->getNumOperands());
  int64_t rank = -1;
  Type elementType;
  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t)
      return {};
    if (rank < 0) {
      rank = t.getRank();
      elementType = t.getElementType();
    } else if (t.getRank() != rank || t.getElementType() != elementType) {
      return {};
    }
    operands.push_back(t);
  }
  if (rank <= 0) // rank-0 tensors have no axis to concatenate.
    return {};

  // ONNX `axis` is signed (`si64`); getSInt sign-extends correctly. Negative
  // axes count from the end.
  int64_t axis = axisAttr.getSInt();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return {};

  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  for (int64_t d : llvm::seq<int64_t>(0, rank)) {
    if (d != axis) {
      // Non-axis dim: adopt the first static extent any operand provides.
      for (RankedTensorType t : operands)
        if (!t.isDynamicDim(d)) {
          shape[d] = t.getDimSize(d);
          break;
        }
      continue;
    }
    // Axis dim: sum of operand extents, dynamic if any contributor is.
    int64_t sum = 0;
    bool allStatic = true;
    for (RankedTensorType t : operands) {
      if (t.isDynamicDim(d)) {
        allStatic = false;
        break;
      }
      sum += t.getDimSize(d);
    }
    shape[d] = allStatic ? sum : ShapedType::kDynamic;
  }
  return RankedTensorType::get(shape, elementType);
}

static Type elementTypeOf(Type type) {
  if (auto ranked = dyn_cast<RankedTensorType>(type))
    return ranked.getElementType();
  if (auto unranked = dyn_cast<UnrankedTensorType>(type))
    return unranked.getElementType();
  return {};
}

static RankedTensorType rankedLike(Type resultType, ArrayRef<int64_t> shape) {
  Type elementType = elementTypeOf(resultType);
  if (!elementType)
    return {};
  return RankedTensorType::get(shape, elementType);
}

/// A 1-D integer constant, including a scalar. Defined below, after the
/// constant folder. A runtime tensor yields nullopt.
static std::optional<SmallVector<int64_t>> readConstInt1D(Value value);

/// Keep every size the result already has. A `?` becomes a number only when
/// `inferred` has one. An unranked result takes `inferred` wholesale, which
/// is how a missing rank is established.
static Type mergeInferredType(Type existing, Type inferred) {
  if (isa<UnrankedTensorType>(existing))
    return inferred;
  auto existingRanked = dyn_cast<RankedTensorType>(existing);
  auto inferredRanked = dyn_cast<RankedTensorType>(inferred);
  if (!existingRanked || !inferredRanked ||
      existingRanked.getRank() != inferredRanked.getRank() ||
      existingRanked.getElementType() != inferredRanked.getElementType())
    return existing;
  SmallVector<int64_t> shape;
  shape.reserve(existingRanked.getRank());
  bool changed = false;
  for (auto [have, got] :
       llvm::zip(existingRanked.getShape(), inferredRanked.getShape())) {
    if (!ShapedType::isDynamic(have)) {
      shape.push_back(have);
    } else if (!ShapedType::isDynamic(got)) {
      shape.push_back(got);
      changed = true;
    } else {
      shape.push_back(have);
    }
  }
  if (!changed)
    return existing;
  return RankedTensorType::get(shape, existingRanked.getElementType());
}

/// Numpy-style broadcast. A missing leading dim counts as size 1. Two static
/// extents that are neither equal nor 1 are a malformed broadcast: decline
/// rather than invent a rank.
static FailureOr<SmallVector<int64_t>> broadcastDims(ArrayRef<int64_t> lhs,
                                                     ArrayRef<int64_t> rhs) {
  int64_t outRank = std::max(lhs.size(), rhs.size());
  SmallVector<int64_t> shape(outRank, ShapedType::kDynamic);
  for (int64_t i = 0; i < outRank; ++i) {
    int64_t lhsDim = i < outRank - (int64_t)lhs.size()
                         ? 1
                         : lhs[i - (outRank - (int64_t)lhs.size())];
    int64_t rhsDim = i < outRank - (int64_t)rhs.size()
                         ? 1
                         : rhs[i - (outRank - (int64_t)rhs.size())];
    if (lhsDim == 1)
      shape[i] = rhsDim;
    else if (rhsDim == 1 || lhsDim == rhsDim)
      shape[i] = lhsDim;
    else if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim))
      shape[i] = ShapedType::kDynamic;
    else
      return failure();
  }
  return shape;
}

static RankedTensorType copyInputShape(Value input, Type resultType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return {};
  return rankedLike(resultType, inputType.getShape());
}

static bool named(StringRef name, ArrayRef<StringRef> ops) {
  return llvm::is_contained(ops, name);
}

static bool preservesShape(StringRef name) {
  static constexpr StringRef kOps[] = {
      "onnx.Cast",    "onnx.Identity", "onnx.Sqrt",  "onnx.Reciprocal",
      "onnx.Tanh",    "onnx.Softmax",  "onnx.Relu",  "onnx.LeakyRelu",
      "onnx.Sigmoid", "onnx.Exp",      "onnx.Log",   "onnx.Neg",
      "onnx.Abs",     "onnx.Erf",      "onnx.Floor", "onnx.Ceil",
      "onnx.Round",   "onnx.Not",      "onnx.Sign"};
  return named(name, kOps);
}

static bool broadcastsShape(StringRef name) {
  static constexpr StringRef kOps[] = {"onnx.Add", "onnx.Mul", "onnx.Sub",
                                       "onnx.Pow", "onnx.Div", "onnx.Max",
                                       "onnx.Min"};
  return named(name, kOps);
}

static bool isReduction(StringRef name) {
  static constexpr StringRef kOps[] = {"onnx.ReduceMean", "onnx.ReduceSum",
                                       "onnx.ReduceMax",  "onnx.ReduceMin",
                                       "onnx.ReduceProd", "onnx.ReduceL2"};
  return named(name, kOps);
}

/// One dynamic slot per element of a shape tensor whose contents are not
/// visible. A static length is required; a `tensor<?xi64>` fixes nothing.
static void appendUnknownExtents(Value value,
                                 SmallVectorImpl<int64_t> &extents) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (type && type.getRank() == 1 && !type.isDynamicDim(0)) {
    extents.append(type.getDimSize(0), ShapedType::kDynamic);
    return;
  }
  extents.push_back(ShapedType::kDynamic);
}

/// Extents of a shape vector, walking the Concat / Unsqueeze / Cast chain the
/// importer builds around scalar constants. A positive constant is a static
/// dim. `0` (copy an input dim) and `-1` (infer) stay dynamic, as does
/// anything computed at runtime.
static void appendShapeExtents(Value value, SmallVectorImpl<int64_t> &extents) {
  Operation *def = value.getDefiningOp();
  while (def && (def->getName().getStringRef() == "onnx.Cast" ||
                 def->getName().getStringRef() == "onnx.Identity")) {
    if (def->getNumOperands() < 1)
      break;
    value = def->getOperand(0);
    def = value.getDefiningOp();
  }
  if (!def) {
    appendUnknownExtents(value, extents);
    return;
  }
  StringRef name = def->getName().getStringRef();
  if (name == "onnx.Concat") {
    for (Value operand : def->getOperands())
      appendShapeExtents(operand, extents);
    return;
  }
  if ((name == "onnx.Unsqueeze" || name == "onnx.Squeeze") &&
      def->getNumOperands() >= 1) {
    appendShapeExtents(def->getOperand(0), extents);
    return;
  }
  if (name == "onnx.Constant") {
    if (auto dense = def->getAttrOfType<DenseElementsAttr>("value");
        dense && dense.getElementType().isIntOrIndex() &&
        dense.getNumElements() > 0) {
      for (llvm::APInt element : dense.getValues<llvm::APInt>()) {
        int64_t extent = element.getSExtValue();
        extents.push_back(extent > 0 ? extent : ShapedType::kDynamic);
      }
      return;
    }
  }
  appendUnknownExtents(value, extents);
}

/// `onnx.Reshape` result rank is the length of a shape vector whose length is
/// static. Constant entries in that vector become static dims; the rest stay
/// dynamic and are read at runtime by the Reshape lowering. A `tensor<?xi64>`
/// shape does not fix the rank.
static RankedTensorType inferReshapeResult(Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return {};
  auto shapeType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!shapeType || shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
    return {};
  int64_t rank = shapeType.getDimSize(0);
  SmallVector<int64_t> extents;
  appendShapeExtents(op->getOperand(1), extents);
  if ((int64_t)extents.size() != rank)
    extents.assign(rank, ShapedType::kDynamic);
  return rankedLike(op->getResult(0).getType(), extents);
}

static RankedTensorType inferTransposeResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> perm;
  if (auto permAttr = op->getAttrOfType<ArrayAttr>("perm")) {
    if ((int64_t)permAttr.size() != rank)
      return {};
    perm.reserve(rank);
    for (Attribute attr : permAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return {};
      int64_t src = intAttr.getValue().getSExtValue();
      if (src < 0)
        src += rank;
      if (src < 0 || src >= rank)
        return {};
      perm.push_back(src);
    }
  } else {
    // ONNX default: reverse the dimensions.
    for (int64_t i = rank - 1; i >= 0; --i)
      perm.push_back(i);
  }
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  for (auto [outDim, srcDim] : llvm::enumerate(perm))
    shape[outDim] = inputType.getDimSize(srcDim);
  return rankedLike(op->getResult(0).getType(), shape);
}

/// Multidirectional broadcast of every operand. Min and Max are variadic, so
/// a later input can raise the rank or a size. An unranked operand can do
/// the same, and then the result is left alone.
static RankedTensorType inferBroadcastResult(Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return {};
  std::optional<SmallVector<int64_t>> shape;
  for (Value operand : op->getOperands()) {
    auto ranked = dyn_cast<RankedTensorType>(operand.getType());
    if (!ranked)
      return {};
    if (!shape) {
      shape = SmallVector<int64_t>(ranked.getShape());
      continue;
    }
    auto next = broadcastDims(*shape, ranked.getShape());
    if (failed(next))
      return {};
    shape = std::move(*next);
  }
  return rankedLike(op->getResult(0).getType(), *shape);
}

/// ONNX MatMul. A 1-D operand is promoted to a row or column and that
/// temporary dim is then dropped from the result. Batch dims broadcast.
static RankedTensorType inferMatMulResult(Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return {};
  auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!lhsType || !rhsType)
    return {};
  bool dropRow = lhsType.getRank() == 1;
  bool dropCol = rhsType.getRank() == 1;
  SmallVector<int64_t> lhs(lhsType.getShape());
  SmallVector<int64_t> rhs(rhsType.getShape());
  if (dropRow)
    lhs.insert(lhs.begin(), 1);
  if (dropCol)
    rhs.push_back(1);
  if (lhs.size() < 2 || rhs.size() < 2)
    return {};
  auto batch = broadcastDims(ArrayRef<int64_t>(lhs).drop_back(2),
                             ArrayRef<int64_t>(rhs).drop_back(2));
  if (failed(batch))
    return {};
  batch->push_back(lhs[lhs.size() - 2]);
  batch->push_back(rhs.back());
  if (dropRow)
    batch->erase(batch->end() - 2);
  if (dropCol)
    batch->pop_back();
  return rankedLike(op->getResult(0).getType(), *batch);
}

static RankedTensorType inferReduceResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();
  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<IntegerAttr>("keepdims"))
    keepdims = keepdimsAttr.getValue().getSExtValue();
  SmallVector<int64_t> axes;
  // A runtime axes operand does not fix which dims disappear.
  if (op->getNumOperands() > 1) {
    auto axisValues = readConstInt1D(op->getOperand(1));
    if (!axisValues)
      return {};
    for (int64_t axis : *axisValues) {
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        return {};
      axes.push_back(axis);
    }
  } else if (auto axesAttr = op->getAttrOfType<ArrayAttr>("axes")) {
    for (Attribute attr : axesAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return {};
      int64_t axis = intAttr.getValue().getSExtValue();
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        return {};
      axes.push_back(axis);
    }
  } else {
    int64_t noop = 0;
    if (auto noopAttr = op->getAttrOfType<IntegerAttr>("noop_with_empty_axes"))
      noop = noopAttr.getValue().getSExtValue();
    if (noop == 0)
      for (int64_t i = 0; i < rank; ++i)
        axes.push_back(i);
  }
  SmallVector<bool> reduced(rank, false);
  for (int64_t axis : axes)
    reduced[axis] = true;
  SmallVector<int64_t> shape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!reduced[i])
      shape.push_back(inputType.getDimSize(i));
    else if (keepdims)
      shape.push_back(1);
  }
  return rankedLike(op->getResult(0).getType(), shape);
}

/// Output shape is data[0:axis] + indices.shape + data[axis+1:].
static RankedTensorType inferGatherResult(Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return {};
  auto dataType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto indicesType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!dataType || !indicesType || !axisAttr || dataType.getRank() == 0)
    return {};
  int64_t axis = axisAttr.getValue().getSExtValue();
  if (axis < 0)
    axis += dataType.getRank();
  if (axis < 0 || axis >= dataType.getRank())
    return {};
  SmallVector<int64_t> shape;
  shape.append(dataType.getShape().begin(), dataType.getShape().begin() + axis);
  shape.append(indicesType.getShape().begin(), indicesType.getShape().end());
  shape.append(dataType.getShape().begin() + axis + 1,
               dataType.getShape().end());
  return rankedLike(op->getResult(0).getType(), shape);
}

/// An omitted optional ONNX input is either a none type or `onnx.NoValue`.
static bool isOnnxNone(Value value) {
  if (isa<NoneType>(value.getType()))
    return true;
  Operation *def = value.getDefiningOp();
  return def && def->getName().getStringRef() == "onnx.NoValue";
}

/// Equal Split, or a Split whose lengths operand is a constant vector.
/// Each output keeps the input rank. The split axis is the equal share, or
/// the matching constant length. A runtime lengths tensor yields no type.
static LogicalResult inferSplitResults(Operation *op,
                                       SmallVectorImpl<Type> &out) {
  if (op->getNumOperands() < 1 || op->getNumResults() == 0)
    return failure();
  SmallVector<int64_t> lengths;
  bool custom = false;
  if (op->getNumOperands() > 1 && !isOnnxNone(op->getOperand(1))) {
    auto lens = readConstInt1D(op->getOperand(1));
    if (!lens || lens->size() != op->getNumResults())
      return failure();
    lengths = std::move(*lens);
    custom = true;
  }
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return failure();
  int64_t axis = 0;
  if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis"))
    axis = axisAttr.getValue().getSExtValue();
  if (axis < 0)
    axis += inputType.getRank();
  if (axis < 0 || axis >= inputType.getRank())
    return failure();
  unsigned numOutputs = op->getNumResults();
  int64_t splitDim = ShapedType::kDynamic;
  if (!custom && !inputType.isDynamicDim(axis) && numOutputs > 0 &&
      inputType.getDimSize(axis) % numOutputs == 0)
    splitDim = inputType.getDimSize(axis) / numOutputs;
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    SmallVector<int64_t> shape(inputType.getShape());
    shape[axis] = custom ? lengths[index] : splitDim;
    RankedTensorType ranked = rankedLike(result.getType(), shape);
    if (!ranked)
      return failure();
    out.push_back(ranked);
  }
  return success();
}

/// ONNX int-list attributes are an ArrayAttr of IntegerAttr. A dense i64
/// array is accepted too.
static bool readIntArrayAttr(Operation *op, StringRef name,
                             SmallVectorImpl<int64_t> &out) {
  if (auto dense = op->getAttrOfType<DenseI64ArrayAttr>(name)) {
    out.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return true;
  }
  auto array = op->getAttrOfType<ArrayAttr>(name);
  if (!array)
    return false;
  out.clear();
  for (Attribute attr : array) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return false;
    out.push_back(intAttr.getValue().getSExtValue());
  }
  return true;
}

static int64_t floorDiv(int64_t n, int64_t d) {
  int64_t q = n / d;
  int64_t r = n % d;
  if (r != 0 && ((n < 0) != (d < 0)))
    --q;
  return q;
}

/// `onnx.Pad`. out[i] = in[i] + begin[i] + end[i]. Pads come from the
/// attribute, or from a constant pads input. A runtime pads tensor yields
/// no type, so a stored shape is left alone.
static RankedTensorType inferPadResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> pads;
  if (!readIntArrayAttr(op, "pads", pads)) {
    if (op->getNumOperands() < 2 || isOnnxNone(op->getOperand(1)))
      return {};
    auto padValues = readConstInt1D(op->getOperand(1));
    if (!padValues)
      return {};
    pads = std::move(*padValues);
  }
  SmallVector<int64_t> axes;
  bool haveAxes = readIntArrayAttr(op, "axes", axes);
  if (!haveAxes && op->getNumOperands() >= 4 &&
      !isOnnxNone(op->getOperand(3))) {
    auto axisValues = readConstInt1D(op->getOperand(3));
    if (!axisValues)
      return {};
    axes = std::move(*axisValues);
    haveAxes = true;
  }
  SmallVector<int64_t> shape(inputType.getShape());
  if (!haveAxes) {
    if ((int64_t)pads.size() != rank * 2)
      return {};
    for (int64_t i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(shape[i]))
        continue;
      shape[i] = shape[i] + pads[i] + pads[i + rank];
    }
  } else {
    if ((int64_t)pads.size() != (int64_t)axes.size() * 2)
      return {};
    for (auto [index, axis] : llvm::enumerate(axes)) {
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        return {};
      if (ShapedType::isDynamic(shape[axis]))
        continue;
      shape[axis] = shape[axis] + pads[index] + pads[index + axes.size()];
    }
  }
  return rankedLike(op->getResult(0).getType(), shape);
}

/// `onnx.Conv`. Rank matches the input. N comes from the input, C from the
/// weight's leading dim, and each spatial dim from the ONNX output formula
/// when the input extent and kernel/stride/pad/dilation are static.
static RankedTensorType inferConvResult(Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto weightType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!inputType || !weightType)
    return {};
  int64_t rank = inputType.getRank();
  if (rank < 3 || rank > 5 || weightType.getRank() != rank)
    return {};
  int64_t spatial = rank - 2;
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  shape[0] = inputType.getDimSize(0);
  if (!weightType.isDynamicDim(0))
    shape[1] = weightType.getDimSize(0);

  StringRef autoPad;
  if (auto attr = op->getAttrOfType<StringAttr>("auto_pad"))
    autoPad = attr.getValue();
  if (!autoPad.empty() && autoPad != "NOTSET")
    return rankedLike(op->getResult(0).getType(), shape);

  SmallVector<int64_t> kernel, strides, pads, dilations;
  if (!readIntArrayAttr(op, "kernel_shape", kernel)) {
    bool weightSpatialStatic = true;
    for (int64_t i = 0; i < spatial; ++i)
      weightSpatialStatic &= !weightType.isDynamicDim(2 + i);
    if (!weightSpatialStatic)
      return rankedLike(op->getResult(0).getType(), shape);
    for (int64_t i = 0; i < spatial; ++i)
      kernel.push_back(weightType.getDimSize(2 + i));
  }
  if (!readIntArrayAttr(op, "strides", strides))
    strides.assign(spatial, 1);
  if (!readIntArrayAttr(op, "pads", pads))
    pads.assign(spatial * 2, 0);
  if (!readIntArrayAttr(op, "dilations", dilations))
    dilations.assign(spatial, 1);
  if ((int64_t)kernel.size() != spatial || (int64_t)strides.size() != spatial ||
      (int64_t)pads.size() != spatial * 2 ||
      (int64_t)dilations.size() != spatial)
    return rankedLike(op->getResult(0).getType(), shape);

  for (int64_t s = 0; s < spatial; ++s) {
    int64_t in = inputType.getDimSize(2 + s);
    if (ShapedType::isDynamic(in) || strides[s] == 0)
      continue;
    int64_t numer =
        in + pads[s] + pads[spatial + s] - dilations[s] * (kernel[s] - 1) - 1;
    shape[2 + s] = floorDiv(numer, strides[s]) + 1;
  }
  return rankedLike(op->getResult(0).getType(), shape);
}

/// `onnx.AveragePool`. N and C are copied. Each spatial extent is
/// floor((in + pads - dilation * (kernel - 1) - 1) / stride) + 1.
/// 1x64x512x512 with kernel 2 and stride 2 is 1x64x256x256.
static RankedTensorType inferAveragePoolResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || inputType.getRank() < 3)
    return {};
  int64_t rank = inputType.getRank();
  int64_t spatial = rank - 2;
  SmallVector<int64_t> kernel, strides, pads, dilations;
  if (!readIntArrayAttr(op, "kernel_shape", kernel) ||
      (int64_t)kernel.size() != spatial)
    return {};
  if (!readIntArrayAttr(op, "strides", strides))
    strides.assign(spatial, 1);
  if (!readIntArrayAttr(op, "dilations", dilations))
    dilations.assign(spatial, 1);
  if ((int64_t)strides.size() != spatial ||
      (int64_t)dilations.size() != spatial)
    return {};

  StringRef autoPad = "NOTSET";
  if (auto attr = op->getAttrOfType<StringAttr>("auto_pad"))
    autoPad = attr.getValue();
  bool same = autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER";
  bool valid = autoPad == "VALID";
  if (!same && !valid && autoPad != "NOTSET")
    return {};
  if (!same && !valid) {
    if (!readIntArrayAttr(op, "pads", pads))
      pads.assign(spatial * 2, 0);
    if ((int64_t)pads.size() != spatial * 2)
      return {};
  }
  int64_t ceilMode = 0;
  if (auto attr = op->getAttrOfType<IntegerAttr>("ceil_mode"))
    ceilMode = attr.getValue().getSExtValue();

  SmallVector<int64_t> shape(inputType.getShape());
  for (int64_t s = 0; s < spatial; ++s) {
    int64_t in = inputType.getDimSize(2 + s);
    if (ShapedType::isDynamic(in) || strides[s] <= 0 || kernel[s] <= 0 ||
        dilations[s] <= 0)
      continue;
    if (same) {
      shape[2 + s] = (in + strides[s] - 1) / strides[s];
      continue;
    }
    int64_t padBegin = valid ? 0 : pads[s];
    int64_t padEnd = valid ? 0 : pads[spatial + s];
    int64_t numer = in + padBegin + padEnd - dilations[s] * (kernel[s] - 1) - 1;
    int64_t quot = ceilMode ? (numer + strides[s] - 1) / strides[s]
                            : floorDiv(numer, strides[s]);
    int64_t out = quot + 1;
    shape[2 + s] = out > 0 ? out : 0;
  }
  return rankedLike(op->getResult(0).getType(), shape);
}

/// `onnx.ConvTranspose` takes its output channel extent from the second
/// weight dimension times group. Exporters that provide `output_shape` give
/// the spatial extents directly, avoiding a loss of rank before conversion.
static RankedTensorType inferConvTransposeResult(Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto weightType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!inputType || !weightType || inputType.getRank() < 3 ||
      inputType.getRank() > 5 || weightType.getRank() != inputType.getRank())
    return {};

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<IntegerAttr>("group"))
    group = attr.getSInt();
  if (group <= 0 || weightType.isDynamicDim(1))
    return {};

  int64_t spatialRank = inputType.getRank() - 2;
  SmallVector<int64_t> shape(inputType.getShape().begin(),
                             inputType.getShape().end());
  shape[1] = weightType.getDimSize(1) * group;
  auto outputShape = op->getAttrOfType<ArrayAttr>("output_shape");
  if (!outputShape || outputShape.size() != static_cast<size_t>(spatialRank))
    return {};
  for (int64_t i : llvm::seq<int64_t>(spatialRank))
    shape[2 + i] = cast<IntegerAttr>(outputShape[i]).getSInt();
  return rankedLike(op->getResult(0).getType(), shape);
}

/// `onnx.Shape` result is a 1-D vector whose length is the (possibly sliced)
/// input rank.
static RankedTensorType inferShapeOpResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();
  int64_t start = 0;
  int64_t end = rank;
  if (auto attr = op->getAttrOfType<IntegerAttr>("start"))
    start = attr.getValue().getSExtValue();
  if (auto attr = op->getAttrOfType<IntegerAttr>("end"))
    end = attr.getValue().getSExtValue();
  if (start < 0)
    start += rank;
  if (end < 0)
    end += rank;
  start = std::max(start, static_cast<int64_t>(0));
  end = std::min(end, rank);
  int64_t length = end > start ? end - start : 0;
  return rankedLike(op->getResult(0).getType(), {length});
}

/// `onnx.Unsqueeze` inserts unit dims at `axes` (attribute, or a constant
/// operand). Output rank is input rank plus the number of axes.
static RankedTensorType inferUnsqueezeResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  SmallVector<int64_t> axes;
  if (!readIntArrayAttr(op, "axes", axes)) {
    if (op->getNumOperands() < 2 || isOnnxNone(op->getOperand(1)))
      return {};
    auto axisValues = readConstInt1D(op->getOperand(1));
    if (!axisValues)
      return {};
    axes = std::move(*axisValues);
  }
  int64_t outRank = inputType.getRank() + (int64_t)axes.size();
  SmallVector<bool> inserted(outRank, false);
  for (int64_t &axis : axes) {
    if (axis < 0)
      axis += outRank;
    if (axis < 0 || axis >= outRank || inserted[axis])
      return {};
    inserted[axis] = true;
  }
  SmallVector<int64_t> shape(outRank, 1);
  int64_t src = 0;
  for (int64_t i = 0; i < outRank; ++i)
    if (!inserted[i])
      shape[i] = inputType.getDimSize(src++);
  return rankedLike(op->getResult(0).getType(), shape);
}

/// `onnx.Slice`. Rank is unchanged. A static axis extent becomes the number
/// of elements from start to end. Bounds come from attributes, or from
/// constant inputs. A runtime bound yields no type, so a stored shape stays.
static RankedTensorType inferSliceResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  SmallVector<int64_t> starts, ends, axes, steps;
  if (!readIntArrayAttr(op, "starts", starts) ||
      !readIntArrayAttr(op, "ends", ends)) {
    if (op->getNumOperands() < 3)
      return {};
    auto startValues = readConstInt1D(op->getOperand(1));
    auto endValues = readConstInt1D(op->getOperand(2));
    if (!startValues || !endValues)
      return {};
    starts = std::move(*startValues);
    ends = std::move(*endValues);
  }
  if (!readIntArrayAttr(op, "axes", axes) && op->getNumOperands() >= 4 &&
      !isOnnxNone(op->getOperand(3))) {
    auto axisValues = readConstInt1D(op->getOperand(3));
    if (!axisValues)
      return {};
    axes = std::move(*axisValues);
  }
  if (axes.empty()) {
    axes.resize(starts.size());
    for (int64_t i = 0; i < (int64_t)starts.size(); ++i)
      axes[i] = i;
  }
  if (!readIntArrayAttr(op, "steps", steps) && op->getNumOperands() >= 5 &&
      !isOnnxNone(op->getOperand(4))) {
    auto stepValues = readConstInt1D(op->getOperand(4));
    if (!stepValues)
      return {};
    steps = std::move(*stepValues);
  }
  if (steps.empty())
    steps.assign(starts.size(), 1);
  if (starts.size() != ends.size() || starts.size() != axes.size() ||
      starts.size() != steps.size())
    return {};
  SmallVector<int64_t> shape(inputType.getShape().begin(),
                             inputType.getShape().end());
  int64_t rank = inputType.getRank();
  for (auto [axis, start, end, step] : llvm::zip(axes, starts, ends, steps)) {
    if (step == 0)
      return {};
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return {};
    int64_t dim = inputType.getDimSize(axis);
    if (ShapedType::isDynamic(dim)) {
      shape[axis] = ShapedType::kDynamic;
      continue;
    }
    if (step < 0)
      return {};
    if (start < 0)
      start += dim;
    if (end < 0)
      end += dim;
    start = std::clamp(start, static_cast<int64_t>(0), dim);
    end = std::clamp(end, static_cast<int64_t>(0), dim);
    int64_t span = end > start ? end - start : 0;
    shape[axis] = (span + step - 1) / step;
  }
  return rankedLike(op->getResult(0).getType(), shape);
}

/// A tiny constant tensor, used to fold the Shape/Gather/Mul/Floor chain that
/// builds Upsample scales. Big tensors (weights) are declined.
struct ConstTensor {
  SmallVector<int64_t> shape;
  SmallVector<double> data;
};

static int64_t constNumel(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    if (dim < 0)
      return -1;
    n *= dim;
  }
  return n;
}

static void unravelIndex(int64_t linear, ArrayRef<int64_t> shape,
                         SmallVectorImpl<int64_t> &index) {
  index.resize(shape.size());
  for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
    int64_t dim = shape[i] == 0 ? 1 : shape[i];
    index[i] = linear % dim;
    linear /= dim;
  }
}

static int64_t broadcastOffset(ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> outIndex) {
  int64_t offset = 0;
  int64_t outRank = outIndex.size();
  int64_t rank = shape.size();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim = shape[i];
    int64_t index = outIndex[outRank - rank + i];
    if (dim == 1)
      index = 0;
    offset = offset * (dim == 0 ? 1 : dim) + index;
  }
  return offset;
}

static std::optional<ConstTensor> evalConstTensor(Value value, int depth);

static std::optional<ConstTensor> evalBinaryConst(Operation *op, bool isDiv,
                                                  int depth) {
  if (op->getNumOperands() < 2)
    return std::nullopt;
  auto lhs = evalConstTensor(op->getOperand(0), depth + 1);
  auto rhs = evalConstTensor(op->getOperand(1), depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;
  int64_t outRank = std::max(lhs->shape.size(), rhs->shape.size());
  SmallVector<int64_t> outShape(outRank, 1);
  for (int64_t i = 0; i < outRank; ++i) {
    int64_t lhsPos = i - (outRank - (int64_t)lhs->shape.size());
    int64_t rhsPos = i - (outRank - (int64_t)rhs->shape.size());
    int64_t lhsDim = lhsPos >= 0 ? lhs->shape[lhsPos] : 1;
    int64_t rhsDim = rhsPos >= 0 ? rhs->shape[rhsPos] : 1;
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return std::nullopt;
    outShape[i] = std::max(lhsDim, rhsDim);
  }
  int64_t n = constNumel(outShape);
  if (n < 0 || n > 256)
    return std::nullopt;
  ConstTensor out;
  out.shape = std::move(outShape);
  out.data.resize(n);
  SmallVector<int64_t> index;
  for (int64_t i = 0; i < n; ++i) {
    unravelIndex(i, out.shape, index);
    double l = lhs->data[broadcastOffset(lhs->shape, index)];
    double r = rhs->data[broadcastOffset(rhs->shape, index)];
    if (isDiv && r == 0.0)
      return std::nullopt;
    out.data[i] = isDiv ? l / r : l * r;
  }
  return out;
}

static std::optional<ConstTensor> evalConstTensor(Value value, int depth) {
  if (depth > 32)
    return std::nullopt;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  StringRef name = def->getName().getStringRef();
  if (name == "onnx.Cast" || name == "onnx.Identity") {
    if (def->getNumOperands() < 1)
      return std::nullopt;
    return evalConstTensor(def->getOperand(0), depth + 1);
  }
  if (name == "onnx.Floor") {
    auto input = evalConstTensor(def->getOperand(0), depth + 1);
    if (!input)
      return std::nullopt;
    for (double &element : input->data)
      element = std::floor(element);
    return input;
  }
  if (name == "onnx.Mul")
    return evalBinaryConst(def, /*isDiv=*/false, depth);
  if (name == "onnx.Div")
    return evalBinaryConst(def, /*isDiv=*/true, depth);
  if (name == "onnx.Shape") {
    auto inputType = dyn_cast<RankedTensorType>(def->getOperand(0).getType());
    if (!inputType || !inputType.hasStaticShape())
      return std::nullopt;
    ConstTensor out;
    out.shape.push_back(inputType.getRank());
    for (int64_t dim : inputType.getShape())
      out.data.push_back(static_cast<double>(dim));
    return out;
  }
  if (name == "onnx.Constant") {
    auto attr = def->getAttrOfType<DenseElementsAttr>("value");
    auto type =
        attr ? dyn_cast<RankedTensorType>(attr.getType()) : RankedTensorType();
    if (!type || !type.hasStaticShape())
      return std::nullopt;
    int64_t n = constNumel(type.getShape());
    if (n < 0 || n > 256)
      return std::nullopt;
    ConstTensor out;
    out.shape.assign(type.getShape().begin(), type.getShape().end());
    out.data.reserve(n);
    if (attr.getElementType().isIntOrIndex()) {
      for (llvm::APInt element : attr.getValues<llvm::APInt>())
        out.data.push_back(static_cast<double>(element.getSExtValue()));
    } else if (isa<FloatType>(attr.getElementType())) {
      for (llvm::APFloat element : attr.getValues<llvm::APFloat>())
        out.data.push_back(element.convertToDouble());
    } else {
      return std::nullopt;
    }
    return out;
  }
  if (name == "onnx.Unsqueeze") {
    auto input = evalConstTensor(def->getOperand(0), depth + 1);
    if (!input)
      return std::nullopt;
    SmallVector<int64_t> axes;
    if (!readIntArrayAttr(def, "axes", axes))
      return std::nullopt;
    int64_t outRank = (int64_t)input->shape.size() + (int64_t)axes.size();
    SmallVector<bool> inserted(outRank, false);
    for (int64_t &axis : axes) {
      if (axis < 0)
        axis += outRank;
      if (axis < 0 || axis >= outRank || inserted[axis])
        return std::nullopt;
      inserted[axis] = true;
    }
    ConstTensor out;
    out.shape.resize(outRank, 1);
    int64_t src = 0;
    for (int64_t i = 0; i < outRank; ++i)
      if (!inserted[i])
        out.shape[i] = input->shape[src++];
    out.data = std::move(input->data);
    return out;
  }
  if (name == "onnx.Concat") {
    auto axisAttr = def->getAttrOfType<IntegerAttr>("axis");
    if (!axisAttr || def->getNumOperands() == 0)
      return std::nullopt;
    SmallVector<ConstTensor> parts;
    for (Value operand : def->getOperands()) {
      auto part = evalConstTensor(operand, depth + 1);
      if (!part)
        return std::nullopt;
      parts.push_back(std::move(*part));
    }
    int64_t rank = parts.front().shape.size();
    int64_t axis = axisAttr.getValue().getSExtValue();
    if (axis < 0)
      axis += rank;
    if (rank == 0 || axis < 0 || axis >= rank)
      return std::nullopt;
    SmallVector<int64_t> shape = parts.front().shape;
    shape[axis] = 0;
    for (ConstTensor &part : parts) {
      if ((int64_t)part.shape.size() != rank)
        return std::nullopt;
      for (int64_t i = 0; i < rank; ++i) {
        if (i == axis)
          shape[i] += part.shape[i];
        else if (part.shape[i] != shape[i])
          return std::nullopt;
      }
    }
    int64_t n = constNumel(shape);
    if (n < 0 || n > 256)
      return std::nullopt;
    ConstTensor out;
    out.shape = shape;
    // Row-major concat: walk each part's elements in order when axis is the
    // only varying placement. Build via output coordinates instead.
    out.data.resize(n);
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i)
      inner *= shape[i];
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i)
      outer *= shape[i];
    int64_t axisCursor = 0;
    for (ConstTensor &part : parts) {
      int64_t partAxis = part.shape[axis];
      int64_t partInner = inner;
      for (int64_t o = 0; o < outer; ++o) {
        for (int64_t a = 0; a < partAxis; ++a) {
          for (int64_t inn = 0; inn < partInner; ++inn) {
            int64_t src = (o * partAxis + a) * partInner + inn;
            int64_t dst = (o * shape[axis] + (axisCursor + a)) * inner + inn;
            out.data[dst] = part.data[src];
          }
        }
      }
      axisCursor += partAxis;
    }
    return out;
  }
  if (name == "onnx.Gather") {
    auto data = evalConstTensor(def->getOperand(0), depth + 1);
    auto indices = evalConstTensor(def->getOperand(1), depth + 1);
    if (!data || !indices || def->getNumOperands() < 2)
      return std::nullopt;
    int64_t axis = 0;
    if (auto attr = def->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getValue().getSExtValue();
    int64_t dataRank = data->shape.size();
    if (axis < 0)
      axis += dataRank;
    if (axis < 0 || axis >= dataRank)
      return std::nullopt;
    int64_t dim = data->shape[axis];
    SmallVector<int64_t> outShape;
    outShape.append(data->shape.begin(), data->shape.begin() + axis);
    outShape.append(indices->shape.begin(), indices->shape.end());
    outShape.append(data->shape.begin() + axis + 1, data->shape.end());
    int64_t n = constNumel(outShape);
    if (n < 0 || n > 256)
      return std::nullopt;
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < dataRank; ++i)
      inner *= data->shape[i];
    int64_t nIndex = constNumel(indices->shape);
    if (nIndex < 0)
      return std::nullopt;
    ConstTensor out;
    out.shape = std::move(outShape);
    out.data.resize(n);
    int64_t outer = n == 0 ? 0 : n / (nIndex * inner);
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t ii = 0; ii < nIndex; ++ii) {
        int64_t gathered = static_cast<int64_t>(indices->data[ii]);
        if (gathered < 0)
          gathered += dim;
        if (gathered < 0 || gathered >= dim)
          return std::nullopt;
        for (int64_t inn = 0; inn < inner; ++inn) {
          int64_t src = (o * dim + gathered) * inner + inn;
          int64_t dst = (o * nIndex + ii) * inner + inn;
          out.data[dst] = data->data[src];
        }
      }
    }
    return out;
  }
  if (name == "onnx.Slice") {
    auto input = evalConstTensor(def->getOperand(0), depth + 1);
    if (!input)
      return std::nullopt;
    SmallVector<int64_t> starts, ends, axes, steps;
    if (!readIntArrayAttr(def, "starts", starts) ||
        !readIntArrayAttr(def, "ends", ends))
      return std::nullopt;
    if (!readIntArrayAttr(def, "axes", axes)) {
      axes.resize(starts.size());
      for (int64_t i = 0; i < (int64_t)starts.size(); ++i)
        axes[i] = i;
    }
    if (!readIntArrayAttr(def, "steps", steps))
      steps.assign(starts.size(), 1);
    if (starts.size() != ends.size() || starts.size() != axes.size() ||
        starts.size() != steps.size())
      return std::nullopt;
    int64_t rank = input->shape.size();
    SmallVector<SmallVector<int64_t>> kept(rank);
    for (int64_t axis = 0; axis < rank; ++axis)
      for (int64_t i = 0; i < input->shape[axis]; ++i)
        kept[axis].push_back(i);
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = axes[i];
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank || steps[i] == 0)
        return std::nullopt;
      int64_t dim = input->shape[axis];
      int64_t start = starts[i];
      int64_t end = ends[i];
      int64_t step = steps[i];
      if (start < 0)
        start += dim;
      if (end < 0)
        end += dim;
      if (step > 0) {
        start = std::clamp(start, static_cast<int64_t>(0), dim);
        end = std::clamp(end, static_cast<int64_t>(0), dim);
      } else {
        start = std::clamp(start, static_cast<int64_t>(0), dim - 1);
        end = std::clamp(end, static_cast<int64_t>(-1), dim - 1);
      }
      SmallVector<int64_t> chosen;
      if (step > 0) {
        for (int64_t v = start; v < end; v += step)
          chosen.push_back(v);
      } else {
        for (int64_t v = start; v > end; v += step)
          chosen.push_back(v);
      }
      kept[axis] = std::move(chosen);
    }
    SmallVector<int64_t> outShape;
    for (SmallVector<int64_t> &axis : kept)
      outShape.push_back(axis.size());
    int64_t n = constNumel(outShape);
    if (n < 0 || n > 256)
      return std::nullopt;
    ConstTensor out;
    out.shape = outShape;
    out.data.resize(n);
    SmallVector<int64_t> outIndex, inIndex;
    for (int64_t i = 0; i < n; ++i) {
      unravelIndex(i, out.shape, outIndex);
      inIndex.resize(rank);
      for (int64_t axis = 0; axis < rank; ++axis)
        inIndex[axis] = kept[axis][outIndex[axis]];
      int64_t src = 0;
      for (int64_t axis = 0; axis < rank; ++axis)
        src = src * input->shape[axis] + inIndex[axis];
      out.data[i] = input->data[src];
    }
    return out;
  }
  return std::nullopt;
}

static std::optional<SmallVector<int64_t>> readConstInt1D(Value value) {
  auto tensor = evalConstTensor(value, /*depth=*/0);
  if (!tensor || tensor->shape.size() > 1)
    return std::nullopt;
  SmallVector<int64_t> values;
  values.reserve(tensor->data.size());
  for (double element : tensor->data) {
    if (!std::isfinite(element) || element != std::trunc(element))
      return std::nullopt;
    values.push_back(static_cast<int64_t>(element));
  }
  return values;
}

/// Opset 7-8 store scales as an f32 attribute and have no scales input.
static std::optional<SmallVector<double>>
readUpsampleScalesAttr(Operation *op) {
  if (auto dense = op->getAttrOfType<DenseElementsAttr>("scales")) {
    if (!dense.getElementType().isFloat())
      return std::nullopt;
    SmallVector<double> values;
    values.reserve(dense.getNumElements());
    for (APFloat value : dense.getValues<APFloat>())
      values.push_back(value.convertToDouble());
    return values;
  }
  auto array = op->getAttrOfType<ArrayAttr>("scales");
  if (!array)
    return std::nullopt;
  SmallVector<double> values;
  values.reserve(array.size());
  for (Attribute element : array) {
    auto floatAttr = dyn_cast<FloatAttr>(element);
    if (!floatAttr)
      return std::nullopt;
    values.push_back(floatAttr.getValueAsDouble());
  }
  return values;
}

static void applyUpsampleScales(SmallVectorImpl<int64_t> &shape,
                                ArrayRef<double> scales) {
  if ((int64_t)scales.size() != (int64_t)shape.size())
    return;
  for (auto [dim, scale] : llvm::zip(shape, scales)) {
    if (ShapedType::isDynamic(dim))
      continue;
    if (!(scale > 0.0) || !std::isfinite(scale))
      continue;
    dim = static_cast<int64_t>(std::floor(static_cast<double>(dim) * scale));
  }
}

/// `onnx.Upsample`. Rank matches the input. A scales tensor that folds to
/// constants (a dense constant, or the Shape/Gather/Mul/Floor pattern style
/// models use) turns a static input dim into floor(dim * scale). Opset 7-8
/// store that list in the `scales` attribute and have no second operand.
static RankedTensorType inferUpsampleResult(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return {};
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  SmallVector<int64_t> shape(inputType.getShape().begin(),
                             inputType.getShape().end());
  if (op->getNumOperands() >= 2) {
    if (auto scales = evalConstTensor(op->getOperand(1), /*depth=*/0))
      applyUpsampleScales(shape, scales->data);
  } else if (auto scales = readUpsampleScalesAttr(op)) {
    applyUpsampleScales(shape, *scales);
  }
  return rankedLike(op->getResult(0).getType(), shape);
}

/// Dispatch a forward shape rule for `op` when a result is unranked or has a
/// dynamic dimension. `out` receives one type per result. Returns failure when
/// no rule can refine every result from the operands. Register new op rules
/// below.
///
/// Why this is keyed on op name (and not the HIP dialect / an interface).
/// Rank must be established BEFORE `convert-onnx-to-hip`: an unranked result
/// blocks the converters (`ConcatConversion` bails on unranked), so the op
/// never reaches the HIP dialect to be refined post-conversion by
/// `--hip-infer-shapes`. At this stage the `onnx.*` ops are still
/// unregistered operations carried by a stub `onnx` dialect (`OnnxStubDialect`
/// in `InitAllPasses.h`, which `allowUnknownOperations` so unranked tensors
/// round-trip) -- this repo matches ONNX by name via the generic `Operation`
/// API rather than depending on onnx-mlir's registered op classes, so there is
/// no op class and no `ShapeInferenceOpInterface` to dispatch on. A name
/// switch is therefore the only handle available, and it mirrors how the
/// converter layer itself is organized (`RewritePattern("onnx.Concat", ...)`).
/// The op-agnostic loop-contract backstop in `inferLoopBodyShapes` is the
/// general safety net for the failure mode (the loop-carried output); these
/// forward rules are the enhancement tier that also ranks *interior* unranked
/// values.
///
/// How to make this generic. Once the build takes a proper dependency on a
/// registered ONNX dialect (onnx-mlir), every op implements
/// `ShapeInferenceOpInterface::inferShapes()`, and this whole switch collapses
/// into a single interface-driven walk -- the onnx-mlir `InferShapesPattern`
/// (`OpInterfaceRewritePattern<ShapeInferenceOpInterface>`) calls each op's
/// own `inferShapes`, so no per-op rule lives here anymore.
static LogicalResult inferUnrankedOnnxResults(Operation *op,
                                              SmallVectorImpl<Type> &out) {
  auto needsShapeRefinement = [](Type type) {
    if (isa<UnrankedTensorType>(type))
      return true;
    if (auto ranked = dyn_cast<RankedTensorType>(type))
      return !ranked.hasStaticShape();
    return false;
  };
  if (op->getNumResults() == 0 ||
      !llvm::any_of(op->getResultTypes(), needsShapeRefinement))
    return failure();

  StringRef name = op->getName().getStringRef();
  auto pushSingle = [&](RankedTensorType ranked) -> LogicalResult {
    if (!ranked || op->getNumResults() != 1)
      return failure();
    out.push_back(ranked);
    return success();
  };

  if (name == "onnx.Concat")
    return pushSingle(inferConcatResult(op));
  if (name == "onnx.Reshape")
    return pushSingle(inferReshapeResult(op));
  if (name == "onnx.Transpose")
    return pushSingle(inferTransposeResult(op));
  if (name == "onnx.MatMul")
    return pushSingle(inferMatMulResult(op));
  if (name == "onnx.Gather")
    return pushSingle(inferGatherResult(op));
  if (isReduction(name))
    return pushSingle(inferReduceResult(op));
  if (broadcastsShape(name))
    return pushSingle(inferBroadcastResult(op));
  if (preservesShape(name)) {
    if (op->getNumOperands() < 1)
      return failure();
    return pushSingle(
        copyInputShape(op->getOperand(0), op->getResult(0).getType()));
  }
  if (name == "onnx.Split")
    return inferSplitResults(op, out);
  if (name == "onnx.Pad")
    return pushSingle(inferPadResult(op));
  if (name == "onnx.Conv")
    return pushSingle(inferConvResult(op));
  if (name == "onnx.AveragePool")
    return pushSingle(inferAveragePoolResult(op));
  if (name == "onnx.ConvTranspose")
    return pushSingle(inferConvTransposeResult(op));
  if (name == "onnx.InstanceNormalization") {
    if (op->getNumOperands() < 1)
      return failure();
    return pushSingle(
        copyInputShape(op->getOperand(0), op->getResult(0).getType()));
  }
  if (name == "onnx.Shape")
    return pushSingle(inferShapeOpResult(op));
  if (name == "onnx.Unsqueeze")
    return pushSingle(inferUnsqueezeResult(op));
  if (name == "onnx.Slice")
    return pushSingle(inferSliceResult(op));
  if (name == "onnx.Upsample")
    return pushSingle(inferUpsampleResult(op));
  return failure();
}

/// Forward-propagate static shape information onto incomplete `onnx.*` results
/// within `body` until no further result can be refined.
///
/// A single in-program-order walk suffices for a straight-line body (every
/// operand is defined by an earlier op or a block arg, so a producer is
/// always ranked before its consumer is visited). The bounded fixed point is
/// a guard for any future shape in which a consumer precedes its producer in
/// walk order. A stored number is kept. A `?` is filled only when the rule
/// has a number, so the walk terminates within the cap.
static bool forwardInferUnrankedResults(func::FuncOp body) {
  static constexpr unsigned kMaxIters = 8;
  bool any = false;
  for (unsigned iter = 0; iter < kMaxIters; ++iter) {
    bool changed = false;
    body.walk([&](Operation *op) {
      SmallVector<Type> inferred;
      if (failed(inferUnrankedOnnxResults(op, inferred)) ||
          inferred.size() != op->getNumResults())
        return;
      for (auto [result, type] : llvm::zip(op->getResults(), inferred)) {
        Type merged = mergeInferredType(result.getType(), type);
        if (result.getType() == merged)
          continue;
        LLVM_DEBUG(DBGS() << "rank " << op->getName() << ": "
                          << result.getType() << " -> " << merged << "\n");
        result.setType(merged);
        ++NumOnnxResultsRanked;
        changed = true;
      }
    });
    if (!changed)
      return any;
    any = true;
  }
  LLVM_DEBUG(DBGS() << body.getSymName()
                    << ": forward inference hit the iteration cap\n");
  return any;
}

/// `hip.loop` in tensor mode requires every result type to equal its `$v_init`
/// operand type. Main-graph inference can refine that operand after the result
/// type was declared.
static bool syncLoopResultsToVInit(hip::LoopOp loopOp) {
  auto vInit = loopOp.getVInit();
  if (vInit.empty() || !isa<RankedTensorType>(vInit.front().getType()))
    return false;
  bool changed = false;
  for (auto [i, v] : llvm::enumerate(vInit)) {
    if (i >= loopOp.getNumResults() || !isa<RankedTensorType>(v.getType()))
      break;
    if (loopOp.getResult(i).getType() == v.getType())
      continue;
    loopOp.getResult(i).setType(v.getType());
    changed = true;
  }
  return changed;
}

/// Establish rank inside one outlined `hip.loop` body func, then reconcile its
/// signature. The body's argument and return layouts are fixed by
/// `OnnxLoopOutlinePass`:
///
///   args:    [0] ctx, [1] iter, [2] cond, [3 .. 3+N) v_carry, [3+N ..]
///   captures returns: v_carry occupies [0 .. N) when `cond_is_passthrough`,
///   else
///            [1 .. 1+N) with cond_out at slot 0.
static void inferLoopBodyShapes(func::FuncOp body, hip::LoopOp loopOp) {
  if (body.getBody().empty())
    return;
  Block &entry = body.getBody().front();
  Operation::operand_range vInit = loopOp.getVInit();
  static constexpr unsigned kArgVCarryStart = 3;

  // 1. Seed loop-carried block args from the loop op's v_init types.
  for (auto [i, v] : llvm::enumerate(vInit)) {
    unsigned argSlot = kArgVCarryStart + i;
    if (argSlot < entry.getNumArguments())
      entry.getArgument(argSlot).setType(v.getType());
  }

  // 2. Forward-infer unranked onnx.* results from their (seeded) operands.
  forwardInferUnrankedResults(body);

  // 3. Loop-contract backstop. A carried output with no forward rule, or one
  //    that is still less specific than $v_init, takes the $v_init type. The
  //    ONNX Loop spec requires those types to match.
  Operation *terminator = entry.getTerminator();
  unsigned resultVCarryStart = loopOp.getCondIsPassthrough() ? 0u : 1u;
  for (auto [i, v] : llvm::enumerate(vInit)) {
    unsigned slot = resultVCarryStart + i;
    if (slot >= terminator->getNumOperands())
      break;
    Value carried = terminator->getOperand(slot);
    Type merged = mergeInferredType(carried.getType(), v.getType());
    if (carried.getType() == merged)
      continue;
    LLVM_DEBUG(DBGS() << "backstop return slot " << slot << ": "
                      << carried.getType() << " -> " << merged << "\n");
    carried.setType(merged);
    ++NumLoopContractRanked;
  }

  // 4. Reconcile the signature so func.return matches the declared result
  //    types. Inputs follow the (seeded) entry block args; results follow the
  //    (now ranked) terminator operands.
  FunctionType reconciled = body.getFunctionType().clone(
      entry.getArgumentTypes(), terminator->getOperandTypes());
  if (reconciled != body.getFunctionType()) {
    body.setType(reconciled);
    ++NumBodyFuncsReconciled;
  }
}

struct InferLoopBodyShapesPass
    : public impl::InferLoopBodyShapesPassBase<InferLoopBodyShapesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallPtrSet<Operation *, 8> loopBodies;
    module.walk([&](hip::LoopOp loopOp) {
      if (auto body =
              module.lookupSymbol<func::FuncOp>(loopOp.getBodyFuncAttr()))
        loopBodies.insert(body);
    });

    // Main-graph inference can refine a $v_init after the body was seeded
    // (constant Slice -> hip.loop). Repeat so the loop result and the body
    // signature catch up, including a nested loop fed by the outer result.
    static constexpr unsigned kMaxIters = 8;
    for (unsigned iter = 0; iter < kMaxIters; ++iter) {
      bool changed = false;
      module.walk([&](func::FuncOp func) {
        if (func.isDeclaration() || func.getBody().empty() ||
            loopBodies.contains(func))
          return;
        if (forwardInferUnrankedResults(func))
          changed = true;
      });
      module.walk([&](hip::LoopOp loopOp) {
        if (syncLoopResultsToVInit(loopOp))
          changed = true;
        auto body = module.lookupSymbol<func::FuncOp>(loopOp.getBodyFuncAttr());
        if (!body)
          return;
        FunctionType before = body.getFunctionType();
        inferLoopBodyShapes(body, loopOp);
        if (body.getFunctionType() != before)
          changed = true;
      });
      if (!changed)
        break;
    }

    module.walk([&](func::FuncOp func) {
      if (func.isDeclaration() || func.getBody().empty() ||
          loopBodies.contains(func))
        return;
      Block &entry = func.getBody().front();
      Operation *terminator = entry.getTerminator();
      if (!terminator || terminator->getNumOperands() != func.getNumResults())
        return;
      FunctionType reconciled = func.getFunctionType().clone(
          entry.getArgumentTypes(), terminator->getOperandTypes());
      if (reconciled != func.getFunctionType()) {
        func.setType(reconciled);
        ++NumBodyFuncsReconciled;
      }
    });
  }
};

} // namespace
} // namespace hip
} // namespace mlir
