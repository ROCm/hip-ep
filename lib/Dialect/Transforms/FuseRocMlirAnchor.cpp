/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "GemmGemm.h"
#include "KernelOutline.h"
#include "QLinearPlan.h"
#include "QuantizedEpilogue.h"
#include "RocMlirElementwise.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Transforms/RegionUtils.h>

#include <functional>
#include <optional>

namespace mlir::hip {
#define GEN_PASS_DEF_FUSEROCMLIRANCHORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

#define DEBUG_TYPE "fuse-rocmlir-anchor"

namespace {

enum class Decomposition { None, QLinearConv, QLinearMatMul, DequantizedConv };

void sortByBlockOrder(SmallVectorImpl<Operation *> &ops) {
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
}

// The anchors and how each becomes the op that rocMLIR compiles. An i32
// matmul or convolution has an exact HIP kernel; outlining it asks rocMLIR to
// compile a non-quantized integer tosa op, which this pipeline does not
// establish. The i32 accumulator of a quantized anchor is its init, not an
// input.
std::optional<Decomposition> getAnchorKind(Operation *op) {
  if (auto conv = dyn_cast<QLinearConvOp>(op))
    return conv->getNumResults() == 1 && planQLinearConv(conv)
               ? std::optional(Decomposition::QLinearConv)
               : std::nullopt;
  if (auto matmul = dyn_cast<QLinearMatMulOp>(op))
    return matmul->getNumResults() == 1 && planQLinearMatMul(matmul)
               ? std::optional(Decomposition::QLinearMatMul)
               : std::nullopt;
  if (!isAnchorOp(op))
    return std::nullopt;
  if (auto conv = dyn_cast<ConvOp>(op); conv && planDequantizedConv(conv))
    return Decomposition::DequantizedConv;
  auto dps = cast<DestinationStyleOpInterface>(op);
  for (Value input : dps.getDpsInputs()) {
    auto shaped = dyn_cast<ShapedType>(input.getType());
    if (shaped && shaped.getElementType().isInteger(32))
      return std::nullopt;
  }
  for (Value operand : op->getOperands())
    if (auto type = dyn_cast<TensorType>(operand.getType());
        type && !type.hasStaticShape())
      return std::nullopt;
  return Decomposition::None;
}

// The activation operands a prologue may feed, as the anchor reads them once
// it is decomposed: a convolution of dequantizes reads the integer input.
SmallVector<std::pair<Value, bool>> getPrologueRoots(Operation *anchor,
                                                     Decomposition kind) {
  if (kind == Decomposition::DequantizedConv) {
    std::optional<DequantizedConvPlan> plan =
        planDequantizedConv(cast<ConvOp>(anchor));
    return {{plan->input, true}, {plan->weights, false}};
  }
  return getInputRoots(anchor);
}

// About how many times the anchor loads each activation element: once for
// each filter tap that covers it, and once for each tile of output columns.
// The tile width is a typical rocMLIR value; the estimate only ranks anchors.
int64_t estimateActivationReloads(Operation *anchor) {
  constexpr int64_t kColumnTile = 128;
  auto getI64s = [](ArrayAttr attr) {
    return llvm::map_to_vector(
        attr, [](Attribute a) { return cast<IntegerAttr>(a).getInt(); });
  };
  SmallVector<int64_t> kernel, strides;
  Value weights;
  if (auto conv = dyn_cast<ConvOp>(anchor)) {
    weights = conv.getWeights();
    strides = getI64s(conv.getStrides());
  } else if (auto conv = dyn_cast<QLinearConvOp>(anchor)) {
    weights = conv.getWeights();
    strides = getI64s(conv.getStrides());
  }
  if (weights)
    kernel = llvm::to_vector(
        cast<ShapedType>(weights.getType()).getShape().drop_front(2));
  auto result = cast<ShapedType>(anchor->getResult(0).getType());
  int64_t columns = weights ? result.getDimSize(1) : result.getShape().back();
  int64_t reloads = llvm::divideCeil(columns, kColumnTile);
  for (auto [i, size] : llvm::enumerate(kernel))
    reloads *= llvm::divideCeil(size, i < strides.size() ? strides[i] : 1);
  return std::max<int64_t>(reloads, 1);
}

// One kernel: an anchor and the modules it takes.
struct KernelPlan {
  Operation *anchor;
  Decomposition kind;
  // The modules that read the anchor's results, in block order.
  SmallVector<Operation *> epilogue;
  // The modules that feed the anchor's activation operands.
  SetVector<Operation *> prologue;
};

// The ops of one rock.kernel and the values it reads from outside. The chain
// is the anchor and its epilogue ops and modules in block order; a module
// stays one op here, and is inlined only in the outlined kernel.
class KernelRegion {
public:
  KernelRegion(Operation *anchor, const SetVector<Operation *> &prologue,
               const DenseSet<Operation *> &claimed)
      : anchor(anchor), prologue(prologue), claimed(claimed) {}

  Operation *getEnd() const { return end; }
  ArrayRef<Value> getResults() const { return results; }
  Value getContext() const { return context; }
  SetVector<Value> &getOperands() { return operands; }
  SetVector<Operation *> &getOps() { return ops; }

  void setChain(ArrayRef<Operation *> chain);
  // Returns why the region cannot be outlined, or null when it can; `moved`
  // then holds the readers to move below the dispatch.
  const char *complete(ArrayRef<std::pair<Value, bool>> roots,
                       SmallVector<Operation *> &moved);

private:
  bool takeContext();
  void cloneInlineScalars();
  void absorbDequantBnb4();
  void absorbInputProducers(ArrayRef<std::pair<Value, bool>> roots);
  void absorbOperandFreeProducers();
  void collectRemainingOperands();
  std::optional<SmallVector<Operation *>> readersToMove() const;

  Operation *anchor;
  const SetVector<Operation *> &prologue;
  // The ops that some plan takes; only this plan's prologue may be absorbed.
  const DenseSet<Operation *> &claimed;
  Operation *end = nullptr;
  SmallVector<Value> results;
  Value context;
  SetVector<Value> operands;
  SetVector<Operation *> ops;
  // What the epilogue reads from outside: the residuals, biases, and scales.
  SmallVector<Value> epilogueInputs;
};

// Each chain value read outside becomes a kernel result, in chain order.
void KernelRegion::setChain(ArrayRef<Operation *> chain) {
  SetVector<Operation *> inChain(chain.begin(), chain.end());
  for (Operation *op : chain) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    SmallVector<Value> inputs = dps ? SmallVector<Value>(dps.getDpsInputs())
                                    : SmallVector<Value>(op->getOperands());
    if (op != anchor) {
      llvm::erase_if(inputs, [&](Value input) {
        return inChain.contains(input.getDefiningOp());
      });
      llvm::append_range(epilogueInputs, inputs);
    }
    operands.insert_range(inputs);
    if (dps)
      for (Value init : dps.getDpsInits())
        // An init with no producer is a block argument, which cannot be
        // cloned. It is picked up as a kernel argument later instead.
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
    ops.insert(op);
  }
  end = chain.back();
  for (Operation *op : chain)
    for (Value result : op->getResults())
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !inChain.contains(user);
          }))
        results.push_back(result);
}

bool KernelRegion::takeContext() {
  if (operands.empty() || !isa<ContextType>(operands.front().getType()))
    return false;
  context = operands.front();
  operands.remove(context);
  return true;
}

// Pointwise scalar literals belong in the outlined subgraph, not the kernel
// ABI. Clone only inline constants, and for a quantized anchor the dequantizes
// and reciprocals of them its epilogue reads: a dynamic Clip bound remains an
// argument.
void KernelRegion::cloneInlineScalars() {
  SetVector<Value> kernelOperands;
  bool quantized = isQuantizedAnchor(anchor);
  for (Value operand : operands) {
    if (isInlineScalarConstant(operand) ||
        (quantized && isConstantScalarExpr(operand))) {
      collectConstantScalarExpr(operand, ops);
      continue;
    }
    kernelOperands.insert(operand);
  }
  operands = std::move(kernelOperands);
}

// A bitsandbytes dequant that feeds only this anchor belongs in the same rock
// kernel. HipToTosa turns it into a nibble lookup and a multiply.
void KernelRegion::absorbDequantBnb4() {
  SmallVector<Value> dequantResults;
  for (Value operand : operands) {
    auto dequant = operand.getDefiningOp<DequantBnb4Op>();
    if (dequant && dequant->hasOneUse())
      dequantResults.push_back(operand);
  }
  for (Value value : dequantResults) {
    auto dequant = value.getDefiningOp<DequantBnb4Op>();
    ops.insert(dequant);
    operands.remove(value);
    operands.insert(dequant.getPacked());
    operands.insert(dequant.getAbsmax());
  }
}

// Pulls the producers of kernel inputs into the kernel: the prologue modules
// of the plan, any input view, and on a root that permits it a unary
// pointwise op. A producer is absorbed only when the kernel is its only
// reader; one that keeps running for another reader saves nothing.
void KernelRegion::absorbInputProducers(
    ArrayRef<std::pair<Value, bool>> roots) {
  auto onlyReadInKernel = [&](Operation *op) {
    return llvm::all_of(op->getUsers(),
                        [&](Operation *user) { return ops.contains(user); });
  };
  SmallVector<Value> original(operands.begin(), operands.end());
  DenseMap<Value, SmallVector<Value>> replacedBy;
  SmallVector<std::pair<Value, bool>> worklist(roots.begin(), roots.end());
  for (Value input : epilogueInputs)
    worklist.push_back({input, false});

  while (!worklist.empty()) {
    auto [value, allowPointwise] = worklist.pop_back_val();
    if (!operands.contains(value))
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || ops.contains(producer) || !onlyReadInKernel(producer) ||
        (claimed.contains(producer) && !prologue.contains(producer)))
      continue;
    std::optional<SmallVector<Value>> dataInputs =
        getDataInputs(producer, context);
    if (!dataInputs)
      continue;
    SmallVector<Value> inputs;
    for (Value input : *dataInputs)
      if (!isInlineScalarConstant(input))
        inputs.push_back(input);
    bool absorb =
        prologue.contains(producer) || allowsInputView(producer) ||
        (allowPointwise && isFusablePrologue(producer) && inputs.size() == 1);
    if (!absorb || !llvm::all_of(inputs, isStaticTensor) ||
        !llvm::all_of(inputs, isKernelIOValue))
      continue;

    ops.insert(producer);
    for (Value result : producer->getResults())
      operands.remove(result);
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(producer))
      for (Value init : dps.getDpsInits())
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
    for (Value input : *dataInputs) {
      if (isInlineScalarConstant(input)) {
        ops.insert(input.getDefiningOp());
        continue;
      }
      operands.insert(input);
      replacedBy[value].push_back(input);
      worklist.push_back({input, allowPointwise});
    }
  }

  // An absorbed producer's inputs take the place of the value it produced,
  // so the kernel arguments keep the order of the anchor's operands.
  SetVector<Value> ordered;
  std::function<void(Value)> place = [&](Value value) {
    if (operands.contains(value))
      ordered.insert(value);
    for (Value input : replacedBy.lookup(value))
      place(input);
  };
  for (Value value : original)
    place(value);
  assert(ordered.size() == operands.size() && "kernel argument lost");
  operands = std::move(ordered);
}

// `ops` is cloned into an IsolatedFromAbove func, so every value it reads has
// to resolve inside that func: a materialised destination, such as the
// reshape of an empty that widens a 1-D convolution, travels with the ops that
// use it. Values already headed for the argument list stay there.
void KernelRegion::absorbOperandFreeProducers() {
  SmallVector<Operation *> worklist(ops.begin(), ops.end());
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      if (operands.contains(operand))
        continue;
      Operation *producer = operand.getDefiningOp();
      if (!producer || ops.contains(producer))
        continue;
      if (producer->getNumOperands() != 0 || !isMemoryEffectFree(producer))
        continue;
      ops.insert(producer);
      worklist.push_back(producer);
    }
  }
}

void KernelRegion::collectRemainingOperands() {
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      if (operand != context && !ops.contains(operand.getDefiningOp()))
        operands.insert(operand);
  // Cloning follows this order, so a producer has to come before its reader.
  ops = topologicalSort(ops);
}

// The dispatch goes where the end is, so an outside reader of an earlier
// result that sits above the end has to move below the dispatch, together
// with whatever reads it in turn. Returns them in block order, or nullopt if
// one feeds the kernel.
std::optional<SmallVector<Operation *>> KernelRegion::readersToMove() const {
  SmallVector<Operation *> moved;
  DenseSet<Value> escaped;
  Operation *first = nullptr;
  for (Value result : results) {
    Operation *def = result.getDefiningOp();
    if (def == end)
      continue;
    escaped.insert(result);
    if (!first || def->isBeforeInBlock(first))
      first = def;
  }
  if (!first)
    return moved;
  auto readsEscaped = [&](Operation *op) {
    WalkResult walk = op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (escaped.contains(operand))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return walk.wasInterrupted();
  };
  for (Operation &op :
       llvm::make_range(std::next(first->getIterator()), end->getIterator())) {
    if (ops.contains(&op) || !readsEscaped(&op))
      continue;
    moved.push_back(&op);
    escaped.insert(op.result_begin(), op.result_end());
  }
  for (Value operand : operands)
    if (escaped.contains(operand))
      return std::nullopt;
  return moved;
}

const char *KernelRegion::complete(ArrayRef<std::pair<Value, bool>> roots,
                                   SmallVector<Operation *> &moved) {
  if (!takeContext())
    return "first operand not hip.context";
  cloneInlineScalars();
  absorbDequantBnb4();
  absorbInputProducers(roots);
  absorbOperandFreeProducers();
  collectRemainingOperands();
  if (!llvm::all_of(operands, isKernelIOValue) ||
      !llvm::all_of(results, isKernelIOValue))
    return "a kernel argument or result has an integer wider than 32 bits";
  std::optional<SmallVector<Operation *>> readers = readersToMove();
  if (!readers)
    return "an outside reader of a kernel result feeds the kernel";
  moved = std::move(*readers);
  return nullptr;
}

/// rocMLIR's arithmetic uses signless integers. ONNX BOOL is ui8 in the
/// importer, but a Where condition is a one-byte 0/1 mask, so the outlined
/// kernel can safely view that argument as i8. The dispatch continues to pass
/// the original ui8 buffer; only the private kernel signature is normalized.
Type getKernelArgumentType(Value operand, const SetVector<Operation *> &ops) {
  Type type = promoteRankZero(operand.getType());
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isUnsignedInteger(8))
    return type;
  auto isCondition = [](Value value) {
    return llvm::any_of(value.getUsers(), [&](Operation *user) {
      auto where = dyn_cast<WhereOp>(user);
      return where && where.getCondition() == value;
    });
  };
  for (Operation *op : ops) {
    auto where = dyn_cast<WhereOp>(op);
    if (where && where.getCondition() == operand)
      return tensorType.clone(IntegerType::get(type.getContext(), 8));
    auto module = dyn_cast<RocMlirElementwiseOp>(op);
    if (!module)
      continue;
    for (auto [input, argument] :
         llvm::zip(module.getInputs(),
                   module.getBody().front().getArguments().drop_front()))
      if (input == operand && isCondition(argument))
        return tensorType.clone(IntegerType::get(type.getContext(), 8));
  }
  return type;
}

// The ops of one kernel, as the outline takes them: `ops` in clone order, the
// dispatch at `end`, and `moved` the outside readers to put below it.
struct KernelParts {
  Operation *end;
  Value context;
  const SetVector<Value> &operands;
  const SetVector<Operation *> &ops;
  ArrayRef<Value> results;
  ArrayRef<Operation *> moved;
};

// `rankOneShape` is the shape operand of every rank-0 promotion in the block,
// made once by the first kernel that needs it.
RocMlirOp outline(const KernelParts &parts, bool elementwise,
                  RewriterBase &rewriter, int &counter, StringRef arch,
                  Value &rankOneShape) {
  Operation *end = parts.end;
  Value context = parts.context;
  const SetVector<Value> &operands = parts.operands;
  const SetVector<Operation *> &ops = parts.ops;
  ArrayRef<Value> results = parts.results;
  auto resultTypes =
      llvm::map_to_vector(results, [](Value v) -> Type { return v.getType(); });
  Location loc = rewriter.getUnknownLoc();

  auto parentModule = end->getParentOfType<ModuleOp>();
  func::FuncOp kernel;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto funcType = rewriter.getFunctionType(
        llvm::map_to_vector(
            operands, [&](Value v) { return getKernelArgumentType(v, ops); }),
        resultTypes);
    kernel = func::FuncOp::create(
        rewriter, loc, "rocMlir" + std::to_string(counter++), funcType);
    kernel->setAttr("rock.kernel", rewriter.getUnitAttr());
    if (elementwise)
      kernel->setAttr("rock.elementwise", rewriter.getUnitAttr());
    if (!arch.empty())
      kernel->setAttr("rock.arch", rewriter.getStringAttr(arch));
    Block *body = kernel.addEntryBlock();
    rewriter.setInsertionPointToStart(body);
    IRMapping mapping;
    mapping.map(context,
                ub::PoisonOp::create(rewriter, loc, context.getType()));
    for (auto [index, operand] : llvm::enumerate(operands))
      mapping.map(operand, body->getArgument(index));
    for (Operation *op : ops)
      rewriter.clone(*op, mapping);
    func::ReturnOp::create(rewriter, loc,
                           llvm::map_to_vector(results, [&](Value v) {
                             return mapping.lookup(v);
                           }));
    SmallVector<RocMlirElementwiseOp> modules(
        body->getOps<RocMlirElementwiseOp>());
    for (RocMlirElementwiseOp module : modules)
      inlineRocMlirElementwise(module, rewriter);
  }

  Operation *last = end;
  for (Operation *op : parts.moved) {
    rewriter.moveOpAfter(op, last);
    last = op;
  }
  rewriter.setInsertionPointAfter(end);

  // The graph outside keeps its rank-0 value; only what crosses into the
  // kernel is reshaped, so the dispatch matches the signature built above.
  SmallVector<Value> dispatchOperands;
  for (Value operand : operands) {
    Type promoted = promoteRankZero(operand.getType());
    if (promoted == operand.getType()) {
      dispatchOperands.push_back(operand);
      continue;
    }
    if (!rankOneShape) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(end->getBlock());
      auto shapeType = RankedTensorType::get({1}, rewriter.getIndexType());
      rankOneShape = arith::ConstantOp::create(
          rewriter, loc, shapeType,
          DenseIntElementsAttr::get(shapeType, ArrayRef<int64_t>{1}));
    }
    dispatchOperands.push_back(tensor::ReshapeOp::create(
        rewriter, loc, promoted, operand, rankOneShape));
  }

  // A DPS result keeps its own init; a module result gets an empty one.
  SmallVector<Value> inits;
  for (Value result : results) {
    auto owner = cast<OpResult>(result);
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(owner.getOwner())) {
      inits.push_back(dps.getTiedOpOperand(owner)->get());
      continue;
    }
    auto type = cast<RankedTensorType>(result.getType());
    inits.push_back(tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                            type.getElementType()));
  }
  auto dispatch =
      RocMlirOp::create(rewriter, loc, resultTypes, SymbolRefAttr::get(kernel),
                        context, dispatchOperands, inits);
  for (auto [value, replacement] : llvm::zip(results, dispatch.getResults()))
    rewriter.replaceUsesWithIf(value, replacement, [&](OpOperand &use) {
      return !ops.contains(use.getOwner());
    });
  for (Operation *op : llvm::reverse(ops))
    if (op->use_empty() && (!isa<ConstantOp>(op) || isOpTriviallyDead(op)))
      rewriter.eraseOp(op);
  rewriteQuantizedBinaryOps(kernel);
  return dispatch;
}

class AnchorPlanner {
public:
  explicit AnchorPlanner(Block &block) : block(block) {}

  SmallVector<KernelPlan> plan() {
    SmallVector<KernelPlan> plans;
    for (Operation &op : block)
      if (std::optional<Decomposition> kind = getAnchorKind(&op))
        plans.push_back({&op, *kind, {}, {}});
    for (KernelPlan &plan : plans)
      planEpilogue(plan);
    for (KernelPlan &plan : plans)
      planPrologue(plan);
    return plans;
  }

  const DenseSet<Operation *> &getTaken() const { return taken; }

private:
  // Whether `module` can run in the epilogue of the kernel that `kernel`
  // holds. Every value it computes from a kernel value keeps that value's
  // shape, since rocMLIR cannot invert a broadcast of an epilogue value. Only
  // a transpose or a reshape inverts. The other inputs must not depend
  // on the kernel, or the kernel would read its own result. A quantized
  // anchor writes no float value that a dequantize widened: whatever would
  // return one stays outside.
  bool canTake(Operation *anchor, RocMlirElementwiseOp module,
               const SetVector<Operation *> &kernel,
               ArrayRef<Operation *> path) const {
    auto inKernel = [&](Operation *op) {
      return op && (kernel.contains(op) || llvm::is_contained(path, op));
    };
    Block &body = module.getBody().front();
    DenseSet<Value> fromKernel;
    for (auto [input, argument] :
         llvm::zip(module.getInputs(), body.getArguments().drop_front())) {
      if (inKernel(input.getDefiningOp()))
        fromKernel.insert(argument);
      else if (dependsOnChain(input, anchor, kernel))
        return false;
    }
    if (fromKernel.empty())
      return false;

    bool quantized = isQuantizedAnchor(anchor);
    DenseSet<Value> dequantized;
    for (Operation &op : body.without_terminator()) {
      bool widened = isa<DequantizeLinearOp>(op) ||
                     (!isa<QuantizeLinearOp>(op) &&
                      llvm::any_of(op.getOperands(), [&](Value v) {
                        return dequantized.contains(v);
                      }));
      if (widened)
        dequantized.insert(op.result_begin(), op.result_end());
      auto derived = llvm::make_filter_range(
          op.getOperands(), [&](Value v) { return fromKernel.contains(v); });
      if (derived.empty())
        continue;
      fromKernel.insert(op.result_begin(), op.result_end());
      if (getFusableViewKind(&op)) {
        if (!allowsForwardView(&op))
          return false;
        continue;
      }
      if (isa<DequantizeLinearOp>(op) && !quantized)
        return false;
      ArrayRef<int64_t> shape =
          cast<ShapedType>(op.getResult(0).getType()).getShape();
      if (llvm::any_of(derived, [&](Value v) {
            return cast<ShapedType>(v.getType()).getShape() != shape;
          }))
        return false;
    }
    if (quantized)
      for (Value value : body.getTerminator()->getOperands())
        if (dequantized.contains(value) &&
            isa<FloatType>(getElementTypeOrSelf(value.getType())))
          return false;
    return true;
  }

  // The reshapes without an init that lead from a kernel value, each read
  // only by the next, up to the op that reads the last one. The kernel cannot
  // store such a reshape, so it goes in only with that op.
  Operation *walkReshapes(Operation *op, SmallVector<Operation *> &path) const {
    path.clear();
    while (!isa<DestinationStyleOpInterface, RocMlirElementwiseOp>(op) &&
           allowsForwardView(op) && op->getNumResults() == 1 &&
           op->hasOneUse()) {
      path.push_back(op);
      op = *op->user_begin();
      if (op->getBlock() != &block || taken.contains(op))
        return nullptr;
    }
    return op;
  }

  // A transpose or reshape outside any module, which pass 1 leaves alone
  // because a module of views saves no kernel. Past the anchor it is free: the
  // kernel writes its result through the inverted view.
  bool canTakeView(Operation *view, const SetVector<Operation *> &kernel,
                   ArrayRef<Operation *> path) const {
    if (!isa<DestinationStyleOpInterface>(view) || !allowsForwardView(view) ||
        view->getNumResults() != 1 || !isKernelIOValue(view->getResult(0)))
      return false;
    std::optional<SmallVector<Value>> inputs = getDataInputs(view, Value());
    auto fromKernel = [&](Value v) {
      Operation *def = v.getDefiningOp();
      return isa<ContextType>(v.getType()) ||
             (def && (kernel.contains(def) || llvm::is_contained(path, def)));
    };
    return inputs && llvm::all_of(*inputs, fromKernel);
  }

  // Producer anchors first: a module that one anchor takes is not available
  // to the next. A quantized anchor continues through the modules that read
  // the modules it took, past each quantize and dequantize.
  void planEpilogue(KernelPlan &plan) {
    SetVector<Operation *> kernel;
    kernel.insert(plan.anchor);
    bool throughModules = isQuantizedAnchor(plan.anchor);
    bool grew = true;
    while (grew) {
      grew = false;
      SmallVector<Operation *> candidates;
      for (Operation *op : kernel)
        for (Operation *user : op->getUsers())
          if (user->getBlock() == &block && !kernel.contains(user) &&
              !taken.contains(user))
            candidates.push_back(user);
      sortByBlockOrder(candidates);
      candidates.erase(llvm::unique(candidates), candidates.end());
      SmallVector<Operation *> path;
      for (Operation *candidate : candidates) {
        if (taken.contains(candidate))
          continue;
        Operation *end = walkReshapes(candidate, path);
        if (!end)
          continue;
        auto module = dyn_cast<RocMlirElementwiseOp>(end);
        if (module ? !canTake(plan.anchor, module, kernel, path)
                   : !canTakeView(end, kernel, path))
          continue;
        path.push_back(end);
        for (Operation *op : path) {
          kernel.insert(op);
          taken.insert(op);
          plan.epilogue.push_back(op);
        }
        grew = throughModules;
      }
    }
    sortByBlockOrder(plan.epilogue);
  }

  // A module that only this anchor reads runs as the prologue of an
  // activation operand: the anchor loads the module inputs and computes the
  // module on the fly. Views in between are absorbed with it. The anchor
  // repeats those loads and that computation on every reload of the operand,
  // so a module with several inputs joins only while the full-size loads
  // times the reloads stay small; else a separate kernel computes it once.
  void planPrologue(KernelPlan &plan) {
    constexpr int64_t kMaxPrologueLoads = 9;
    int64_t reloads = estimateActivationReloads(plan.anchor);
    DenseSet<Operation *> readers = {plan.anchor};
    for (auto [root, allowPointwise] :
         getPrologueRoots(plan.anchor, plan.kind)) {
      if (!allowPointwise)
        continue;
      int64_t fullLoads = 1;
      SmallVector<Value> worklist = {root};
      while (!worklist.empty()) {
        Operation *producer = worklist.pop_back_val().getDefiningOp();
        if (!producer || producer->getBlock() != &block ||
            taken.contains(producer))
          continue;
        bool onlyReadByKernel =
            llvm::all_of(producer->getUsers(), [&](Operation *user) {
              return readers.contains(user);
            });
        if (!onlyReadByKernel)
          continue;
        if (auto module = dyn_cast<RocMlirElementwiseOp>(producer)) {
          SmallVector<Value> inputs =
              llvm::filter_to_vector(module.getInputs(), [](Value v) {
                return !isInlineScalarConstant(v);
              });
          if (inputs.size() > 1) {
            int64_t resultSize = 0;
            for (Type type : module->getResultTypes())
              resultSize =
                  std::max(resultSize, cast<ShapedType>(type).getNumElements());
            int64_t moduleLoads = llvm::count_if(inputs, [&](Value v) {
              return cast<ShapedType>(v.getType()).getNumElements() >=
                     resultSize;
            });
            if ((fullLoads + moduleLoads - 1) * reloads > kMaxPrologueLoads)
              continue;
            fullLoads += moduleLoads - 1;
          }
          taken.insert(module);
          plan.prologue.insert(module);
          readers.insert(module);
          llvm::append_range(worklist, inputs);
        } else if (allowsInputView(producer)) {
          readers.insert(producer);
          if (std::optional<SmallVector<Value>> inputs =
                  getDataInputs(producer, Value()))
            llvm::append_range(worklist, *inputs);
        }
      }
    }
  }

  Block &block;
  DenseSet<Operation *> taken;
};

// The chain of `plan` on the current IR: the anchor, then its epilogue
// modules.
SmallVector<Operation *> getChain(Operation *anchor,
                                  ArrayRef<Operation *> epilogue) {
  SmallVector<Operation *> chain = {anchor};
  llvm::append_range(chain, epilogue);
  return chain;
}

// Replaces a planned QLinear op or convolution of dequantizes with the integer
// anchor and its requantize ops, and returns them in block order, anchor
// first.
SmallVector<Operation *> decompose(const KernelPlan &plan,
                                   RewriterBase &rewriter) {
  Operation *op = plan.anchor;
  Block *block = op->getBlock();
  Operation *before = op->getPrevNode();
  Operation *after = op->getNextNode();
  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);
  if (auto conv = dyn_cast<QLinearConvOp>(op)) {
    QLinearPlan qplan = *planQLinearConv(conv);
    auto accType = cast<RankedTensorType>(conv.getOutput().getType())
                       .clone(rewriter.getI32Type());
    Value acc =
        QuantConvOp::create(
            rewriter, loc, TypeRange{accType}, conv.getCtx(), conv.getInput(),
            conv.getWeights(), emitTensorEmpty(rewriter, loc, accType),
            conv.getKernelShape(), conv.getStrides(), conv.getPads(),
            conv.getDilations(), conv.getGroupAttr())
            .getResult(0);
    rewriter.replaceOp(op,
                       emitRequantizeEpilogue(rewriter, loc, conv.getCtx(), acc,
                                              qplan, conv.getOutput()));
  } else if (auto matmul = dyn_cast<QLinearMatMulOp>(op)) {
    QLinearPlan qplan = *planQLinearMatMul(matmul);
    auto accType = cast<RankedTensorType>(matmul.getOutput().getType())
                       .clone(rewriter.getI32Type());
    Value acc =
        QuantMatmulOp::create(rewriter, loc, TypeRange{accType},
                              matmul.getCtx(), matmul.getA(), matmul.getB(),
                              emitTensorEmpty(rewriter, loc, accType))
            .getResult(0);
    rewriter.replaceOp(op,
                       emitRequantizeEpilogue(rewriter, loc, matmul.getCtx(),
                                              acc, qplan, matmul.getOutput()));
  } else {
    auto conv = cast<ConvOp>(op);
    DequantizedConvPlan dplan = *planDequantizedConv(conv);
    Value ctx = conv.getCtx();
    auto resultType = cast<RankedTensorType>(conv->getResult(0).getType());
    auto accType = resultType.clone(rewriter.getI32Type());
    Value acc = QuantConvOp::create(
                    rewriter, loc, TypeRange{accType}, ctx, dplan.input,
                    dplan.weights, emitTensorEmpty(rewriter, loc, accType),
                    conv.getKernelShape(), conv.getStrides(), conv.getPads(),
                    conv.getDilations(), conv.getGroupAttr())
                    .getResult(0);
    Value result = emitScaledAccumulator(rewriter, loc, ctx, acc,
                                         dplan.correction, dplan.multiplier);
    if (dplan.bias)
      result = AddOp::create(rewriter, loc, TypeRange{resultType}, ctx, result,
                             emitConstant(rewriter, loc, dplan.bias),
                             emitTensorEmpty(rewriter, loc, resultType))
                   .getResult(0);
    rewriter.replaceOp(op, result);
  }

  SmallVector<Operation *> chain;
  for (Operation *emitted = before ? before->getNextNode() : &block->front();
       emitted != after; emitted = emitted->getNextNode())
    if (isa<DestinationStyleOpInterface>(emitted) &&
        !isa<tensor::EmptyOp>(emitted))
      chain.push_back(emitted);
  return chain;
}

class FuseRocMlirAnchorPass
    : public impl::FuseRocMlirAnchorPassBase<FuseRocMlirAnchorPass> {
public:
  using impl::FuseRocMlirAnchorPassBase<
      FuseRocMlirAnchorPass>::FuseRocMlirAnchorPassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph" || funcOp.isExternal())
      return;
    Block &block = funcOp.getBody().front();
    AnchorPlanner planner(block);
    SmallVector<KernelPlan> plans = planner.plan();
    claimed = &planner.getTaken();
    rankOneShape = Value();
    // A module that no plan takes runs as HIP ops. Inlined now, its views
    // and unary ops can still be absorbed one by one into the kernels that
    // read them, which a module with several inputs or several readers
    // cannot be as a whole.
    // A module that saves a launch stays whole and becomes a kernel of its
    // own instead.
    IRRewriter rewriter(&getContext());
    for (RocMlirElementwiseOp module :
         llvm::make_early_inc_range(block.getOps<RocMlirElementwiseOp>()))
      if (!claimed->contains(module) && !outlinesAlone(module)) {
        rewriter.setInsertionPoint(module);
        inlineRocMlirElementwise(module, rewriter);
      }

    // The last anchor outlines first, so the kernels keep the names and the
    // order that the greedy fusion gave them.
    int counter = 0;
    for (KernelPlan &plan : llvm::reverse(plans))
      if (failed(outlinePlan(plan, rewriter, counter)))
        return signalPassFailure();

    SmallVector<RocMlirElementwiseOp> leftover(
        block.getOps<RocMlirElementwiseOp>());
    for (RocMlirElementwiseOp module : leftover) {
      if (outlinesAlone(module)) {
        module = peelBoundaryTransposes(module, rewriter);
        if (!module)
          continue;
        if (outlinesAlone(module)) {
          outlineElementwise(module, rewriter, counter);
          continue;
        }
      }
      rewriter.setInsertionPoint(module);
      inlineRocMlirElementwise(module, rewriter);
    }
    eraseDeadOps(block, rewriter);

    // hip.quant_conv and hip.quant_matmul have no lowering outside a rocMLIR
    // kernel.
    WalkResult remaining = funcOp.walk([](Operation *op) {
      if (!isa<QuantConvOp, QuantMatmulOp, RocMlirElementwiseOp>(op))
        return WalkResult::advance();
      op->emitOpError("was not outlined into a rocMLIR kernel");
      return WalkResult::interrupt();
    });
    if (remaining.wasInterrupted())
      return signalPassFailure();
    fuseGemmGemmKernels(funcOp, counter, gemmGemmAccelTypes);
  }

private:
  // A plan is checked on the IR as it is when its turn comes, since an
  // earlier kernel may have moved its readers. A QLinear op or a convolution
  // of dequantizes is decomposed only after that check passes, so a declined
  // op keeps its HIP kernel. Without the prologue modules a plan may still
  // outline; they are then inlined back with the other leftovers.
  LogicalResult outlinePlan(KernelPlan &plan, IRRewriter &rewriter,
                            int &counter) {
    auto check = [&](const SetVector<Operation *> &prologue) {
      KernelRegion trial(plan.anchor, prologue, *claimed);
      trial.setChain(getChain(plan.anchor, plan.epilogue));
      SmallVector<Operation *> moved;
      const char *reason = trial.complete(getInputRoots(plan.anchor), moved);
      LLVM_DEBUG(if (reason) llvm::dbgs()
                 << "declined " << *plan.anchor << ": " << reason << "\n");
      return reason == nullptr;
    };
    if (!check(plan.prologue)) {
      plan.prologue.clear();
      if (!check(plan.prologue)) {
        if (plan.epilogue.empty())
          return success();
        plan.epilogue.clear();
        if (!check(plan.prologue))
          return success();
      }
    }

    SmallVector<Operation *> chain;
    if (plan.kind == Decomposition::None) {
      chain = getChain(plan.anchor, plan.epilogue);
    } else {
      chain = decompose(plan, rewriter);
      llvm::append_range(chain, plan.epilogue);
    }
    KernelRegion region(chain.front(), plan.prologue, *claimed);
    region.setChain(chain);
    SmallVector<Operation *> moved;
    if (const char *reason =
            region.complete(getInputRoots(chain.front()), moved))
      return chain.front()->emitOpError("cannot be outlined: ") << reason;
    outline({region.getEnd(), region.getContext(), region.getOperands(),
             region.getOps(), region.getResults(), moved},
            /*elementwise=*/false, rewriter, counter, arch, rankOneShape);
    return success();
  }

  // A module outlines alone only when it saves a launch. A static reshape is a
  // free view on the host, so the module needs two other ops, and each op must
  // lower to TOSA outside an anchor kernel. rocMLIR's elementwise pipeline
  // cannot store a result that only transposes a kernel argument (through
  // reshapes) next to other results; such a module stays on the host.
  bool outlinesAlone(RocMlirElementwiseOp module) const {
    if (!standaloneElementwise)
      return false;
    Block &body = module.getBody().front();
    int launches = 0;
    for (Operation &op : body.without_terminator()) {
      if (isa<tensor::EmptyOp>(op) || isTosaLowerableReshape(&op))
        continue;
      if (!isTosaExpressiblePointwiseOp(&op) && !isTosaLowerableQdq(&op))
        return false;
      ++launches;
    }
    if (launches < 2)
      return false;
    auto movesArgument = [](Value result) {
      auto transpose = result.getDefiningOp<TransposeOp>();
      if (!transpose)
        return false;
      Value source = transpose.getInput();
      while (Operation *def = source.getDefiningOp()) {
        if (!isTosaLowerableReshape(def))
          return false;
        source = def->getOperand(0);
      }
      return true;
    };
    auto yield = cast<YieldOp>(body.getTerminator());
    return yield.getValues().size() == 1 ||
           llvm::none_of(yield.getValues(), movesArgument);
  }

  // rocMLIR tiles an elementwise kernel over its flattened output, so a
  // transpose that moves the innermost dimension makes one side of the kernel
  // access memory with a stride. Such a transpose at the edge of a module
  // runs as a HIP transpose instead, and the kernel keeps the layout that it
  // computes in. A peeled result transpose (with the reshapes around it) is
  // one that only the module results read; a peeled input transpose reads a
  // module input through reshapes. Returns the module that remains, or null
  // when its ops could not form one again and stay as HIP ops.
  RocMlirElementwiseOp peelBoundaryTransposes(RocMlirElementwiseOp module,
                                              IRRewriter &rewriter) {
    Block &body = module.getBody().front();
    Operation *yield = body.getTerminator();
    DenseSet<Value> yielded(yield->operand_begin(), yield->operand_end());
    auto movesInnermost = [](TransposeOp transpose) {
      ArrayAttr perm = transpose.getPerm();
      int64_t last = static_cast<int64_t>(perm.size()) - 1;
      return last > 0 && cast<IntegerAttr>(perm[last]).getInt() != last;
    };
    auto storable = [&](Value value) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape() || type.getRank() == 0)
        return false;
      if (auto intType = dyn_cast<IntegerType>(type.getElementType());
          intType && intType.getWidth() > 32)
        return false;
      return yielded.contains(value) ||
             isa_and_nonnull<DestinationStyleOpInterface>(
                 value.getDefiningOp());
    };

    DenseSet<Operation *> peel;
    for (TransposeOp transpose :
         llvm::make_early_inc_range(body.getOps<TransposeOp>())) {
      if (!movesInnermost(transpose))
        continue;
      // The reshapes between the transpose and what it reads.
      SmallVector<Operation *> above;
      Value source = transpose.getInput();
      while (Operation *def = source.getDefiningOp()) {
        if (!isTosaLowerableReshape(def))
          break;
        above.push_back(def);
        source = def->getOperand(0);
      }
      bool readsInput = isa<BlockArgument>(source);

      // The reshapes between the transpose and the yield, when only the yield
      // reads the chain.
      SmallVector<Operation *> below;
      bool onlyYielded = true;
      for (Operation *op = transpose;;) {
        if (!op->use_empty() && llvm::all_of(op->getUsers(), [&](Operation *u) {
              return u == yield;
            }))
          break;
        if (!op->hasOneUse() ||
            !isTosaLowerableReshape(*op->getUsers().begin())) {
          onlyYielded = false;
          break;
        }
        op = *op->getUsers().begin();
        below.push_back(op);
      }

      bool peelsResult = onlyYielded && (readsInput || storable(source));
      if (!peelsResult && !readsInput)
        continue;
      // A reshape that another module op also reads stays in the module, so
      // the transpose gets its own copy of the reshapes it reads through.
      if (llvm::any_of(above, [](Operation *op) { return !op->hasOneUse(); })) {
        rewriter.setInsertionPoint(transpose);
        Value copy = source;
        SmallVector<Operation *> copies;
        for (Operation *reshape : llvm::reverse(above)) {
          IRMapping mapping;
          mapping.map(reshape->getOperand(0), copy);
          Operation *clone = rewriter.clone(*reshape, mapping);
          copies.push_back(clone);
          copy = clone->getResult(0);
        }
        rewriter.modifyOpInPlace(
            transpose, [&] { transpose.getInputMutable().assign(copy); });
        above = std::move(copies);
      }
      peel.insert(transpose);
      peel.insert_range(above);
      if (peelsResult)
        peel.insert_range(below);
    }
    if (peel.empty())
      return module;

    SmallVector<Operation *> kept;
    bool computes = false;
    for (Operation &op : body.without_terminator()) {
      if (isa<tensor::EmptyOp>(op) || peel.contains(&op))
        continue;
      kept.push_back(&op);
      computes |= !isa<TransposeOp>(op) && !isTosaLowerableReshape(&op);
    }
    if (!computes)
      return module;

    rewriter.setInsertionPoint(module);
    inlineRocMlirElementwise(module, rewriter);
    return groupIntoRocMlirElementwise(kept, rewriter);
  }

  void outlineElementwise(RocMlirElementwiseOp module, IRRewriter &rewriter,
                          int &counter) {
    SetVector<Operation *> ops;
    SetVector<Value> operands;
    for (Value input : module.getInputs()) {
      if (isInlineScalarConstant(input))
        ops.insert(input.getDefiningOp());
      else
        operands.insert(input);
    }
    ops.insert(module);
    SmallVector<Value> results(module.getResults());
    outline({module, module.getCtx(), operands, ops, results, {}},
            /*elementwise=*/true, rewriter, counter, arch, rankOneShape);
  }

  const DenseSet<Operation *> *claimed = nullptr;
  Value rankOneShape;

  // Ops whose only readers moved into a kernel, such as the dequantizes of a
  // decomposed convolution.
  static void eraseDeadOps(Block &block, RewriterBase &rewriter) {
    bool erased = true;
    while (erased) {
      erased = false;
      for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
        if (isOpTriviallyDead(&op)) {
          rewriter.eraseOp(&op);
          erased = true;
        }
    }
  }
};

} // namespace
} // namespace mlir::hip
