/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::hip {
#define GEN_PASS_DEF_FUSEROCMLIRPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

#define DEBUG_TYPE "fuse-rocmlir"

namespace {

// Rank-0 inline hip.constant used by fused pointwise ops (Relu zero, Clip
// bounds, residual Mul scale). File- and memory-backed carriers have no dense
// attr for in-kernel tosa.const and stay kernel arguments.
static bool isInlineScalarConstant(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 0)
    return false;
  auto constant = v.getDefiningOp<ConstantOp>();
  return constant && constant.getValueAttr();
}

// The kernel ABI rocMLIR expects for a scalar: MIGraphX gives one the shape
// {1}, and rock-flatten-tosa-func-args leaves rank-1 boundaries alone. A rank-0
// argument reaches rock.transforms_to_ptr with no coordinate to linearize
// (`affine_map -> ()`), which fails as "Transforms are not well formed";
// tensor<1xT> still carries the single index 0.
static Type promoteRankZero(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getRank() != 0)
    return type;
  return RankedTensorType::get({1}, tensorType.getElementType(),
                               tensorType.getEncoding());
}

static bool isaPointwiseOp(Operation *op) {
  return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp, TanhOp,
                         SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                         LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                         AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                         RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                         SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp>(op);
}

static bool isaAnchorOp(Operation *op) {
  return isa_and_present<ConvOp, MatmulOp, GemmOp>(op);
}

// MIGraphX fuse_mlir `reshaper_names()`, minus broadcast and multibroadcast
// (`make_conv_dot_reshaper_names` drops those on the conv/dot path). squeeze,
// flatten, and unsqueeze are tensor.collapse_shape / tensor.expand_shape here,
// and transpose is hip.transpose, which has an output buffer and so is a chain
// step of its own rather than one of these. tensor.reshape is left out:
// hip-to-tosa and rocMLIR have no lowering for it. `slice` is not in this set.
static bool isLayoutOp(Operation *op) {
  return isa_and_present<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op);
}

static Value layoutSource(Operation *op) {
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op))
    return expand.getSrc();
  return cast<tensor::CollapseShapeOp>(op).getSrc();
}

// hip-to-tosa converts only static shapes, and one op it cannot convert fails
// the whole rock kernel.
static bool hasStaticTensorTypes(Operation *op) {
  auto isStatic = [](Type type) {
    auto tensorType = dyn_cast<TensorType>(type);
    return !tensorType || tensorType.hasStaticShape();
  };
  return llvm::all_of(op->getOperandTypes(), isStatic) &&
         llvm::all_of(op->getResultTypes(), isStatic);
}

// Whether `op` can be cloned into a rock kernel as a chain step. A rank-0
// result is excluded: promoteRankZero reshapes only the kernel arguments, so
// the cloned ops would mix tensor<1xT> inputs with rank-0 inits.
static bool isFusibleChainOp(Operation *op) {
  bool scalarResult = llvm::any_of(op->getResultTypes(), [](Type type) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    return tensorType && tensorType.getRank() == 0;
  });
  return !scalarResult && hasStaticTensorTypes(op) &&
         isTosaExpressiblePointwiseOp(op);
}

// The op a chain ending at `from` grows into, or null where it stops. Every
// walk over a chain steps with this, so they agree on where chains end.
// expand_shape and collapse_shape have no output buffer, so they are stepped
// over, and collected in `layout`, only when a pointwise op or hip.transpose
// follows them.
static DestinationStyleOpInterface
nextChainOp(Operation *from, SmallVectorImpl<Operation *> *layout = nullptr) {
  if (!from->hasOneUse())
    return {};
  Operation *user = *from->user_begin();
  SmallVector<Operation *> skipped;
  while (isLayoutOp(user)) {
    if (!user->hasOneUse() || !hasStaticTensorTypes(user))
      return {};
    skipped.push_back(user);
    user = *user->user_begin();
  }
  if (!isaPointwiseOp(user) && !isa<TransposeOp>(user))
    return {};
  if (!isFusibleChainOp(user))
    return {};
  auto next = dyn_cast<DestinationStyleOpInterface>(user);
  if (next && layout)
    layout->append(skipped);
  return next;
}

// The op that `nextChainOp` steps from into `op` through `input`, if any.
static Operation *chainPredecessor(Operation *op, Value input) {
  Operation *def = input.getDefiningOp();
  while (isLayoutOp(def))
    def = layoutSource(def).getDefiningOp();
  if (!def || nextChainOp(def) != op)
    return nullptr;
  return def;
}

// Outline `start` and the single-use pointwise chain that consumes it into one
// rocMlir function. `start` is a conv/matmul/gemm or the head of a pointwise
// chain; rocMLIR compiles the latter with no FusionRoot (PR 122 elementwise).
static LogicalResult outlineRocMlirChain(DestinationStyleOpInterface start,
                                         PatternRewriter &rewriter,
                                         int *counter) {
  DestinationStyleOpInterface endOp = start;
  SetVector<Value> operands;
  SetVector<Operation *> ops;

  // Fail on dynamic shapes
  if (!hasStaticTensorTypes(start))
    return rewriter.notifyMatchFailure(start,
                                       "only static shapes are supported");

  // Match the pointwise chain, stepping through reshapers the way MIGraphX
  // `match::skip(conv_dot_reshaper_names)` does. hip.transpose itself may end
  // the chain. The chain stops before any op hip-to-tosa cannot convert, so
  // the prefix still compiles and the rest stays on the HIP path.
  auto advance = [&](Operation *from) -> bool {
    SmallVector<Operation *> layout;
    DestinationStyleOpInterface next = nextChainOp(from, &layout);
    if (!next)
      return false;
    ops.insert_range(layout);
    Value carried =
        layout.empty() ? from->getResult(0) : layout.back()->getResult(0);
    endOp = next;
    auto newOperands = endOp.getDpsInputs();
    if (auto it = llvm::find(newOperands, carried); it != newOperands.end())
      newOperands.erase(it);
    operands.insert_range(newOperands);
    for (auto init : endOp.getDpsInits()) {
      if (Operation *initOp = init.getDefiningOp())
        ops.insert(initOp);
    }
    ops.insert(endOp);
    return true;
  };

  {
    auto newOperands = endOp.getDpsInputs();
    operands.insert_range(newOperands);
    for (auto init : endOp.getDpsInits()) {
      // An init with no producer is a block argument, which cannot be cloned.
      // It is picked up as a kernel argument below instead.
      if (Operation *initOp = init.getDefiningOp())
        ops.insert(initOp);
    }
    ops.insert(endOp);
  }
  while (advance(endOp))
    ;

  // Remove the hip.context operand
  auto context = operands.front();
  if (!isa<mlir::hip::ContextType>(context.getType())) {
    return rewriter.notifyMatchFailure(start, "first operand not hip.context");
  }
  operands.erase(operands.begin());

  // Pointwise scalar literals belong in the outlined subgraph, not the
  // kernel ABI. A rank-0 buffer argument has no index for
  // rock.transforms_to_ptr. Clone only inline constants: a dynamic Clip
  // bound remains an argument.
  SetVector<Value> kernelOperands;
  for (Value operand : operands) {
    if (isInlineScalarConstant(operand)) {
      ops.insert(operand.getDefiningOp());
      continue;
    }
    kernelOperands.insert(operand);
  }
  operands = std::move(kernelOperands);

  // A bitsandbytes dequant that feeds only this anchor belongs in the same
  // rock kernel. HipToTosa turns it into a nibble lookup and a multiply, and
  // the matmul into tosa.matmul. A dynamic activation never reaches this
  // pattern, so that case stays a HIP dequant plus hipBLASLt.
  SmallVector<Value> dequantResults;
  for (Value operand : operands) {
    auto dequant = operand.getDefiningOp<DequantBnb4Op>();
    if (!dequant || !dequant->hasOneUse())
      continue;
    dequantResults.push_back(operand);
  }
  for (Value value : dequantResults) {
    auto dequant = value.getDefiningOp<DequantBnb4Op>();
    ops.insert(dequant);
    operands.remove(value);
    operands.insert(dequant.getPacked());
    operands.insert(dequant.getAbsmax());
  }

  // `ops` is cloned into an IsolatedFromAbove func, so every value it reads
  // has to resolve inside that func. The DPS inputs above cover the data,
  // but they do not cover what the inits are built from: an init is only a
  // bare tensor.empty when the anchor already has the rank rocMLIR wants.
  // A 1-D convolution does not -- convert-onnx-to-hip widens it to 2-D by
  // wrapping the operands *and the init* in tensor.expand_shape -- so the
  // init's producer is a reshape whose own operand is the empty. Cloning
  // just the reshape leaves it pointing at an empty back in main_graph.
  //
  // Absorb producers that carry no operands and no side effects, so a
  // materialised destination travels with the ops that use it instead of
  // becoming a kernel argument. That keeps the signature rocMLIR sees
  // unchanged for the shapes that already worked. Values already headed for
  // the argument list stay there: a non-scalar weight is passed in even
  // though hip.constant would otherwise qualify to be cloned.
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

  // Whatever is still read from outside has to be passed in, which costs one
  // more kernel argument and is what any init producer that cannot be
  // rematerialized -- a caller-supplied buffer, say -- falls back on.
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      if (operand != context && !ops.contains(operand.getDefiningOp()))
        operands.insert(operand);

  // Cloning follows this order, so a producer absorbed above has to come
  // before the op that reads it.
  ops = topologicalSort(ops);

  auto parentModule = start->getParentOfType<ModuleOp>();
  func::FuncOp newFunc;
  {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto funcType = rewriter.getFunctionType(
        llvm::map_to_vector(
            operands, [](Value v) { return promoteRankZero(v.getType()); }),
        endOp->getResultTypes());

    newFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                   "rocMlir" + std::to_string((*counter)++),
                                   funcType);
    newFunc->setAttr("rock.kernel", rewriter.getUnitAttr());
    newFunc->setAttr("rock.arch", rewriter.getStringAttr("gfx1151"));
    auto *funcBlock = newFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(funcBlock);
    IRMapping mapping;

    // Map the hip.context to ub.poison
    mapping.map(context,
                ub::PoisonOp::create(rewriter, rewriter.getUnknownLoc(),
                                     context.getType()));

    // Map other operands
    for (auto [idx, operand] : llvm::enumerate(operands)) {
      mapping.map(operand, funcBlock->getArgument(idx));
    }

    for (auto *op : ops) {
      rewriter.clone(*op, mapping);
    }
    auto returns = llvm::map_to_vector(
        endOp->getResults(), [&mapping](Value v) { return mapping.lookup(v); });
    func::ReturnOp::create(rewriter, rewriter.getUnknownLoc(), returns);
  }

  rewriter.setInsertionPointAfter(endOp);

  // The graph outside keeps its rank-0 value; only what crosses into the
  // kernel is reshaped, so the dispatch matches the signature built above.
  SmallVector<Value> dispatchOperands;
  dispatchOperands.reserve(operands.size());
  Value rankOneShape;
  for (Value operand : operands) {
    Type promoted = promoteRankZero(operand.getType());
    if (promoted == operand.getType()) {
      dispatchOperands.push_back(operand);
      continue;
    }
    if (!rankOneShape) {
      auto shapeType = RankedTensorType::get({1}, rewriter.getIndexType());
      rankOneShape = arith::ConstantOp::create(
          rewriter, rewriter.getUnknownLoc(), shapeType,
          DenseIntElementsAttr::get(shapeType, ArrayRef<int64_t>{1}));
    }
    dispatchOperands.push_back(tensor::ReshapeOp::create(
        rewriter, rewriter.getUnknownLoc(), promoted, operand, rankOneShape));
  }

  auto rocMlirOp =
      RocMlirOp::create(rewriter, rewriter.getUnknownLoc(),
                        endOp->getResultTypes(), SymbolRefAttr::get(newFunc),
                        context, dispatchOperands, endOp.getDpsInits().front());

  // The matched op may be the head of the chain rather than `endOp`. Erase
  // the originals once nothing reads them, or the greedy driver matches the
  // head again and outlines a second kernel.
  Operation *end = endOp.getOperation();
  rewriter.replaceOp(end, rocMlirOp);
  for (Operation *op : llvm::reverse(ops)) {
    if (op == end || !op->use_empty())
      continue;
    rewriter.eraseOp(op);
  }
  return success();
}

template <typename AnchorOp>
class FuseAnchorPointwise : public OpRewritePattern<AnchorOp> {
public:
  FuseAnchorPointwise(MLIRContext *context, int *counter,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<AnchorOp>(context, benefit), counter(counter) {}

  LogicalResult matchAndRewrite(AnchorOp anchorOp,
                                PatternRewriter &rewriter) const override {
    return outlineRocMlirChain(anchorOp, rewriter, counter);
  }

private:
  int *counter = nullptr;
};

// A conv, matmul, or gemm the anchor patterns outline, chain included.
static bool isOutlinedAnchor(Operation *op) {
  return isaAnchorOp(op) && hasStaticTensorTypes(op);
}

// Whether a chain started at another op takes in `op`. Anchors and fusible
// pointwise ops start chains; hip.transpose is only ever a step in one.
static bool isReachedByChain(Operation *op) {
  for (Value input : cast<DestinationStyleOpInterface>(op).getDpsInputs()) {
    Operation *pred = chainPredecessor(op, input);
    if (!pred)
      continue;
    if (isa<TransposeOp>(pred)
            ? isReachedByChain(pred)
            : isOutlinedAnchor(pred) ||
                  (isaPointwiseOp(pred) && isFusibleChainOp(pred)))
      return true;
  }
  return false;
}

// Whether the chain of an outlined anchor takes in `op`, either straight from
// the anchor or through the chain ops between them, as in conv -> exp -> add.
// Each op has at most one chain successor, so the ops this walks back over
// form a tree and each is visited once.
static bool isReachedByAnchorChain(Operation *op) {
  SmallVector<Operation *> worklist = {op};
  while (!worklist.empty()) {
    auto current = cast<DestinationStyleOpInterface>(worklist.pop_back_val());
    for (Value input : current.getDpsInputs()) {
      Operation *pred = chainPredecessor(current, input);
      if (!pred)
        continue;
      if (isOutlinedAnchor(pred))
        return true;
      if (isa<TransposeOp>(pred) ||
          (isaPointwiseOp(pred) && isFusibleChainOp(pred)))
        worklist.push_back(pred);
    }
  }
  return false;
}

// True when an anchor chain takes in some op of the chain starting at `head`.
// The anchor pattern fuses from there on, so the pointwise pattern must not
// outline it. The head's own inputs are not enough: a cast of a resize can
// sit in front of an add that consumes the conv. Every op of the chain lies
// on the way back from its last op, so one walk from there covers them all.
static bool chainJoinsAnchorChain(Operation *head) {
  Operation *last = head;
  while (Operation *next = nextChainOp(last).getOperation())
    last = next;
  return isReachedByAnchorChain(last);
}

static bool isIdentityCast(Operation *op) {
  auto castOp = dyn_cast<CastOp>(op);
  return castOp &&
         castOp.getInput().getType() == castOp->getResult(0).getType();
}

// rocMLIR recognises an elementwise kernel by its arith and math ops and has
// no tuning space for a kernel without them. hip.transpose lowers to a view
// and an identity hip.cast to its input, so a chain of only those has none.
static bool chainComputes(Operation *head) {
  for (Operation *op = head; op; op = nextChainOp(op).getOperation())
    if (!isa<TransposeOp>(op) && !isIdentityCast(op))
      return true;
  return false;
}

// A pointwise chain whose producer is not a conv, matmul, or gemm. The anchor
// patterns own a chain that starts at one of those, so this only claims the
// head of a chain they leave behind.
class FusePointwiseChain : public RewritePattern {
public:
  FusePointwiseChain(MLIRContext *context, int *counter)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isaPointwiseOp(op))
      return failure();
    if (!isFusibleChainOp(op))
      return rewriter.notifyMatchFailure(
          op, "hip-to-tosa cannot convert this op in a rock kernel");
    if (isReachedByChain(op))
      return rewriter.notifyMatchFailure(op, "not the head of a chain");
    if (chainJoinsAnchorChain(op))
      return rewriter.notifyMatchFailure(
          op, "pointwise chain is the epilogue of an anchor");
    if (!chainComputes(op))
      return rewriter.notifyMatchFailure(
          op, "chain has no arith or math op to form an elementwise kernel");
    return outlineRocMlirChain(cast<DestinationStyleOpInterface>(op), rewriter,
                               counter);
  }

private:
  int *counter = nullptr;
};

class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    int counter = 0;
    patterns.add<FuseAnchorPointwise<MatmulOp>, FuseAnchorPointwise<ConvOp>,
                 FuseAnchorPointwise<GemmOp>>(ctx, &counter);
    patterns.add<FusePointwiseChain>(ctx, &counter);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }

private:
  int counter = 0;
};

} // namespace

}; // namespace mlir::hip
