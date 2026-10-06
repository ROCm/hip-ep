/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- RocMlirFusion.cpp --------------------------------------------------===//
//
// Native constraints/rewrites backing RocMlirFusion.pdll.
//
// The anchor+pointwise outlining that FuseROCMlir.cpp used to do in one
// monolithic C++ pattern is decomposed into three *local*, composable rewrites
// that the greedy driver sequences into arbitrary-length chains:
//
//   1. outlinePointwise          a lone pointwise op at the end of a chain is
//                                extracted into its own func.func and replaced
//                                by a hip.pointwise dispatch.
//   2. fusePointwiseIntoConsumer a pointwise op feeding a hip.pointwise is
//                                inlined into that op's function (growing the
//                                fused subgraph one op at a time, tail first).
//   3. fuseAnchorIntoConsumer    a matmul/conv/gemm feeding a hip.pointwise is
//                                absorbed the same way; the function is marked
//                                a rock kernel and the dispatch becomes
//                                hip.rocmlir.
//
// PDLL owns only the match surface; these bodies are native because
// creating/editing a func.func and its signature cannot be expressed
// declaratively.
//===----------------------------------------------------------------------===//

#include "RocMlir/RocMlirFusion.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <atomic>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser/Parser.h>

namespace mlir::hip::rocmlir {

namespace {

// Process-wide monotonic ids for the outlined kernels. Shared across every
// rewrite so a module that both outlines a standalone anchor and fuses another
// anchor cannot mint two @rocMlir0 symbols.
int nextRocMlirId() {
  static std::atomic<int> counter{0};
  return counter++;
}
int nextPointwiseId() {
  static std::atomic<int> counter{0};
  return counter++;
}

//===----------------------------------------------------------------------===//
// Shared predicates (carried over from FuseROCMlir.cpp)
//===----------------------------------------------------------------------===//

bool isInlineScalarConstant(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 0)
    return false;
  auto constant = v.getDefiningOp<ConstantOp>();
  return constant && constant.getValueAttr();
}

Type promoteRankZero(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getRank() != 0)
    return type;
  return RankedTensorType::get({1}, tensorType.getElementType(),
                               tensorType.getEncoding());
}

bool isaPointwiseOp(Operation *op) {
  return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp, TanhOp,
                         SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                         LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                         AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                         RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                         SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp>(op);
}

bool isAnchorOp(Operation *op) {
  return isa_and_present<MatmulOp, ConvOp, GemmOp>(op);
}

//===----------------------------------------------------------------------===//
// Subgraph closure
//===----------------------------------------------------------------------===//

// Everything needed to clone `seed` (a DPS op) into an isolated function: the
// ops to clone (seed, its init producers, inline scalar constants, and any
// operand-free side-effect-free producers it reads), the external data values
// that must become function arguments, and the hip.context operand.
struct Closure {
  SetVector<Operation *> ops;
  SmallVector<Value> inputs;
  Value ctx;
};

Closure collectClosure(Operation *seed) {
  Closure c;
  auto dps = cast<DestinationStyleOpInterface>(seed);

  SetVector<Value> operands;
  for (Value init : dps.getDpsInits())
    if (Operation *initOp = init.getDefiningOp())
      c.ops.insert(initOp);
  operands.insert_range(dps.getDpsInputs());
  c.ops.insert(seed);

  // Operand 0 of every hip op is the hip.context input.
  c.ctx = operands.front();
  operands.erase(operands.begin());

  // Pointwise scalar literals belong inside the subgraph, not the kernel ABI.
  SetVector<Value> kept;
  for (Value v : operands) {
    if (isInlineScalarConstant(v)) {
      c.ops.insert(v.getDefiningOp());
      continue;
    }
    kept.insert(v);
  }
  operands = std::move(kept);

  // Absorb operand-free side-effect-free producers (e.g. a reshape-of-empty
  // init) so a materialized destination travels with the ops that use it.
  SmallVector<Operation *> worklist(c.ops.begin(), c.ops.end());
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value v : op->getOperands()) {
      if (operands.contains(v))
        continue;
      Operation *producer = v.getDefiningOp();
      if (!producer || c.ops.contains(producer))
        continue;
      if (producer->getNumOperands() != 0 || !isMemoryEffectFree(producer))
        continue;
      c.ops.insert(producer);
      worklist.push_back(producer);
    }
  }

  // Anything still read from outside has to be passed in.
  for (Operation *op : c.ops)
    for (Value v : op->getOperands())
      if (v != c.ctx && !c.ops.contains(v.getDefiningOp()))
        operands.insert(v);

  c.inputs.assign(operands.begin(), operands.end());
  c.ops = topologicalSort(c.ops);
  return c;
}

//===----------------------------------------------------------------------===//
// Shared: inline a producer op into a hip.pointwise function in place
//===----------------------------------------------------------------------===//

struct AbsorbResult {
  func::FuncOp func;
  SmallVector<Value> newInputs;
};

// Clones `producer`'s closure into the front of the consumer dispatch's
// function and rewires the block argument that stood for `producer`'s result to
// the cloned computation. Returns the new dispatch input list: `producer`'s
// external inputs first, then the consumer's surviving inputs (the edge
// removed). Producer-first keeps the absorbed anchor/producer operands ahead of
// the ones the consumer contributed, matching the ABI order the monolithic
// pattern produced. The function signature is updated; `producer` is left dead
// for the caller.
AbsorbResult absorbProducer(PatternRewriter &rewriter, Operation *producer,
                            PointwiseOp consumer) {
  auto module = consumer->getParentOfType<ModuleOp>();
  auto func = module.lookupSymbol<func::FuncOp>(consumer.getKernelAttr());
  Block &body = func.front();

  SmallVector<Value> oldInputs(consumer.getInputs().begin(),
                               consumer.getInputs().end());
  Value edge = producer->getResult(0);
  // producer feeds `consumer` at exactly one input slot (single-use guard).
  unsigned edgeIdx =
      *llvm::find_if(llvm::seq<unsigned>(0, oldInputs.size()),
                     [&](unsigned k) { return oldInputs[k] == edge; });

  Closure c = collectClosure(producer);

  // Producer inputs that are not already a surviving consumer input become new
  // arguments, inserted at the front so they lead the signature.
  auto survivingSlotOf = [&](Value v) -> int {
    for (unsigned k : llvm::seq<unsigned>(0, oldInputs.size()))
      if (k != edgeIdx && oldInputs[k] == v)
        return static_cast<int>(k);
    return -1;
  };
  SmallVector<Value> prepended;
  for (Value in : c.inputs)
    if (survivingSlotOf(in) < 0)
      prepended.push_back(in);
  for (auto [i, in] : llvm::enumerate(prepended))
    body.insertArgument(i, in.getType(), rewriter.getUnknownLoc());
  unsigned numNew = prepended.size();

  rewriter.setInsertionPointToStart(&body);
  IRMapping mapping;
  mapping.map(c.ctx, ub::PoisonOp::create(rewriter, rewriter.getUnknownLoc(),
                                          c.ctx.getType()));
  for (Value in : c.inputs) {
    Value arg;
    for (unsigned i : llvm::seq<unsigned>(0, numNew))
      if (prepended[i] == in) {
        arg = body.getArgument(i);
        break;
      }
    if (!arg)
      arg = body.getArgument(numNew + survivingSlotOf(in));
    mapping.map(in, arg);
  }

  for (Operation *op : c.ops)
    rewriter.clone(*op, mapping);

  // The arg that carried producer's result now resolves to the inlined chain.
  BlockArgument edgeArg = body.getArgument(numNew + edgeIdx);
  rewriter.replaceAllUsesWith(edgeArg, mapping.lookup(edge));
  body.eraseArgument(numNew + edgeIdx);

  AbsorbResult result;
  result.func = func;
  result.newInputs.append(prepended.begin(), prepended.end());
  for (unsigned k : llvm::seq<unsigned>(0, oldInputs.size()))
    if (k != edgeIdx)
      result.newInputs.push_back(oldInputs[k]);

  func.setFunctionType(
      rewriter.getFunctionType(body.getArgumentTypes(), func.getResultTypes()));
  return result;
}

// Promote a rank-0 function argument to rank-1 (the ABI rocMLIR expects for a
// scalar: rock.transforms_to_ptr has no coordinate to linearize a rank-0
// boundary, tensor<1xT> carries the single index 0). The pointwise op reading
// the argument broadcasts the rank-1 value, so no in-body reshape is needed.
// Returns the dispatch operand (a rank-1 reshape of `dispatchInput`), created
// at the rewriter's current insertion point -- which the caller must leave in
// the graph next to the dispatch, not inside the kernel body.
Value promoteScalarArg(PatternRewriter &rewriter, BlockArgument arg,
                       Value dispatchInput) {
  Type rank1 = promoteRankZero(arg.getType());
  arg.setType(rank1);

  auto shapeTy = RankedTensorType::get({1}, rewriter.getIndexType());
  Value shape = arith::ConstantOp::create(
      rewriter, rewriter.getUnknownLoc(), shapeTy,
      DenseIntElementsAttr::get(shapeTy, ArrayRef<int64_t>{1}));
  return tensor::ReshapeOp::create(rewriter, rewriter.getUnknownLoc(), rank1,
                                   dispatchInput, shape);
}

// Promote every rank-0 kernel argument to the rank-1 scalar ABI and rewrite the
// matching dispatch operands. The rewriter insertion point must sit in the
// graph next to the dispatch; the rank-1 reshapes are emitted there, and on
// return the insertion point is left after the last of them so the dispatch is
// built downstream of its operands.
void applyScalarPromotion(PatternRewriter &rewriter, func::FuncOp func,
                          SmallVectorImpl<Value> &dispatchInputs) {
  Block &body = func.front();
  for (unsigned i : llvm::seq<unsigned>(0, dispatchInputs.size())) {
    BlockArgument arg = body.getArgument(i);
    if (promoteRankZero(arg.getType()) == arg.getType())
      continue;
    dispatchInputs[i] = promoteScalarArg(rewriter, arg, dispatchInputs[i]);
  }
  func.setFunctionType(
      rewriter.getFunctionType(body.getArgumentTypes(), func.getResultTypes()));
}

} // namespace

//===----------------------------------------------------------------------===//
// Native PDL constraints
//===----------------------------------------------------------------------===//

bool isPointwiseOp(Operation *op) { return isaPointwiseOp(op); }

// True when `op` is the last pointwise op in a chain: nothing downstream is
// another pointwise op or an existing hip.pointwise dispatch (those are handled
// by fusePointwiseIntoConsumer). Such an op is the one that gets outlined; its
// pointwise predecessors then fold into it one at a time.
bool isPointwiseChainTerminus(Operation *op) {
  if (!isaPointwiseOp(op))
    return false;
  if (op->hasOneUse()) {
    Operation *user = *op->user_begin();
    if (isaPointwiseOp(user) || isa<PointwiseOp>(user))
      return false;
  }
  return true;
}

// True when `op` has exactly one use and that user is a hip.pointwise dispatch,
// so the op can be inlined into the dispatch's function.
bool hasSinglePointwiseConsumer(Operation *op) {
  return op->hasOneUse() && isa<PointwiseOp>(*op->user_begin());
}

// Fusable anchor: matmul/conv/gemm, first operand is hip.context, all tensor
// operands statically shaped.
bool isFusableRocMlirAnchor(Operation *op) {
  if (!isAnchorOp(op) || op->getNumOperands() == 0 ||
      !isa<ContextType>(op->getOperand(0).getType()))
    return false;
  for (Value operand : op->getOperands())
    if (auto t = dyn_cast<TensorType>(operand.getType());
        t && !t.hasStaticShape())
      return false;
  return true;
}

// True when a fusable anchor terminates its chain: it has no single pointwise /
// hip.pointwise consumer to be absorbed into, so it is outlined directly into
// its own hip.rocmlir rather than waiting to be folded in by
// fuseAnchorIntoConsumer. An anchor that feeds a pointwise op is left for that
// path (the consumer is outlined first, then the anchor folds into it).
bool isRocMlirAnchorTerminus(Operation *op) {
  if (!isFusableRocMlirAnchor(op))
    return false;
  if (op->hasOneUse()) {
    Operation *user = *op->user_begin();
    if (isaPointwiseOp(user) || isa<PointwiseOp>(user))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Native PDL rewrites
//===----------------------------------------------------------------------===//

void outlinePointwise(PatternRewriter &rewriter, Operation *pointOp) {
  Closure c = collectClosure(pointOp);
  auto module = pointOp->getParentOfType<ModuleOp>();

  func::FuncOp func;
  {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcType = rewriter.getFunctionType(
        llvm::map_to_vector(c.inputs, [](Value v) { return v.getType(); }),
        pointOp->getResultTypes());
    func = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                "pointwise" + std::to_string(nextPointwiseId()),
                                funcType);
    Block *body = func.addEntryBlock();
    rewriter.setInsertionPointToStart(body);

    IRMapping mapping;
    mapping.map(c.ctx, ub::PoisonOp::create(rewriter, rewriter.getUnknownLoc(),
                                            c.ctx.getType()));
    for (auto [idx, in] : llvm::enumerate(c.inputs))
      mapping.map(in, body->getArgument(idx));
    for (Operation *op : c.ops)
      rewriter.clone(*op, mapping);
    auto returns = llvm::map_to_vector(
        pointOp->getResults(), [&](Value v) { return mapping.lookup(v); });
    func::ReturnOp::create(rewriter, rewriter.getUnknownLoc(), returns);
  }

  rewriter.setInsertionPointAfter(pointOp);
  auto init = cast<DestinationStyleOpInterface>(pointOp).getDpsInits().front();
  auto dispatch = PointwiseOp::create(
      rewriter, rewriter.getUnknownLoc(), pointOp->getResultTypes(),
      SymbolRefAttr::get(func), c.ctx, c.inputs, init);
  rewriter.replaceOp(pointOp, dispatch);
}

void fusePointwiseIntoConsumer(PatternRewriter &rewriter, Operation *producer) {
  auto consumer = cast<PointwiseOp>(*producer->user_begin());
  AbsorbResult a = absorbProducer(rewriter, producer, consumer);

  rewriter.setInsertionPointAfter(consumer);
  auto fused =
      PointwiseOp::create(rewriter, consumer.getLoc(),
                          consumer->getResultTypes(), consumer.getKernelAttr(),
                          consumer.getCtx(), a.newInputs, consumer.getOutput());
  rewriter.replaceOp(consumer, fused);
  rewriter.eraseOp(producer);
}

void fuseAnchorIntoConsumer(PatternRewriter &rewriter, Operation *anchor) {
  auto consumer = cast<PointwiseOp>(*anchor->user_begin());
  Value ctx = consumer.getCtx();
  Value output = consumer.getOutput();
  SmallVector<Type> resultTypes(consumer->getResultTypes());

  AbsorbResult a = absorbProducer(rewriter, anchor, consumer);
  func::FuncOp func = a.func;

  // The function is now a rock kernel.
  func.setName("rocMlir" + std::to_string(nextRocMlirId()));
  func->setAttr("rock.kernel", rewriter.getUnitAttr());
  func->setAttr("rock.arch", rewriter.getStringAttr("gfx1151"));

  // rocMLIR has no coordinate to linearize a rank-0 boundary, so scalars cross
  // as tensor<1xT>; only the crossing value is reshaped. The reshapes and the
  // dispatch are built in the graph, just after the consumer being replaced.
  rewriter.setInsertionPointAfter(consumer);
  SmallVector<Value> dispatchInputs(a.newInputs.begin(), a.newInputs.end());
  applyScalarPromotion(rewriter, func, dispatchInputs);

  auto dispatch =
      RocMlirOp::create(rewriter, consumer.getLoc(), resultTypes,
                        SymbolRefAttr::get(func), ctx, dispatchInputs, output);
  rewriter.replaceOp(consumer, dispatch);
  rewriter.eraseOp(anchor);
}

// A fusable anchor with no pointwise consumer to fold into: outline it (and the
// producers of its init) straight into its own hip.rocmlir kernel. Mirrors
// fuseAnchorIntoConsumer's kernel shape without an intervening hip.pointwise.
void outlineRocMlirAnchor(PatternRewriter &rewriter, Operation *anchor) {
  Closure c = collectClosure(anchor);
  auto module = anchor->getParentOfType<ModuleOp>();
  Value output =
      cast<DestinationStyleOpInterface>(anchor).getDpsInits().front();

  func::FuncOp func;
  {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcType = rewriter.getFunctionType(
        llvm::map_to_vector(c.inputs, [](Value v) { return v.getType(); }),
        anchor->getResultTypes());
    func = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                "rocMlir" + std::to_string(nextRocMlirId()),
                                funcType);
    func->setAttr("rock.kernel", rewriter.getUnitAttr());
    func->setAttr("rock.arch", rewriter.getStringAttr("gfx1151"));
    Block *body = func.addEntryBlock();
    rewriter.setInsertionPointToStart(body);

    IRMapping mapping;
    mapping.map(c.ctx, ub::PoisonOp::create(rewriter, rewriter.getUnknownLoc(),
                                            c.ctx.getType()));
    for (auto [idx, in] : llvm::enumerate(c.inputs))
      mapping.map(in, body->getArgument(idx));
    for (Operation *op : c.ops)
      rewriter.clone(*op, mapping);
    auto returns = llvm::map_to_vector(
        anchor->getResults(), [&](Value v) { return mapping.lookup(v); });
    func::ReturnOp::create(rewriter, rewriter.getUnknownLoc(), returns);
  }

  rewriter.setInsertionPointAfter(anchor);
  SmallVector<Value> dispatchInputs(c.inputs.begin(), c.inputs.end());
  applyScalarPromotion(rewriter, func, dispatchInputs);

  auto dispatch = RocMlirOp::create(
      rewriter, rewriter.getUnknownLoc(), anchor->getResultTypes(),
      SymbolRefAttr::get(func), c.ctx, dispatchInputs, output);
  rewriter.replaceOp(anchor, dispatch);
}

//===----------------------------------------------------------------------===//
// Generated patterns
//===----------------------------------------------------------------------===//

// mlir-pdll -x cpp emits the three pattern structs plus
// populateGeneratedPDLLPatterns, and registers each trampoline above by name.
#include "RocMlirFusion.pdll.h.inc"

void populateRocMlirFusionPatterns(RewritePatternSet &patterns) {
  populateGeneratedPDLLPatterns(patterns);
}

} // namespace mlir::hip::rocmlir
