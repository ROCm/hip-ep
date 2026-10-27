/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"
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

#include <optional>
#include <type_traits>
#include <utility>

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

static bool isStaticTensor(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  return type && type.hasStaticShape();
}

// Elementwise ops HipToTosa lowers to TOSA elementwise ops, which rocMLIR
// fuses into the anchor's epilogue or prologue.
static bool isFusablePointwise(Operation *op) {
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

static std::optional<ViewKind> getFusableViewKind(Operation *op) {
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

// Which neighbours of an anchor may join its kernel, beyond the forward
// pointwise chain every anchor takes.
struct FusionPolicy {
  // ViewKind bits the forward chain and the input absorption may take.
  unsigned viewKinds = 0;
  // A unary pointwise producer of the activation.
  bool activationPrologue = false;

  bool allowsView(Operation *op) const {
    std::optional<ViewKind> kind = getFusableViewKind(op);
    return kind && (viewKinds & static_cast<unsigned>(*kind));
  }
  bool absorbsInputs() const { return viewKinds != 0 || activationPrologue; }
};

// rocMLIR reads every kernel input through rock.transform maps, so a view on
// one becomes an index remap instead of a separate copy: a transpose next to
// a convolution folds into its input_layout / output_layout, and a reshape,
// slice, or broadcast on any input folds into that input's map. Matmul and
// gemm anchors keep their current boundaries.
template <typename AnchorOp> FusionPolicy getFusionPolicy(bool fuseViews) {
  FusionPolicy policy;
  if (std::is_same_v<AnchorOp, ConvOp> && fuseViews) {
    policy.viewKinds = static_cast<unsigned>(ViewKind::Transpose) |
                       static_cast<unsigned>(ViewKind::Reshape) |
                       static_cast<unsigned>(ViewKind::Slice) |
                       static_cast<unsigned>(ViewKind::Broadcast);
    policy.activationPrologue = true;
  }
  return policy;
}

// Where input absorption starts, with whether each root may take a pointwise
// prologue. Only the activation does: an elementwise op on the filter would
// rerun in every workgroup that loads it.
template <typename AnchorOp>
SmallVector<std::pair<Value, bool>> getInputRoots(AnchorOp anchorOp) {
  SmallVector<std::pair<Value, bool>> roots;
  if constexpr (std::is_same_v<AnchorOp, ConvOp>) {
    roots.push_back({anchorOp.getInput(), true});
    roots.push_back({anchorOp.getWeights(), false});
    if (Value bias = anchorOp.getBias())
      roots.push_back({bias, false});
  }
  return roots;
}

// The ops one rock.kernel holds and the values it reads from outside, grown
// outward from an anchor. The steps run in the order matchAndRewrite calls
// them; kernel argument order follows insertion order, so reordering them
// changes the kernel signature.
class KernelRegion {
public:
  KernelRegion(Operation *anchor, const FusionPolicy &policy)
      : anchor(anchor), policy(policy) {}

  // The op whose results the kernel returns; a DPS op, so its init becomes
  // the dispatch's outs.
  DestinationStyleOpInterface getEnd() const { return end; }
  Value getContext() const { return context; }
  SetVector<Value> &getOperands() { return operands; }
  SetVector<Operation *> &getOps() { return ops; }

  // Follows single-use pointwise ops, and views the policy allows, from the
  // anchor forward.
  void growForward();
  // Drops the leading hip.context the anchor's inputs carry.
  bool takeContext();
  // Moves inline scalar constants into the kernel.
  void cloneInlineScalars();
  // A bitsandbytes dequant that feeds only this anchor belongs in the same
  // rock kernel. HipToTosa turns it into a nibble lookup and a multiply, and
  // the matmul into tosa.matmul. A dynamic activation never reaches this
  // pattern, so that case stays a HIP dequant plus hipBLASLt.
  void absorbDequantBnb4();
  void absorbInputProducers(ArrayRef<std::pair<Value, bool>> roots);
  void absorbOperandFreeProducers();
  // Whatever is still read from outside has to be passed in, which costs one
  // more kernel argument and is what any init producer that cannot be
  // rematerialized -- a caller-supplied buffer, say -- falls back on.
  void collectRemainingOperands();

private:
  Operation *anchor;
  const FusionPolicy &policy;
  DestinationStyleOpInterface end;
  Value context;
  SetVector<Value> operands;
  SetVector<Operation *> ops;
  // What the forward chain past the anchor reads from outside: the residuals,
  // biases, and scales of the epilogue.
  SmallVector<Value> epilogueInputs;
};

void KernelRegion::growForward() {
  SmallVector<Operation *> chain = {anchor};
  while (chain.back()->hasOneUse()) {
    Operation *user = *chain.back()->user_begin();
    if (!isFusablePointwise(user) && !policy.allowsView(user))
      break;
    chain.push_back(user);
  }
  // A view outside the kernel is free, and one with no init cannot supply the
  // dispatch's outs.
  while (!isa<DestinationStyleOpInterface>(chain.back()))
    chain.pop_back();

  Operation *prev = nullptr;
  for (Operation *op : chain) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    SmallVector<Value> inputs = dps ? SmallVector<Value>(dps.getDpsInputs())
                                    : SmallVector<Value>(op->getOperands());
    if (prev) {
      inputs.erase(inputs.begin() + prev->use_begin()->getOperandNumber());
      llvm::append_range(epilogueInputs, inputs);
    }
    operands.insert_range(inputs);
    if (dps) {
      for (Value init : dps.getDpsInits()) {
        // An init with no producer is a block argument, which cannot be
        // cloned. It is picked up as a kernel argument later instead.
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
      }
      end = dps;
    }
    ops.insert(op);
    prev = op;
  }
}

bool KernelRegion::takeContext() {
  if (operands.empty() || !isa<ContextType>(operands.front().getType()))
    return false;
  context = operands.front();
  operands.erase(operands.begin());
  return true;
}

// Pointwise scalar literals belong in the outlined subgraph, not the kernel
// ABI. A rank-0 buffer argument has no index for rock.transforms_to_ptr. Clone
// only inline constants: a dynamic Clip bound remains an argument.
void KernelRegion::cloneInlineScalars() {
  SetVector<Value> kernelOperands;
  for (Value operand : operands) {
    if (isInlineScalarConstant(operand)) {
      ops.insert(operand.getDefiningOp());
      continue;
    }
    kernelOperands.insert(operand);
  }
  operands = std::move(kernelOperands);
}

void KernelRegion::absorbDequantBnb4() {
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
}

// The values an absorbed producer reads as data, which become kernel arguments
// unless absorbed in turn. A hip.expand's target shape is not data: HipToTosa
// takes the broadcast from the result type, so the shape only has to resolve
// inside the kernel, and absorbOperandFreeProducers clones a constant one.
// Returns nullopt for an expand whose shape is computed at run time.
static std::optional<SmallVector<Value>> getDataInputs(Operation *producer,
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

// Pulls the producers of kernel inputs into the kernel: any view the policy
// allows, on the anchor's inputs and on the epilogue's, and on a root that
// permits it a unary pointwise op. A producer is absorbed only when the kernel
// is its only reader. One that keeps running for another reader saves nothing
// -- a hip.transpose outside rocMLIR is a real copy -- and the kernel would
// become a second reader of the producer's input, which stops that input's
// own kernel from growing forward through the producer.
//
// Before:
//   %t0 = hip.transpose ins(%x : tensor<1x96x96x512xf16>) {perm = [0,3,1,2]}
//   %c  = hip.conv ins(%t0, %w, %b) -> tensor<1x512x96x96xf16>
//   %t1 = hip.transpose ins(%c) {perm = [0,2,3,1]}
//   %rb = hip.expand ins(%r : tensor<1x1x1x512xf16>, %shape)
//   %a  = hip.add ins(%t1, %rb) -> tensor<1x96x96x512xf16>
// After:
//   %a = hip.rocmlir @rocMlir0(%x, %w, %b, %r)
//   func @rocMlir0 { transpose -> conv -> transpose, expand -> add }
void KernelRegion::absorbInputProducers(
    ArrayRef<std::pair<Value, bool>> roots) {
  if (!policy.absorbsInputs())
    return;
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
    if (!producer || ops.contains(producer) || !onlyReadInKernel(producer))
      continue;
    std::optional<SmallVector<Value>> dataInputs =
        getDataInputs(producer, context);
    if (!dataInputs)
      continue;
    SmallVector<Value> inputs;
    for (Value input : *dataInputs)
      if (!isInlineScalarConstant(input))
        inputs.push_back(input);
    bool absorb = policy.allowsView(producer) ||
                  (allowPointwise && policy.activationPrologue &&
                   isFusablePointwise(producer) && inputs.size() == 1);
    if (!absorb || !llvm::all_of(inputs, isStaticTensor))
      continue;

    ops.insert(producer);
    operands.remove(value);
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
// to resolve inside that func. The DPS inputs above cover the data, but they do
// not cover what the inits are built from: an init is only a bare tensor.empty
// when the anchor already has the rank rocMLIR wants. A 1-D convolution does
// not -- convert-onnx-to-hip widens it to 2-D by wrapping the operands *and the
// init* in tensor.expand_shape -- so the init's producer is a reshape whose own
// operand is the empty. Cloning just the reshape leaves it pointing at an empty
// back in main_graph.
//
// Absorb producers that carry no operands and no side effects, so a
// materialised destination travels with the ops that use it instead of
// becoming a kernel argument. That keeps the signature rocMLIR sees unchanged
// for the shapes that already worked. Values already headed for the argument
// list stay there: a non-scalar weight is passed in even though hip.constant
// would otherwise qualify to be cloned.
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
  // Cloning follows this order, so a producer absorbed above has to come
  // before the op that reads it.
  ops = topologicalSort(ops);
}

template <typename AnchorOp>
class FuseAnchorPointwise : public OpRewritePattern<AnchorOp> {
public:
  FuseAnchorPointwise(MLIRContext *context, int *counter, bool fuseViews,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<AnchorOp>(context, benefit), counter(counter),
        policy(getFusionPolicy<AnchorOp>(fuseViews)) {}

  LogicalResult matchAndRewrite(AnchorOp anchorOp,
                                PatternRewriter &rewriter) const override {
    // Fail on dynamic shapes
    for (Value operand : anchorOp->getOperands()) {
      if (auto tensorType = dyn_cast<TensorType>(operand.getType());
          tensorType && !tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(anchorOp,
                                           "only static shapes are supported");
    }

    KernelRegion region(anchorOp, policy);
    region.growForward();
    if (!region.takeContext())
      return rewriter.notifyMatchFailure(anchorOp,
                                         "first operand not hip.context");
    region.cloneInlineScalars();
    region.absorbDequantBnb4();
    region.absorbInputProducers(getInputRoots(anchorOp));
    region.absorbOperandFreeProducers();
    region.collectRemainingOperands();

    DestinationStyleOpInterface endOp = region.getEnd();
    Value context = region.getContext();
    SetVector<Value> &operands = region.getOperands();

    auto parentModule = anchorOp->template getParentOfType<ModuleOp>();
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

      for (auto *op : region.getOps()) {
        rewriter.clone(*op, mapping);
      }
      auto returns =
          llvm::map_to_vector(endOp->getResults(), [&mapping](Value v) {
            return mapping.lookup(v);
          });
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

    auto rocMlirOp = RocMlirOp::create(
        rewriter, rewriter.getUnknownLoc(), endOp->getResultTypes(),
        SymbolRefAttr::get(newFunc), context, dispatchOperands,
        endOp.getDpsInits().front());

    rewriter.replaceOp(endOp, rocMlirOp);
    return success();
  }

private:
  int *counter = nullptr;
  FusionPolicy policy;
};

class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  using impl::FuseROCMlirPassBase<FuseROCMlirPass>::FuseROCMlirPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    int counter = 0;
    patterns.add<FuseAnchorPointwise<MatmulOp>, FuseAnchorPointwise<ConvOp>,
                 FuseAnchorPointwise<GemmOp>>(ctx, &counter, fuseViews);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

}; // namespace mlir::hip
