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

#include <type_traits>

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

template <typename AnchorOp>
class FuseAnchorPointwise : public OpRewritePattern<AnchorOp> {
public:
  FuseAnchorPointwise(MLIRContext *context, int *counter,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<AnchorOp>(context, benefit), counter(counter) {}

  LogicalResult matchAndRewrite(AnchorOp anchorOp,
                                PatternRewriter &rewriter) const override {
    DestinationStyleOpInterface endOp = anchorOp;
    Operation *prevOp = nullptr;
    SetVector<Value> operands;
    SetVector<Operation *> ops;

    // An i32 matmul or convolution has an exact HIP kernel. Outlining it
    // asks rocMLIR to compile a non-quantized integer tosa op, which this
    // pipeline does not establish. Leave the op in main_graph.
    for (Type type : anchorOp->getOperandTypes()) {
      auto shaped = dyn_cast<ShapedType>(type);
      if (shaped && shaped.getElementType().isInteger(32))
        return rewriter.notifyMatchFailure(
            anchorOp, "i32 anchor stays on the HIP kernel");
    }

    // Fail on dynamic shapes
    for (Value operand : anchorOp->getOperands()) {
      if (auto tensorType = dyn_cast<TensorType>(operand.getType());
          tensorType && !tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(anchorOp,
                                           "only static shapes are supported");
    }

    // Match all pointwise-ops
    do {
      if (prevOp) {
        endOp = dyn_cast<DestinationStyleOpInterface>(*prevOp->user_begin());
      }
      auto newOperands = endOp.getDpsInputs();
      if (prevOp) {
        newOperands.erase(newOperands.begin() +
                          prevOp->use_begin()->getOperandNumber());
      }
      operands.insert_range(newOperands);
      for (auto init : endOp.getDpsInits()) {
        // An init with no producer is a block argument, which cannot be cloned.
        // It is picked up as a kernel argument below instead.
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
      }
      ops.insert(endOp);
      prevOp = endOp;
    } while (endOp->hasOneUse() && (isaPointwiseOp(*endOp->user_begin()) ||
                                    isaFusableView(*endOp->user_begin())));

    // Remove the hip.context operand
    auto context = operands.front();
    if (!isa<mlir::hip::ContextType>(context.getType())) {
      return rewriter.notifyMatchFailure(anchorOp,
                                         "first operand not hip.context");
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

    absorbInputProducers(anchorOp, context, operands, ops);

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

      for (auto *op : ops) {
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

  // A convolution reads and writes through rock.transform views, so a
  // transpose next to it becomes a layout of the kernel instead of a separate
  // copy (TosaToRock folds it into input_layout / output_layout). Matmul and
  // gemm anchors keep their current boundaries.
  static constexpr bool kFuseViews = std::is_same_v<AnchorOp, ConvOp>;

  bool isaFusableView(Operation *op) const {
    return kFuseViews && isa_and_present<TransposeOp>(op);
  }

  // Pulls the producers of the anchor's own inputs into the kernel: any
  // transpose, and on the activation a unary pointwise op whose only reader is
  // being absorbed (absorbing one with other readers would compute it twice).
  // Epilogue inputs are left alone; their transposes would not fold into the
  // conv layout.
  //
  // Before:
  //   %t0 = hip.transpose ins(%x : tensor<1x96x96x512xf16>) {perm = [0,3,1,2]}
  //   %c  = hip.conv ins(%t0, %w, %b) -> tensor<1x512x96x96xf16>
  //   %t1 = hip.transpose ins(%c) {perm = [0,2,3,1]}
  //   %a  = hip.add ins(%t1, %r) -> tensor<1x96x96x512xf16>
  // After:
  //   %a = hip.rocmlir @rocMlir0(%x, %w, %b, %r)
  //   func @rocMlir0 { transpose -> conv -> transpose -> add }
  void absorbInputProducers(AnchorOp anchorOp, Value context,
                            SetVector<Value> &operands,
                            SetVector<Operation *> &ops) const {
    if (!kFuseViews)
      return;
    auto isStatic = [](Value v) {
      auto type = dyn_cast<RankedTensorType>(v.getType());
      return type && type.hasStaticShape();
    };
    // An absorbed op keeps running outside the kernel while it has readers
    // there, so a pointwise op feeding it would not be saved, only duplicated.
    auto onlyReadInKernel = [&](Operation *op) {
      return llvm::all_of(op->getUsers(),
                          [&](Operation *user) { return ops.contains(user); });
    };
    SmallPtrSet<Operation *, 8> readOutside;
    // Only the activation takes a pointwise prologue: an elementwise op on the
    // filter would rerun in every workgroup that loads it.
    SmallVector<std::pair<Value, bool>> worklist;
    if constexpr (kFuseViews) {
      worklist.push_back({anchorOp.getInput(), true});
      worklist.push_back({anchorOp.getWeights(), false});
      if (Value bias = anchorOp.getBias())
        worklist.push_back({bias, false});
    }

    while (!worklist.empty()) {
      auto [value, allowPointwise] = worklist.pop_back_val();
      if (!operands.contains(value))
        continue;
      auto producer = value.getDefiningOp<DestinationStyleOpInterface>();
      if (!producer || ops.contains(producer))
        continue;
      SmallVector<Value> inputs;
      for (Value input : producer.getDpsInputs())
        if (input != context && !isInlineScalarConstant(input))
          inputs.push_back(input);
      bool keepsRunning =
          !onlyReadInKernel(producer) ||
          llvm::any_of(producer->getUsers(), [&](Operation *user) {
            return readOutside.contains(user);
          });
      bool absorb = isaFusableView(producer) ||
                    (allowPointwise && isaPointwiseOp(producer) &&
                     inputs.size() == 1 && !keepsRunning);
      if (!absorb || !llvm::all_of(inputs, isStatic))
        continue;

      if (keepsRunning)
        readOutside.insert(producer);
      ops.insert(producer);
      operands.remove(value);
      for (Value init : producer.getDpsInits())
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
      for (Value input : producer.getDpsInputs()) {
        if (input == context)
          continue;
        if (isInlineScalarConstant(input)) {
          ops.insert(input.getDefiningOp());
          continue;
        }
        operands.insert(input);
        worklist.push_back({input, allowPointwise});
      }
    }
  }

  bool isaPointwiseOp(Operation *op) const {
    return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp,
                           TanhOp, SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                           LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                           AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                           RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                           SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp>(op);
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

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }

private:
  int counter = 0;
};

} // namespace

}; // namespace mlir::hip
