/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
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

// An inline hip.constant has a dense value ConstantConverter can turn into
// tosa.const. File- and memory-backed carriers are Pure and operand-free too,
// but they name a byte range nothing has read, so they stay kernel arguments.
static bool isInlineDenseConstant(Value v) {
  auto constant = v.getDefiningOp<ConstantOp>();
  return constant && constant.getValueAttr();
}

// Rank-0 inline hip.constant used by fused pointwise ops (Relu zero, Clip
// bounds, residual Mul scale). A rank-0 buffer argument has no index for
// rock.transforms_to_ptr, so only the inline form is cloned.
static bool isInlineScalarConstant(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 0)
    return false;
  return isInlineDenseConstant(v);
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

// rocMLIR's blockwise load rejects f64. An f64 value keeps its op on the HIP
// kernel. An f64 hip.matmul and hip.gemm stay on hipBLASLt as HIP_R_64F.
static bool isF64Type(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  Type element = shaped ? shaped.getElementType() : type;
  return element.isF64();
}

// A cast is fused into a rocMLIR kernel only when every float it touches is
// f16, bf16, or f32. Otherwise it stays a HIP cast outside the kernel.
static bool isRocMlirCastEndpoint(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  Type element = shaped ? shaped.getElementType() : type;
  if (!isa<FloatType>(element))
    return true;
  return element.isF16() || element.isBF16() || element.isF32();
}

/// rocMLIR's arithmetic uses signless integers. ONNX BOOL is ui8 in the
/// importer, but a Where condition is a one-byte 0/1 mask, so the outlined
/// kernel can safely view that argument as i8. The dispatch continues to pass
/// the original ui8 buffer; only the private kernel signature is normalized.
static Type getRocMlirKernelArgumentType(Value operand,
                                         const SetVector<Operation *> &ops) {
  Type type = promoteRankZero(operand.getType());
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isUnsignedInteger(8))
    return type;

  for (Operation *op : ops) {
    auto where = dyn_cast<WhereOp>(op);
    if (where && where.getCondition() == operand)
      return tensorType.clone(IntegerType::get(type.getContext(), 8));
  }
  return type;
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

    // An f64 dot stays on hipBLASLt; rocMLIR's blockwise load rejects f64.
    for (Type type : anchorOp->getOperandTypes())
      if (isF64Type(type))
        return rewriter.notifyMatchFailure(
            anchorOp, "f64 anchor stays on the HIP runtime");
    for (Type type : anchorOp->getResultTypes())
      if (isF64Type(type))
        return rewriter.notifyMatchFailure(
            anchorOp, "f64 anchor stays on the HIP runtime");

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
    } while (endOp->hasOneUse() && isaPointwiseOp(*endOp->user_begin()));

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

    // hip.dequant_bnb4 stays outside the kernel. Absorbing it makes HipToTosa
    // emit a tosa.gather of the codebook over every weight element, which
    // rocMLIR marks illegal. The matmul sees the dequantized f32 weight.

    // MatMulNBits reaches this anchor already split into unpack_int4 and a
    // blocked dequantize_linear feeding hip.matmul {transB = 1}. Both belong
    // in the rock kernel: HipToTosa turns them into a nibble extract, a block
    // broadcast, and tosa.matmul. An inline dense zero point, including the
    // synthesized default 8, is cloned by the walk below. A file- or
    // memory-backed zero point has no dense value to lower, so it is a kernel
    // argument, as are the packed bytes and the scales.
    SmallVector<Value> nbitsWeights;
    for (Value operand : operands) {
      auto dequant = operand.getDefiningOp<DequantizeLinearOp>();
      if (!dequant || !dequant->hasOneUse() || dequant.getPackedInt4() ||
          dequant.getBlockSize() == 0)
        continue;
      auto unpack = dequant.getInput().getDefiningOp<UnpackInt4Op>();
      if (!unpack || !unpack->hasOneUse())
        continue;
      nbitsWeights.push_back(operand);
    }
    for (Value value : nbitsWeights) {
      auto dequant = value.getDefiningOp<DequantizeLinearOp>();
      auto unpack = dequant.getInput().getDefiningOp<UnpackInt4Op>();
      ops.insert(dequant);
      ops.insert(unpack);
      operands.remove(value);
      operands.insert(unpack.getPacked());
      operands.insert(dequant.getScale());
      if (Value zp = dequant.getZeroPoint()) {
        if (!isInlineDenseConstant(zp))
          operands.insert(zp);
      }
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

    auto parentModule = anchorOp->template getParentOfType<ModuleOp>();
    func::FuncOp newFunc;
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      auto funcType = rewriter.getFunctionType(
          llvm::map_to_vector(
              operands,
              [&ops](Value v) { return getRocMlirKernelArgumentType(v, ops); }),
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

  bool isaPointwiseOp(Operation *op) const {
    if (auto cast = dyn_cast_if_present<CastOp>(op))
      return llvm::all_of(cast->getOperandTypes(), isRocMlirCastEndpoint) &&
             llvm::all_of(cast->getResultTypes(), isRocMlirCastEndpoint);
    return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp,
                           TanhOp, SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                           LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                           AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                           RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                           SubOp, LessOp, SignOp, ModOp, WhereOp>(op);
  }

private:
  int *counter = nullptr;
};

// hip.matmul_nbits -> unpack_int4, blocked dequantize_linear, hip.matmul
// {transB = 1}, and a bias add when one is present. The matmul pattern then
// outlines that chain. Anything this rejects stays on the custom kernel.
class DecomposeMatMulNBits : public OpRewritePattern<MatMulNBitsOp> {
public:
  explicit DecomposeMatMulNBits(MLIRContext *context)
      : OpRewritePattern<MatMulNBitsOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MatMulNBitsOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    if (op.getBits() != 4)
      return rewriter.notifyMatchFailure(op, "only bits=4 is split");
    if (op.getGIdx())
      return rewriter.notifyMatchFailure(op, "g_idx stays on the HIP kernel");

    int64_t k = op.getK();
    int64_t n = op.getN();
    int64_t blockSize = op.getBlockSize();
    if (blockSize < 16 || (blockSize & (blockSize - 1)) != 0)
      return rewriter.notifyMatchFailure(
          op, "block_size must be a power of 2 and >= 16");
    if (k <= 0 || n <= 0 || k % blockSize != 0)
      return rewriter.notifyMatchFailure(
          op, "K must be positive and divisible by block_size");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto aType = dyn_cast<RankedTensorType>(op.getA().getType());
    auto bType = dyn_cast<RankedTensorType>(op.getB().getType());
    auto scaleType = dyn_cast<RankedTensorType>(op.getScales().getType());
    if (!resultType || !resultType.hasStaticShape() || !aType ||
        !aType.hasStaticShape() || !bType || !bType.hasStaticShape() ||
        !scaleType || !scaleType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static shapes are split");
    Type elemType = resultType.getElementType();
    if ((!elemType.isF16() && !elemType.isF32()) ||
        aType.getElementType() != elemType)
      return rewriter.notifyMatchFailure(
          op, "activation and result must be f16 or f32");
    if (!scaleType.getElementType().isF16() &&
        !scaleType.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "scales must be f16 or f32");
    if (aType.getRank() < 2 || aType.getShape().back() != k)
      return rewriter.notifyMatchFailure(op, "A trailing dim must be K");
    if (resultType.getRank() != aType.getRank() ||
        resultType.getShape().drop_back() != aType.getShape().drop_back() ||
        resultType.getShape().back() != n)
      return rewriter.notifyMatchFailure(op, "result shape must be A with N");

    int64_t kBlocks = k / blockSize;
    int64_t blob = blockSize / 2;
    if (bType.getRank() != 3 || bType.getDimSize(0) != n ||
        bType.getDimSize(1) != kBlocks || bType.getDimSize(2) != blob ||
        !bType.getElementType().isUnsignedInteger(8))
      return rewriter.notifyMatchFailure(
          op, "B must be packed ui8 [N, k_blocks, block_size/2]");
    if (scaleType.getNumElements() != n * kBlocks)
      return rewriter.notifyMatchFailure(op, "scales must have N * k_blocks");

    Value zp = op.getZeroPoints();
    if (zp) {
      // zp_elem_size 1 is a packed nibble stream, which this split does not
      // unpack. One zero point per block is passed through.
      if (op.getZpElemSize() == 1)
        return rewriter.notifyMatchFailure(
            op, "packed zero points stay on the HIP kernel");
      auto zpType = dyn_cast<RankedTensorType>(zp.getType());
      if (!zpType || !zpType.hasStaticShape() ||
          zpType.getNumElements() != n * kBlocks)
        return rewriter.notifyMatchFailure(
            op, "zero point must be static with N * k_blocks values");
      Type zpElem = zpType.getElementType();
      if (!isa<FloatType>(zpElem) && !isa<IntegerType>(zpElem))
        return rewriter.notifyMatchFailure(op, "unsupported zero point type");
    }
    if (Value bias = op.getBias()) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || !biasType.hasStaticShape() ||
          biasType.getElementType() != elemType ||
          biasType.getRank() > resultType.getRank())
        return rewriter.notifyMatchFailure(op, "bias is not broadcastable");
      int64_t offset = resultType.getRank() - biasType.getRank();
      for (auto [idx, dim] : llvm::enumerate(biasType.getShape())) {
        int64_t resultDim = resultType.getDimSize(offset + idx);
        if (dim != 1 && dim != resultDim)
          return rewriter.notifyMatchFailure(op, "bias is not broadcastable");
      }
    }

    Location loc = op.getLoc();
    Value ctx = op.getHandle();
    auto ui8 =
        IntegerType::get(rewriter.getContext(), 8, IntegerType::Unsigned);
    auto packedType = RankedTensorType::get({n, kBlocks * blob}, ui8);
    SmallVector<ReassociationIndices, 2> reassoc(2);
    reassoc[0].push_back(0);
    reassoc[1].push_back(1);
    reassoc[1].push_back(2);
    Value flat = tensor::CollapseShapeOp::create(rewriter, loc, packedType,
                                                 op.getB(), reassoc);

    auto unpackedType = RankedTensorType::get({n, k}, ui8);
    Value unpackInit =
        tensor::EmptyOp::create(rewriter, loc, unpackedType.getShape(), ui8);
    Value unpacked =
        UnpackInt4Op::create(rewriter, loc, TypeRange{unpackedType}, ctx, flat,
                             unpackInit)
            .getResult(0);

    if (!zp) {
      auto zpType = RankedTensorType::get({n, kBlocks}, elemType);
      zp = ConstantOp::create(
          rewriter, loc, zpType,
          DenseElementsAttr::get(zpType, rewriter.getFloatAttr(elemType, 8.0)));
    }

    auto weightType = RankedTensorType::get({n, k}, elemType);
    Value weightInit =
        tensor::EmptyOp::create(rewriter, loc, weightType.getShape(), elemType);
    Value weight =
        DequantizeLinearOp::create(rewriter, loc, TypeRange{weightType}, ctx,
                                   unpacked, op.getScales(), zp, weightInit,
                                   rewriter.getI64IntegerAttr(/*axis=*/1),
                                   rewriter.getI64IntegerAttr(blockSize))
            .getResult(0);

    Value matmulInit =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), elemType);
    SmallVector<NamedAttribute> attrs = {
        rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(1))};
    SmallVector<Value> matmulOperands = {ctx, op.getA(), weight, matmulInit};
    Value y =
        MatmulOp::create(rewriter, loc, matmulOperands, attrs).getResult(0);

    Value result = y;
    if (Value bias = op.getBias()) {
      Value addInit = tensor::EmptyOp::create(rewriter, loc,
                                              resultType.getShape(), elemType);
      result = AddOp::create(rewriter, loc, resultType, ctx, y, bias, addInit)
                   .getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
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
    // Benefit 2 so the split lands before the matmul anchor is outlined.
    patterns.add<DecomposeMatMulNBits>(ctx);
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
