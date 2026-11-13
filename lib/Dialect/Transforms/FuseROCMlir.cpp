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

// A static integer nearest upsample is a repeated read of the input: with
// asymmetric coordinates and floor rounding, output o copies input o / scale.
// round_prefer_floor matches that repeat only for scales 1 and 2, because a
// larger scale rounds some samples in the block to the next input pixel.
// Returns the resize input when `value` is that upsample and its only user is
// this convolution's input. The view is rebuilt inside the outlined kernel so
// rocMLIR folds it into the convolution instead of launching hip.resize.
static Value nearestRepeatInput(Value value, ConvOp conv) {
  auto resize = value.getDefiningOp<ResizeOp>();
  if (!resize || !resize->hasOneUse() || conv.getInput() != value)
    return {};
  if (!resize.getRuntimeScales().empty())
    return {};
  if (resize.getMode() != 0 || resize.getCoordTransform() != 1)
    return {};
  int64_t nearest = resize.getNearestMode();
  if (nearest != 0 && nearest != 2)
    return {};

  auto inType = dyn_cast<RankedTensorType>(resize.getInput().getType());
  auto outType = dyn_cast<RankedTensorType>(value.getType());
  if (!inType || !outType || !inType.hasStaticShape() ||
      !outType.hasStaticShape() || inType.getRank() != outType.getRank())
    return {};
  // rocMLIR convolution is 2D or 3D.
  if (inType.getRank() != 4 && inType.getRank() != 5)
    return {};

  bool grew = false;
  SmallVector<int64_t> ratios;
  ratios.reserve(inType.getRank());
  for (auto [inDim, outDim] :
       llvm::zip(inType.getShape(), outType.getShape())) {
    if (inDim <= 0 || outDim < inDim || outDim % inDim != 0)
      return {};
    int64_t ratio = outDim / inDim;
    if (ratio > 1)
      grew = true;
    if (nearest == 0 && ratio > 2)
      return {};
    ratios.push_back(ratio);
  }
  if (!grew)
    return {};

  // The runtime samples with the attribute scale, not out/in. A near-integer
  // value such as 2.0000004 for an 8 -> 16 resize still grows by 2, but
  // floor(o / scale) disagrees with the repeat view's o / 2, so reject it.
  if (ArrayAttr scales = resize.getScalesAttr()) {
    if (static_cast<int64_t>(scales.size()) != inType.getRank())
      return {};
    for (auto [attr, ratio] : llvm::zip(scales.getValue(), ratios)) {
      double scale = cast<FloatAttr>(attr).getValueAsDouble();
      if (scale != static_cast<double>(ratio))
        return {};
    }
  }
  return resize.getInput();
}

// [N, C, H, W] -> [N, C, H, 1, W, 1] -> broadcast the new axes to the integer
// scale -> [N, C, H*scale, W*scale]. Axes with scale 1 are left unchanged.
static Value buildNearestRepeatView(PatternRewriter &rewriter, Location loc,
                                    Value context, Value input,
                                    Type resultType) {
  auto inType = cast<RankedTensorType>(input.getType());
  auto outType = cast<RankedTensorType>(resultType);
  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();

  SmallVector<int64_t> splitShape;
  SmallVector<int64_t> broadcastShape;
  SmallVector<ReassociationIndices> reassoc;
  int64_t axis = 0;
  for (int64_t i = 0, rank = inType.getRank(); i < rank; ++i) {
    int64_t scale = outShape[i] / inShape[i];
    ReassociationIndices group;
    group.push_back(axis++);
    splitShape.push_back(inShape[i]);
    broadcastShape.push_back(inShape[i]);
    if (scale > 1) {
      group.push_back(axis++);
      splitShape.push_back(1);
      broadcastShape.push_back(scale);
    }
    reassoc.push_back(std::move(group));
  }

  Type elemType = inType.getElementType();
  auto splitType = RankedTensorType::get(splitShape, elemType);
  SmallVector<OpFoldResult> splitFolds;
  for (int64_t dim : splitShape)
    splitFolds.push_back(rewriter.getIndexAttr(dim));
  Value split = tensor::ExpandShapeOp::create(rewriter, loc, splitType, input,
                                              reassoc, splitFolds);

  if (splitShape == broadcastShape)
    return tensor::CollapseShapeOp::create(rewriter, loc, outType, split,
                                           reassoc);

  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(broadcastShape.size())}, rewriter.getI64Type());
  Value shape = arith::ConstantOp::create(
      rewriter, loc, shapeType,
      DenseIntElementsAttr::get(shapeType, broadcastShape));
  Value init = tensor::EmptyOp::create(rewriter, loc, broadcastShape, elemType);
  Value expanded = ExpandOp::create(rewriter, loc, context, split, shape, init)
                       ->getResult(0);
  return tensor::CollapseShapeOp::create(rewriter, loc, outType, expanded,
                                         reassoc);
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

    // A nearest integer upsample in front of this convolution is a view of
    // the smaller tensor. Pass that tensor in and rebuild the view inside the
    // kernel; leaving hip.resize outside would still write the expanded copy.
    ResizeOp absorbedResize;
    if (auto conv = dyn_cast<ConvOp>(anchorOp.getOperation())) {
      Value input = conv.getInput();
      if (Value root = nearestRepeatInput(input, conv)) {
        SmallVector<Value> ordered(operands.begin(), operands.end());
        for (Value &operand : ordered)
          if (operand == input)
            operand = root;
        operands.clear();
        operands.insert_range(ordered);
        absorbedResize = input.getDefiningOp<ResizeOp>();
      }
    }

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

      if (absorbedResize) {
        Value viewed = buildNearestRepeatView(
            rewriter, absorbedResize.getLoc(), mapping.lookup(context),
            mapping.lookup(absorbedResize.getInput()),
            absorbedResize.getResult(0).getType());
        mapping.map(absorbedResize.getResult(0), viewed);
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

    Operation *replaced = endOp.getOperation();
    rewriter.replaceOp(endOp, rocMlirOp);
    // Ops cloned into the kernel still sit in main_graph. A pointwise
    // epilogue leaves the original convolution with no users, and that
    // convolution is the only reader of an absorbed resize. Drop trivially
    // dead intermediates first so the resize cleanup below sees use_empty
    // instead of erasing a live SSA producer.
    for (Operation *op : llvm::reverse(ops)) {
      if (op == replaced)
        continue;
      if (isOpTriviallyDead(op))
        rewriter.eraseOp(op);
    }
    if (absorbedResize && absorbedResize->use_empty()) {
      Value init = absorbedResize.getOutput();
      rewriter.eraseOp(absorbedResize);
      // Only a tensor.empty destination is safe to drop. A side-effecting
      // producer of the outs() buffer must stay even when its result is unused.
      if (auto empty = dyn_cast_or_null<tensor::EmptyOp>(init.getDefiningOp());
          empty && empty->use_empty())
        rewriter.eraseOp(empty);
    }
    return success();
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
