/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "GemmGemm.h"
#include "KernelOutline.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>

#include <array>
#include <optional>

namespace mlir::hip {

// The kernel's one matmul or gemm, if it has exactly one and no convolution.
static Operation *getSoleGemmAnchor(func::FuncOp kernel) {
  Operation *anchor = nullptr;
  for (Operation &op : kernel.getBody().front()) {
    if (isa<ConvOp, QuantConvOp, QuantMatmulOp>(op))
      return nullptr;
    if (isa<MatmulOp, GemmOp>(op)) {
      if (anchor)
        return nullptr;
      anchor = &op;
    }
  }
  return anchor;
}

static OpOperand &getGemmAOperand(Operation *anchor) {
  if (auto matmul = dyn_cast<MatmulOp>(anchor))
    return matmul.getAMutable();
  return cast<GemmOp>(anchor).getInputAMutable();
}

// (m, n, k) of `A[m, k] @ B[k, n]`, after the gemm's own transposes.
static std::optional<std::array<int64_t, 3>> getGemmDims(Operation *anchor) {
  Value a = getGemmAOperand(anchor).get();
  Value b;
  bool transA = false, transB = false;
  if (auto matmul = dyn_cast<MatmulOp>(anchor)) {
    b = matmul.getB();
    transA = matmul.getTransA() != 0;
    transB = matmul.getTransB() != 0;
  } else {
    auto gemm = cast<GemmOp>(anchor);
    b = gemm.getInputB();
    transA = gemm.getTransA() != 0;
    transB = gemm.getTransB() != 0;
  }
  auto aType = dyn_cast<RankedTensorType>(a.getType());
  auto bType = dyn_cast<RankedTensorType>(b.getType());
  if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 2)
    return std::nullopt;
  ArrayRef<int64_t> aShape = aType.getShape(), bShape = bType.getShape();
  int64_t aRows = aShape[aShape.size() - 2], aCols = aShape.back();
  int64_t bRows = bShape[bShape.size() - 2], bCols = bShape.back();
  return std::array<int64_t, 3>{transA ? aCols : aRows, transB ? bRows : bCols,
                                transA ? aRows : aCols};
}

// A gemm-gemm kernel without matrix acceleration for its element types falls
// back to a slow lowering, so both gemms must read types the target
// accelerates. `accelTypes` holds them as printed (f16, bf16, ...).
static bool isGemmGemmSupported(Operation *anchor,
                                ArrayRef<std::string> accelTypes) {
  Value b = isa<MatmulOp>(anchor) ? cast<MatmulOp>(anchor).getB()
                                  : cast<GemmOp>(anchor).getInputB();
  auto isAccelerated = [&](Value operand) {
    std::string name;
    llvm::raw_string_ostream os(name);
    getElementTypeOrSelf(operand.getType()).print(os);
    return llvm::is_contained(accelTypes, name);
  };
  return isAccelerated(getGemmAOperand(anchor).get()) && isAccelerated(b);
}

// The shape gate for gemm0 (m x k) @ (k x n) followed by gemm1
// (m x n) @ (n x g). Fusing keeps the
// m x n intermediate on chip, which pays only for a column-vector
// intermediate or a skinny-k MLP bottom over a large batch. Both gates cap g
// at 256, so rocMLIR's LDS estimate for its conservative default perf config,
// (32*32 + 32*32 + 32*pow2ceil(g)) elements, always fits in 64 KiB.
static bool passesGemmGemmHeuristic(std::array<int64_t, 3> first, int64_t g) {
  auto [m, n, k] = first;
  if (m <= 0 || n <= 0 || k <= 0 || g <= 1)
    return false;
  if (m <= 8 && n >= 512)
    return false;
  if (n == 1 && k <= 128 && m <= 1024 && g <= 64)
    return true;
  return m >= 98304 && n >= 512 && k <= 13 && g <= 256;
}

// What may lie between the two gemms: pointwise ops, and a reshape that only
// regroups the rows (batch and m) and keeps n. A transpose there makes the
// second gemm's reduction dimension disagree with the first gemm's n.
static bool isGemmGemmLink(Operation *op) {
  if (isFusablePointwise(op))
    return true;
  if (getFusableViewKind(op) != ViewKind::Reshape)
    return false;
  auto in = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return in && out && in.getRank() > 0 && out.getRank() > 0 &&
         in.getShape().back() == out.getShape().back();
}

// Whether only links lie between the anchor and the value the kernel
// returns, so that value is the gemm-elementwise result rocMLIR expects
// between the two gemms. Views of epilogue operands (a reshaped bias) do not
// read the anchor and may sit anywhere.
static bool endsInGemmPointwise(func::FuncOp kernel, Operation *anchor) {
  DenseSet<Value> derived(anchor->result_begin(), anchor->result_end());
  for (Operation &op : llvm::make_range(std::next(anchor->getIterator()),
                                        kernel.getBody().front().end())) {
    if (isa<func::ReturnOp>(op) ||
        llvm::none_of(op.getOperands(),
                      [&](Value v) { return derived.contains(v); }))
      continue;
    if (!isGemmGemmLink(&op))
      return false;
    derived.insert(op.result_begin(), op.result_end());
  }
  return true;
}

// Whether `arg` reaches the anchor's A operand through single-use links only
// (a prologue or row reshape the second kernel absorbed).
static bool feedsGemmA(BlockArgument arg, Operation *anchor) {
  Value value = arg;
  while (value.hasOneUse()) {
    OpOperand &use = *value.use_begin();
    Operation *user = use.getOwner();
    if (user == anchor)
      return use.getOperandNumber() ==
             getGemmAOperand(anchor).getOperandNumber();
    if (!isGemmGemmLink(user) || user->getNumResults() != 1)
      return false;
    value = user->getResult(0);
  }
  return false;
}

// Whether another input of `second` is computed from `first`'s result, so a
// merged kernel would read its own output. `between` holds the host ops that
// sit between the two kernels and are absorbed with them.
static bool closesCycle(RocMlirOp first, RocMlirOp second, unsigned index,
                        ArrayRef<Operation *> between) {
  SetVector<Operation *> absorbed;
  absorbed.insert(first);
  absorbed.insert(between.begin(), between.end());
  return llvm::any_of(llvm::enumerate(second.getInputs()), [&](auto other) {
    return other.index() != index &&
           dependsOnChain(other.value(), first, absorbed);
  });
}

// Replace `first`, the host ops `between` and `second` with one kernel: the
// first kernel's body, then `link` applied to its result, then the second
// kernel's body reading the linked value in place of its argument `index`.
// `link` builds inside the new kernel and receives the kernel's context.
static void mergeKernels(RocMlirOp first, RocMlirOp second, unsigned index,
                         ArrayRef<Operation *> between, int &counter,
                         function_ref<Value(OpBuilder &, Value, Value)> link) {
  auto module = second->getParentOfType<ModuleOp>();
  auto firstKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      first, first.getKernelAttr());
  auto secondKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      second, second.getKernelAttr());

  OpBuilder builder(second);
  SmallVector<Value> operands(first.getInputs());
  for (auto [otherIndex, other] : llvm::enumerate(second.getInputs()))
    if (otherIndex != index)
      operands.push_back(other);

  func::FuncOp merged;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    // The kernel signatures, not the dispatch operands: a kernel may view a
    // ui8 Where mask as i8.
    SmallVector<Type> argTypes(firstKernel.getArgumentTypes());
    for (auto [argIndex, type] :
         llvm::enumerate(secondKernel.getArgumentTypes()))
      if (argIndex != index)
        argTypes.push_back(type);
    SmallVector<Operation *> fusedOps;
    for (Operation &op : firstKernel.getBody().front())
      fusedOps.push_back(&op);
    fusedOps.append(between.begin(), between.end());
    for (Operation &op : secondKernel.getBody().front())
      fusedOps.push_back(&op);
    merged = func::FuncOp::create(
        builder, secondKernel.getLoc(),
        rocMlirKernelName(module, fusedOps, counter),
        builder.getFunctionType(argTypes,
                                secondKernel.getFunctionType().getResults()));
    merged->setAttrs(secondKernel->getDiscardableAttrDictionary());
    Block *body = merged.addEntryBlock();
    builder.setInsertionPointToStart(body);

    IRMapping mapping;
    unsigned numFirst = first.getInputs().size();
    for (auto [arg, newArg] :
         llvm::zip(firstKernel.getArguments(),
                   body->getArguments().take_front(numFirst)))
      mapping.map(arg, newArg);
    Value intermediate;
    Value ctx;
    for (Operation &op : firstKernel.getBody().front()) {
      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        intermediate = mapping.lookup(ret.getOperand(0));
        break;
      }
      Operation *clone = builder.clone(op, mapping);
      if (!ctx && clone->getNumResults() == 1 &&
          isa<ContextType>(clone->getResult(0).getType()))
        ctx = clone->getResult(0);
    }
    if (!ctx)
      ctx = ub::PoisonOp::create(builder, secondKernel.getLoc(),
                                 ContextType::get(builder.getContext()));
    Value linked = link(builder, intermediate, ctx);
    unsigned next = numFirst;
    for (BlockArgument arg : secondKernel.getArguments())
      mapping.map(arg, arg.getArgNumber() == index ? linked
                                                   : body->getArgument(next++));
    for (Operation &op : secondKernel.getBody().front())
      builder.clone(op, mapping);
  }

  auto fused =
      RocMlirOp::create(builder, second.getLoc(), second->getResultTypes(),
                        SymbolRefAttr::get(merged), second.getCtx(), operands,
                        SmallVector<Value>(second.getOutputs()));
  second->replaceAllUsesWith(fused);
  second->erase();
  SmallVector<Operation *> dead(llvm::reverse(between));
  dead.push_back(first);
  for (Operation *op : dead) {
    SmallVector<Value> operands(op->getOperands());
    op->erase();
    for (Value operand : operands)
      if (auto empty = operand.getDefiningOp<tensor::EmptyOp>();
          empty && empty->use_empty())
        empty->erase();
  }
  firstKernel->erase();
  secondKernel->erase();
}

// Gemm-pointwise-gemm: merge two outlined kernels when the first kernel's
// only result feeds the second kernel's gemm A operand, so rocMLIR lowers the
// pair to one rock.gemm_elementwise_gemm. Convolutions stay out.
//
// Before:
//   %h = hip.rocmlir @mlir_matmul_sigmoid_0 ins(%x, %w0) outs(%e0)
//   %y = hip.rocmlir @mlir_matmul_1 ins(%h, %w1) outs(%e1)
// After:
//   %y = hip.rocmlir @mlir_matmul_sigmoid_matmul_2 ins(%x, %w0, %w1) outs(%e1)
static bool fuseGemmGemmPair(RocMlirOp second, int &counter,
                             ArrayRef<std::string> accelTypes) {
  auto secondKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      second, second.getKernelAttr());
  if (!secondKernel)
    return false;
  Operation *secondAnchor = getSoleGemmAnchor(secondKernel);
  if (!secondAnchor || !isGemmGemmSupported(secondAnchor, accelTypes))
    return false;

  for (auto [index, input] : llvm::enumerate(second.getInputs())) {
    auto first = input.getDefiningOp<RocMlirOp>();
    if (!first || first->getNumResults() != 1 || !input.hasOneUse() ||
        first->getBlock() != second->getBlock())
      continue;
    auto firstKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        first, first.getKernelAttr());
    if (!firstKernel)
      continue;
    Operation *firstAnchor = getSoleGemmAnchor(firstKernel);
    if (!firstAnchor || !endsInGemmPointwise(firstKernel, firstAnchor) ||
        !isGemmGemmSupported(firstAnchor, accelTypes))
      continue;
    if (!feedsGemmA(secondKernel.getArgument(index), secondAnchor))
      continue;
    std::optional<std::array<int64_t, 3>> firstDims = getGemmDims(firstAnchor);
    std::optional<std::array<int64_t, 3>> secondDims =
        getGemmDims(secondAnchor);
    if (!firstDims || !secondDims ||
        !passesGemmGemmHeuristic(*firstDims, (*secondDims)[1]))
      continue;
    if (closesCycle(first, second, index, {}))
      continue;
    mergeKernels(
        first, second, index, {}, counter,
        [](OpBuilder &, Value intermediate, Value) { return intermediate; });
    return true;
  }
  return false;
}

// rocMLIR's attention kernel keeps a head dimension of each gemm in LDS.
// MIGraphX fuses attention up to the same limit.
static constexpr int64_t kMaxAttentionHeadDim = 256;

// The last-level cache of the largest target (gfx1151's 32 MiB MALL). A score
// matrix that fits stays in cache between the unfused kernels.
static constexpr int64_t kAttentionCacheBytes = int64_t{32} << 20;

// Both gemms read a floating-point type. With matrix acceleration fusing
// always pays; without it (f32) the fused kernel is slower per flop, so it
// pays only when the score matrix would otherwise spill out of the cache. A
// BERT-large f32 block (9 MiB of scores) measured faster unfused.
static bool isAttentionSupported(Operation *anchor,
                                 ArrayRef<std::string> accelTypes,
                                 int64_t scoreBytes) {
  Type elementType =
      getElementTypeOrSelf(getGemmAOperand(anchor).get().getType());
  if (!isa<FloatType>(elementType))
    return false;
  return isGemmGemmSupported(anchor, accelTypes) ||
         (elementType.isF32() && scoreBytes > kAttentionCacheBytes);
}

// Attention: merge a gemm kernel, the last-axis softmax that reads its only
// result, and the gemm kernel that reads the softmax as its A operand, so
// rocMLIR lowers the three to one rock.attention and the score matrix stays
// on chip. Scale, bias and cast epilogues of the first gemm and cast
// prologues of the second ride along as their kernels' pointwise ops.
//
// Before:
//   %s = hip.rocmlir @mlir_matmul_mul_0 ins(%q, %k) outs(%e0)
//   %p = hip.miopen.softmax(%ctx) ins(%s) outs(%e1)
//   %y = hip.rocmlir @mlir_matmul_1 ins(%p, %v) outs(%e2)
// After:
//   %y = hip.rocmlir @mlir_matmul_mul_miopen.softmax_matmul_2
//       ins(%q, %k, %v) outs(%e2)
static bool fuseAttentionTriple(RocMlirOp second, int &counter,
                                ArrayRef<std::string> accelTypes) {
  auto secondKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      second, second.getKernelAttr());
  if (!secondKernel)
    return false;
  Operation *secondAnchor = getSoleGemmAnchor(secondKernel);
  if (!secondAnchor)
    return false;

  for (auto [index, input] : llvm::enumerate(second.getInputs())) {
    auto softmax = input.getDefiningOp<MiopenSoftmaxOp>();
    if (!softmax || softmax->getNumResults() != 1 || !input.hasOneUse() ||
        softmax->getBlock() != second->getBlock())
      continue;
    Value scores = softmax.getInput();
    auto first = scores.getDefiningOp<RocMlirOp>();
    if (!first || first->getNumResults() != 1 || !scores.hasOneUse() ||
        first->getBlock() != second->getBlock())
      continue;
    auto scoresType = dyn_cast<RankedTensorType>(scores.getType());
    if (!scoresType || !scoresType.hasStaticShape() ||
        scoresType != input.getType())
      continue;
    auto firstKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        first, first.getKernelAttr());
    if (!firstKernel)
      continue;
    int64_t scoreBytes =
        scoresType.getNumElements() *
        llvm::divideCeil(scoresType.getElementTypeBitWidth(), 8);
    Operation *firstAnchor = getSoleGemmAnchor(firstKernel);
    if (!firstAnchor || !endsInGemmPointwise(firstKernel, firstAnchor) ||
        !isAttentionSupported(firstAnchor, accelTypes, scoreBytes) ||
        !isAttentionSupported(secondAnchor, accelTypes, scoreBytes))
      continue;
    if (!feedsGemmA(secondKernel.getArgument(index), secondAnchor))
      continue;
    std::optional<std::array<int64_t, 3>> firstDims = getGemmDims(firstAnchor);
    std::optional<std::array<int64_t, 3>> secondDims =
        getGemmDims(secondAnchor);
    // (m, n, k) of Q @ K^T and (m, g, n) of P @ V: k and g are head dims.
    if (!firstDims || !secondDims || (*firstDims)[2] > kMaxAttentionHeadDim ||
        (*secondDims)[1] > kMaxAttentionHeadDim)
      continue;
    if (closesCycle(first, second, index, {softmax}))
      continue;
    mergeKernels(first, second, index, {softmax}, counter,
                 [&](OpBuilder &builder, Value intermediate, Value ctx) {
                   Value init = tensor::EmptyOp::create(
                       builder, softmax.getLoc(), scoresType.getShape(),
                       scoresType.getElementType());
                   IRMapping mapping;
                   mapping.map(softmax.getCtx(), ctx);
                   mapping.map(scores, intermediate);
                   mapping.map(softmax.getOutput(), init);
                   return builder.clone(*softmax, mapping)->getResult(0);
                 });
    return true;
  }
  return false;
}

void fuseAttentionKernels(func::FuncOp funcOp, int &counter,
                          ArrayRef<std::string> accelTypes) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto op : llvm::make_early_inc_range(funcOp.getOps<RocMlirOp>()))
      if (fuseAttentionTriple(op, counter, accelTypes)) {
        changed = true;
        break;
      }
  }
}

void fuseGemmGemmKernels(func::FuncOp funcOp, int &counter,
                         ArrayRef<std::string> accelTypes) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto op : llvm::make_early_inc_range(funcOp.getOps<RocMlirOp>()))
      if (fuseGemmGemmPair(op, counter, accelTypes)) {
        changed = true;
        break;
      }
  }
}

} // namespace mlir::hip
