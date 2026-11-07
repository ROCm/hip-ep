/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "GemmGemm.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
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

// Gemm-pointwise-gemm: merge two outlined kernels when the first kernel's
// only result feeds the second kernel's gemm A operand, so rocMLIR lowers the
// pair to one rock.gemm_elementwise_gemm. Convolutions stay out.
//
// Before:
//   %h = hip.rocmlir @rocMlir0 ins(%x, %w0) outs(%e0)   // matmul -> sigmoid
//   %y = hip.rocmlir @rocMlir1 ins(%h, %w1) outs(%e1)   // matmul
// After:
//   %y = hip.rocmlir @rocMlir2 ins(%x, %w0, %w1) outs(%e1)
//   // @rocMlir2: matmul -> sigmoid -> matmul
static bool fuseGemmGemmPair(RocMlirOp second, int &counter,
                             ArrayRef<std::string> accelTypes) {
  auto module = second->getParentOfType<ModuleOp>();
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
    // Another input computed from the first kernel's result would make the
    // merged kernel read its own output.
    SetVector<Operation *> firstOnly;
    firstOnly.insert(first);
    bool closesCycle =
        llvm::any_of(llvm::enumerate(second.getInputs()), [&](auto other) {
          return other.index() != index &&
                 dependsOnChain(other.value(), first, firstOnly);
        });
    if (closesCycle)
      continue;

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
      merged = func::FuncOp::create(
          builder, secondKernel.getLoc(), "rocMlir" + std::to_string(counter++),
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
      for (Operation &op : firstKernel.getBody().front()) {
        if (auto ret = dyn_cast<func::ReturnOp>(op)) {
          intermediate = mapping.lookup(ret.getOperand(0));
          break;
        }
        builder.clone(op, mapping);
      }
      unsigned next = numFirst;
      for (BlockArgument arg : secondKernel.getArguments())
        mapping.map(arg, arg.getArgNumber() == index
                             ? intermediate
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
    first->erase();
    firstKernel->erase();
    secondKernel->erase();
    return true;
  }
  return false;
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
