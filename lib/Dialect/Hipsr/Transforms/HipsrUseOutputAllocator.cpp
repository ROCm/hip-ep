/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

#define DEBUG_TYPE "hipsr-use-output-allocator"
namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRUSEOUTPUTALLOCATORPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Find a visible !hipsr.context for `op`, starting from its block and moving
// outward. At each nesting level, try in this order:
//   1. Block arguments of the current block (first !hipsr.context wins).
//   2. Block arguments of the region entry block, but only when the current
//      block is not the entry block.
//   3. Move to the parent operation and repeat steps 1-2.
// Stop with no result if the parent has IsIsolatedFromAbove (for example
// hipsr.pool_domain): ctx must be passed in through the region, not taken
// from outside.
static Value findVisibleContext(Operation *op) {
  Operation *anchor = op;
  while (Block *block = anchor->getBlock()) {
    for (BlockArgument arg : block->getArguments())
      if (isa<ContextType>(arg.getType()))
        return arg;

    Region *region = block->getParent();
    if (!region)
      return {};

    // Entry block arguments dominate all other blocks in the same region.
    if (!region->empty() && &region->front() != block)
      for (BlockArgument arg : region->front().getArguments())
        if (isa<ContextType>(arg.getType()))
          return arg;

    Operation *parentOp = region->getParentOp();
    if (!parentOp || parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return {};

    anchor = parentOp;
  }

  return {};
}

// return shapes is empty if no preserve_shape op is found.
static SmallVector<Value> findPreservedShapes(Value data) {
  SmallVector<Value> shapes;
  for (Operation *user : data.getUsers()) {
    auto preserveShape = dyn_cast<PreserveShapeOp>(user);
    if (!preserveShape || preserveShape.getData() != data)
      continue;
    if (!llvm::is_contained(shapes, preserveShape.getShape()))
      shapes.push_back(preserveShape.getShape());
  }
  return shapes;
}

// Trace backward from a graph output SSA value along its
// alias chain to find all underlying memref.alloc operations.
static SmallVector<memref::AllocOp>
findAliasedAllocs(BufferViewFlowAnalysis &aliasAnalysis, Value value) {
  SmallVector<memref::AllocOp> allocs;
  for (Value alias : aliasAnalysis.resolveReverse(value)) {
    auto allocOp = alias.getDefiningOp<memref::AllocOp>();
    if (allocOp && !llvm::is_contained(allocs, allocOp))
      allocs.push_back(allocOp);
  }
  return allocs;
}

struct PoolResult {
  PoolDomainOp poolDomain;
  unsigned resultIndex;
};
static SmallVector<PoolResult>
findAliasedPoolResults(BufferViewFlowAnalysis &aliasAnalysis, Value value) {
  SmallVector<PoolResult> poolResults;
  for (Value alias : aliasAnalysis.resolveReverse(value)) {
    auto result = dyn_cast<OpResult>(alias);
    if (!result)
      continue;
    auto poolDomain = dyn_cast<PoolDomainOp>(result.getOwner());
    if (!poolDomain)
      continue;

    unsigned resultIndex = result.getResultNumber();
    bool alreadyFound = llvm::any_of(poolResults, [&](PoolResult &found) {
      return found.poolDomain.getOperation() == poolDomain.getOperation() &&
             found.resultIndex == resultIndex;
    });
    if (!alreadyFound)
      poolResults.push_back({poolDomain, resultIndex});
  }
  return poolResults;
}

static SmallVector<Value> materializeDimensions(Value preservedShape,
                                                MemRefType type, Location loc,
                                                OpBuilder &builder) {
  SmallVector<Value> dimensions;
  dimensions.reserve(type.getRank());

  for (int64_t dimension : llvm::seq<int64_t>(0, type.getRank())) {
    Value size;
    if (type.isDynamicDim(dimension)) {
      assert(preservedShape && "dynamic dims require a preserved shape");
      size =
          shape::GetExtentOp::create(builder, loc, preservedShape, dimension);
    } else {
      size = arith::ConstantIndexOp::create(builder, loc,
                                            type.getDimSize(dimension));
    }

    dimensions.push_back(size);
  }
  return dimensions;
}

static SmallVector<Value> getDynamicDimensions(ArrayRef<Value> dimensions,
                                               MemRefType type) {
  SmallVector<Value> dynamicDimensions;
  for (int64_t dimension : llvm::seq<int64_t>(0, type.getRank()))
    if (type.isDynamicDim(dimension))
      dynamicDimensions.push_back(dimensions[dimension]);
  return dynamicDimensions;
}

static OpFoldResult multiplyIndexValues(OpFoldResult lhs, OpFoldResult rhs,
                                        Location loc, OpBuilder &builder) {
  std::optional<int64_t> lhsConstant = getConstantIntValue(lhs);
  std::optional<int64_t> rhsConstant = getConstantIntValue(rhs);
  if (lhsConstant && rhsConstant)
    return builder.getIndexAttr((*lhsConstant) * (*rhsConstant));
  if (lhsConstant && (*lhsConstant) == 1)
    return rhs;
  if (rhsConstant && (*rhsConstant) == 1)
    return lhs;

  Value lhsValue = getValueOrCreateConstantIndexOp(builder, loc, lhs);
  Value rhsValue = getValueOrCreateConstantIndexOp(builder, loc, rhs);
  return arith::MulIOp::create(builder, loc, lhsValue, rhsValue).getResult();
}

static Value createContiguousView(OpBuilder &builder, Location loc,
                                  Value source, MemRefType targetType,
                                  ArrayRef<Value> targetDimensions) {
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(targetType.getRank());
  for (int64_t dimension : llvm::seq<int64_t>(0, targetType.getRank())) {
    if (targetType.isDynamicDim(dimension))
      sizes.push_back(targetDimensions[dimension]);
    else
      sizes.push_back(builder.getIndexAttr(targetType.getDimSize(dimension)));
  }

  SmallVector<OpFoldResult> strides(targetType.getRank());
  OpFoldResult runningStride = builder.getIndexAttr(1);
  for (int64_t dimension = targetType.getRank(); dimension-- > 0;) {
    strides[dimension] = runningStride;
    if (dimension != 0)
      runningStride =
          multiplyIndexValues(runningStride, sizes[dimension], loc, builder);
  }

  return memref::ReinterpretCastOp::create(builder, loc, targetType, source,
                                           builder.getIndexAttr(0), sizes,
                                           strides);
}

struct OutputInfo {
  int64_t outIdx;
  memref::AllocOp allocOp;
  Value context;
  MemRefType externalType;
  Value externalShape;
  Value internalShape;
};

struct HipsrUseOutputAllocatorPass
    : impl::HipsrUseOutputAllocatorPassBase<HipsrUseOutputAllocatorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, hipsr::HipsrDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!funcOp.isPublic() || funcOp.empty())
      return;

    if (funcOp.getNumArguments() == 0 ||
        !isa<ContextType>(funcOp.getArgument(0).getType()))
      return;

    BufferViewFlowAnalysis aliasAnalysis(funcOp);
    DominanceInfo dominance(funcOp);

    SmallVector<func::ReturnOp> returnOps;
    funcOp.walk(
        [&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

    if (returnOps.size() != 1) {
      funcOp.emitError() << "expected exactly one func.return, got "
                         << returnOps.size();
      signalPassFailure();
      return;
    }
    func::ReturnOp returnOp = returnOps.front();

    if (returnOp.getNumOperands() != funcOp.getFunctionType().getNumResults()) {
      returnOp.emitError()
          << "return operand count does not match function result count";
      signalPassFailure();
      return;
    }

    SmallVector<OutputInfo> outputs;
    llvm::DenseMap<Operation *, int64_t> allocToOutputIndex;

    for (auto [outIdx, returnedValue] :
         llvm::enumerate(returnOp.getOperands())) {
      auto externalType = dyn_cast<MemRefType>(funcOp.getResultTypes()[outIdx]);

      if (!externalType)
        continue;

      // Find hipsr.pool_domain that produces this output.
      Value yieldedValue = returnedValue;
      SmallVector<PoolResult> poolResults =
          findAliasedPoolResults(aliasAnalysis, returnedValue);
      if (poolResults.size() > 1) {
        returnOp.emitError() << "graph output " << outIdx
                             << " aliases multiple pool_domain results";
        signalPassFailure();
        return;
      }

      // Find hipsr.pool_domain_yield - terminator of the pool_domain.
      PoolDomainOp poolDomain;
      if (!poolResults.empty()) {
        poolDomain = poolResults.front().poolDomain;
        if (poolDomain.getBody().empty() ||
            !isa<PoolDomainYieldOp>(
                poolDomain.getBody().front().getTerminator())) {
          poolDomain.emitError("expected a pool_domain_yield terminator");
          signalPassFailure();
          return;
        }
        // Get yield operand (SSA value being yielded)
        unsigned resultIndex = poolResults.front().resultIndex;
        auto yieldOp = cast<PoolDomainYieldOp>(
            poolDomain.getBody().front().getTerminator());
        if (resultIndex >= yieldOp.getNumOperands()) {
          poolDomain.emitError()
              << "result " << resultIndex
              << " has no corresponding pool_domain_yield operand";
          signalPassFailure();
          return;
        }
        yieldedValue = yieldOp.getOperand(resultIndex);
      }

      SmallVector<memref::AllocOp> aliasedAllocs =
          findAliasedAllocs(aliasAnalysis, yieldedValue);

      // pool_domain_yaid has exactly one aliased alloc
      if (aliasedAllocs.empty())
        continue;

      if (aliasedAllocs.size() != 1) {
        returnOp.emitError()
            << "graph output " << outIdx << " aliases " << aliasedAllocs.size()
            << " memref.alloc operations";
        signalPassFailure();
        return;
      }
      memref::AllocOp allocOp = aliasedAllocs.front();

      SmallVector<Value> externalShapes = findPreservedShapes(yieldedValue);
      SmallVector<Value> internalShapes =
          findPreservedShapes(allocOp.getResult());
      if (externalShapes.size() > 1) {
        returnOp.emitError()
            << "graph output " << outIdx
            << " has multiple distinct preserved external shapes";
        signalPassFailure();
        return;
      }
      if (internalShapes.size() > 1) {
        allocOp.emitError()
            << "has multiple distinct preserved internal shapes";
        signalPassFailure();
        return;
      }

      Value externalShape =
          externalShapes.empty() ? Value{} : externalShapes.front();
      Value internalShape =
          internalShapes.empty() ? Value{} : internalShapes.front();

      MemRefType internalType = allocOp.getType();

      if (externalType.getNumDynamicDims() != 0 && !externalShape) {
        returnOp.emitError()
            << "graph output " << outIdx
            << " has dynamic dims but no preserved external shape";
        signalPassFailure();
        return;
      }

      if (internalType.getNumDynamicDims() != 0 && !internalShape) {
        allocOp.emitError()
            << "has dynamic dims but no preserved internal shape";
        signalPassFailure();
        return;
      }

      Value context = findVisibleContext(allocOp);
      if (!context) {
        allocOp.emitError()
            << "cannot find a visible !hipsr.context for graph output "
               "allocation";
        signalPassFailure();
        return;
      }

      if (internalType.getElementType() != externalType.getElementType() ||
          internalType.getMemorySpace() != externalType.getMemorySpace()) {
        allocOp.emitError() << "cannot reinterpret graph output " << outIdx
                            << " as the original allocation type";
        signalPassFailure();
        return;
      }

      // Check graph output type is supported by alloc_output lowering.
      if (!externalType.getLayout().isIdentity()) {
        returnOp.emitError() << "graph output " << outIdx
                             << " must have an identity memref layout";
        signalPassFailure();
        return;
      }

      bool needsView =
          externalType != internalType || externalShape != internalShape;

      if (needsView && !internalType.getLayout().isIdentity()) {
        allocOp.emitError()
            << "cannot create a contiguous view with the original allocation "
               "layout";
        signalPassFailure();
        return;
      }

      // Check external and internal shapes are defined before the new
      // alloc_output insertion point.
      if ((externalShape && !dominance.dominates(externalShape, allocOp)) ||
          (internalShape && !dominance.dominates(internalShape, allocOp))) {
        allocOp.emitError()
            << "preserved output shapes do not dominate the allocation";
        signalPassFailure();
        return;
      }

      // Check if the same memref.alloc is mapped to multiple graph output
      // indices. This theoretically could cause errors, but usually won't
      // happen. eg: input -> Matmul(graph_output) -> Squeeze(graph_output)
      auto [seenIt, inserted] =
          allocToOutputIndex.try_emplace(allocOp.getOperation(), outIdx);
      if (!inserted) {
        allocOp.emitError() << "is shared by graph outputs " << seenIt->second
                            << " and " << outIdx;
        signalPassFailure();
        return;
      }

      outputs.push_back({static_cast<int64_t>(outIdx), allocOp, context,
                         externalType, externalShape, internalShape});
    }

    // Rewrite only after all structural checks and alias-analysis queries have
    // completed, because BufferViewFlowAnalysis caches the original IR.
    OpBuilder builder(funcOp.getContext());
    for (const OutputInfo &output : outputs) {
      memref::AllocOp allocOp = output.allocOp;
      MemRefType internalType = allocOp.getType();
      Location loc = allocOp.getLoc();
      builder.setInsertionPoint(allocOp);

      SmallVector<Value> externalDimensions = materializeDimensions(
          output.externalShape, output.externalType, loc, builder);
      SmallVector<Value> internalDimensions = materializeDimensions(
          output.internalShape, internalType, loc, builder);

      SmallVector<Value> externalDynamicSizes =
          getDynamicDimensions(externalDimensions, output.externalType);
      auto allocOutput = AllocOutputOp::create(
          builder, loc, output.externalType, output.context,
          externalDynamicSizes, builder.getI64IntegerAttr(output.outIdx));

      Value replacement = allocOutput.getResult();
      if (output.externalType != internalType ||
          output.externalShape != output.internalShape)
        replacement = createContiguousView(builder, loc, replacement,
                                           internalType, internalDimensions);

      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          dealloc.erase();

      allocOp.getResult().replaceAllUsesWith(replacement);
      allocOp.erase();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
