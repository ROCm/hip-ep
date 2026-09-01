/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeInitTensorsPass.cpp - Materialize placeholder inits -----===//
//
// Replaces every hipsr.placeholder in a pool domain with a shape computation
// and a tensor.empty. Runs last in --hipsr-pipeline, so domains are cut and
// shape regions are filled.
//
// Before:
//   %init = hipsr.placeholder(%ctx) ins(%a) : tensor<?x4xf32> shape_region {
//   ^bb0(%a_shape: !shape.shape):
//     hipsr.shape_yield %a_shape : !shape.shape
//   }
//   %0 = hipsr.cast(%ctx) ins(%a) outs(%init) : tensor<?x4xf32>
//
// After:
//   %s = scf.execute_region -> !shape.shape {
//     %a_shape = shape.shape_of %a : tensor<?x4xf16> -> !shape.shape
//     scf.yield %a_shape : !shape.shape
//   }
//   %d0 = shape.size_to_index (shape.get_extent %s, 0)
//   %empty = tensor.empty(%d0) : tensor<?x4xf32>
//   %0 = hipsr.cast(%ctx) ins(%a) outs(%empty) : tensor<?x4xf32>
//   hipsr.preserve_shape %s, %0 : tensor<?x4xf32>
//
// The domain is rebuilt in a fresh block in five steps: constants, shape
// computations, allocations, every other op, then one shape link per
// allocation. hipsr-pool-alloc replaces the allocations in a domain with views
// of one pool it emits after the last of them, so every allocation has to come
// before the first op that reads one. A link names the result of the op that
// fills the buffer, so it can only come after the data ops.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_MATERIALIZEINITTENSORSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// What each shape region argument becomes, in the placeholder's operand order.
//
// Barrier, ctx then the inputs, read as data:
//   a. ctx                -> the new ctx argument
//   b. arith.constant     -> the value itself
//   c. hipsr.constant     -> the value itself
//   d. block argument     -> the new block argument
//   e. placeholder result -> rejected by verifyMaterializable
//
// Normal, one !shape.shape per input:
//   a. arith.constant     -> shape.shape_of it
//   b. hipsr.constant     -> shape.shape_of it
//   c. block argument     -> shape.shape_of it
//   d. placeholder result -> the shape its region yielded
SmallVector<Value> getArgumentReplacements(PlaceholderOp placeholder,
                                           const IRMapping &shapes,
                                           const IRMapping &cloned,
                                           OpBuilder &builder) {
  if (placeholder.getPlaceholderType() == PlaceholderType::Barrier) {
    return llvm::map_to_vector(placeholder->getOperands(), [&](Value operand) {
      return cloned.lookup(operand);
    });
  }

  Location loc = placeholder.getLoc();
  auto shapeType = shape::ShapeType::get(builder.getContext());
  return llvm::map_to_vector(
      placeholder.getInputs(), [&](Value input) -> Value {
        if (Value recorded = shapes.lookupOrNull(input)) {
          return recorded;
        }
        return shape::ShapeOfOp::create(builder, loc, shapeType,
                                        cloned.lookup(input));
      });
}

// One placeholder's shape region, rebuilt as an scf.execute_region.
ResultRange materializeShapeRegion(PlaceholderOp placeholder,
                                   const IRMapping &shapes, IRMapping &cloned,
                                   RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<Type> shapeTypes(placeholder.getNumResults(),
                               shape::ShapeType::get(rewriter.getContext()));
  auto executeRegion =
      scf::ExecuteRegionOp::create(rewriter, placeholder.getLoc(), shapeTypes);

  // The block comes first, so a replacement that builds IR lands in the body.
  Block &shapeBody = placeholder.getShapeRegion().front();
  rewriter.createBlock(&executeRegion.getRegion());
  SmallVector<Value> replacements =
      getArgumentReplacements(placeholder, shapes, cloned, rewriter);
  assert(shapeBody.getNumArguments() == replacements.size() &&
         "expected one replacement per shape region argument");
  cloned.map(shapeBody.getArguments(), replacements);
  llvm::for_each(shapeBody.without_terminator(),
                 [&](Operation &op) { rewriter.clone(op, cloned); });

  // HasParent binds hipsr.shape_yield to placeholder, so scf.yield replaces it.
  auto shapeYield = cast<ShapeYieldOp>(shapeBody.getTerminator());
  scf::YieldOp::create(
      rewriter, shapeYield.getLoc(),
      llvm::map_to_vector(shapeYield.getShapes(),
                          [&](Value shape) { return cloned.lookup(shape); }));

  return executeRegion.getResults();
}

// Only dynamic dimensions come from the shape; a static extent is in the type.
Value createInitTensor(Value result, Value resultShape, OpBuilder &builder) {
  Location loc = result.getLoc();
  auto tensorType = cast<RankedTensorType>(result.getType());
  auto isDynamic = [tensorType](int64_t dimension) {
    return tensorType.isDynamicDim(dimension);
  };
  auto readExtent = [&](int64_t dimension) -> Value {
    Value extent =
        shape::GetExtentOp::create(builder, loc, resultShape, dimension);
    return shape::SizeToIndexOp::create(builder, loc, extent);
  };

  SmallVector<Value> dynamicSizes = llvm::map_to_vector(
      llvm::make_filter_range(llvm::seq<int64_t>(tensorType.getRank()),
                              isDynamic),
      readExtent);
  return tensor::EmptyOp::create(builder, loc, tensorType, dynamicSizes);
}

LogicalResult verifyMaterializable(ArrayRef<PlaceholderOp> placeholders) {
  for (PlaceholderOp placeholder : placeholders) {
    if (placeholder.getShapeRegion().empty()) {
      return placeholder.emitOpError(
          "shape region must be populated by -hipsr-populate-shape-region");
    }
    if (placeholder.getPlaceholderType() == PlaceholderType::Barrier &&
        llvm::any_of(placeholder.getInputs(), [](Value input) {
          return isa_and_nonnull<PlaceholderOp>(input.getDefiningOp());
        })) {
      return placeholder.emitOpError(
          "barrier input must be allocated outside this pool domain");
    }
  }
  return success();
}

// Taking no operands, a constant can lead the block and dominate its readers.
void cloneConstants(Block &oldBlock, IRMapping &cloned,
                    RewriterBase &rewriter) {
  for (Operation &op : oldBlock) {
    if (matchPattern(&op, m_Constant())) {
      rewriter.clone(op, cloned);
    }
  }
}

// Block order is SSA order, so a shape from an earlier placeholder is mapped.
IRMapping createShapeComputations(ArrayRef<PlaceholderOp> placeholders,
                                  IRMapping &cloned, RewriterBase &rewriter) {
  IRMapping shapes;
  for (PlaceholderOp placeholder : placeholders) {
    shapes.map(placeholder.getResults(),
               materializeShapeRegion(placeholder, shapes, cloned, rewriter));
  }
  return shapes;
}

// Each result maps to its tensor.empty, so ops cloned later use that buffer.
void createAllocations(ArrayRef<PlaceholderOp> placeholders,
                       const IRMapping &shapes, IRMapping &cloned,
                       OpBuilder &builder) {
  for (PlaceholderOp placeholder : placeholders) {
    for (OpResult result : placeholder.getResults()) {
      cloned.map(result,
                 createInitTensor(result, shapes.lookup(result), builder));
    }
  }
}

// Whatever an earlier step did not emit, still in its original order.
void cloneRemainingOps(Block &oldBlock, IRMapping &cloned,
                       RewriterBase &rewriter) {
  for (Operation &op : oldBlock.without_terminator()) {
    if (!isa<PlaceholderOp>(op) && !cloned.contains(&op)) {
      rewriter.clone(op, cloned);
    }
  }
}

// Linking the allocation would miss a hipsr.compute result that is only a view.
Value getTiedConsumerResult(OpResult init) {
  for (OpOperand &use : init.getUses()) {
    if (OpResult held = getResultForDestination(use)) {
      return held;
    }
  }
  return init;
}

// -hip-use-output-allocator reads these links to size a graph output.
void preserveShapes(ArrayRef<PlaceholderOp> placeholders,
                    const IRMapping &shapes, const IRMapping &cloned,
                    OpBuilder &builder) {
  for (PlaceholderOp placeholder : placeholders) {
    for (OpResult result : placeholder.getResults()) {
      PreserveShapeOp::create(builder, result.getLoc(), shapes.lookup(result),
                              cloned.lookup(getTiedConsumerResult(result)));
    }
  }
}

// Rebuilds one domain body with every placeholder materialized.
LogicalResult materializePoolDomain(Region &body, RewriterBase &rewriter) {
  Block &oldBlock = body.front();
  SmallVector<PlaceholderOp> placeholders =
      llvm::to_vector(oldBlock.getOps<PlaceholderOp>());
  if (placeholders.empty()) {
    return success();
  }
  if (failed(verifyMaterializable(placeholders))) {
    return failure();
  }

  // Every step appends to the new block, so emission order is final order.
  Block *newBlock = rewriter.createBlock(
      &body, body.end(), oldBlock.getArgumentTypes(),
      llvm::map_to_vector(oldBlock.getArguments(), [](BlockArgument argument) {
        return argument.getLoc();
      }));

  // Maps every old value to what stands in for it in the new block.
  IRMapping cloned;
  cloned.map(oldBlock.getArguments(), newBlock->getArguments());

  cloneConstants(oldBlock, cloned, rewriter);
  IRMapping shapes = createShapeComputations(placeholders, cloned, rewriter);
  createAllocations(placeholders, shapes, cloned, rewriter);
  cloneRemainingOps(oldBlock, cloned, rewriter);
  preserveShapes(placeholders, shapes, cloned, rewriter);
  rewriter.clone(*oldBlock.getTerminator(), cloned);

  rewriter.eraseBlock(&oldBlock);
  return success();
}

struct MaterializeInitTensorsPass
    : impl::MaterializeInitTensorsPassBase<MaterializeInitTensorsPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    WalkResult walkResult =
        getOperation().walk([&rewriter](PoolDomainOp poolDomain) {
          if (failed(materializePoolDomain(poolDomain.getBody(), rewriter))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
