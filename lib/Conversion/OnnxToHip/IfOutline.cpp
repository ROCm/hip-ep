/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- IfOutline.cpp - Outline onnx.If branches into func.func ------------===//
//
// Converts each `onnx.If` op into a `hip.if` op plus two outlined
// `func.func` bodies (then / else).  Runs BEFORE `--convert-onnx-to-hip`
// so branch onnx.* ops get the same conversion treatment as ops in
// `main_graph`.
//
// Transformations performed per onnx.If:
//   1. Collect free variables (captures) from both branches. Each outlined
//      function takes that union, in one order, so the hip.if trampoline
//      can call either body with the same capture list.
//   2. Unbox the single-element `cond` operand to `i1`. ONNX requires one
//      element, not a rank-0 tensor: `Greater` of `tensor<1xf32>` and a
//      scalar broadcasts to `tensor<1xui8>`. A unit-rank condition is
//      collapsed to rank 0 and read back with `hip.readback_scalar`,
//      because that value is GPU-computed.
//   3. Materialize `tensor.empty` DPS inits for each If output.
//   4. Outline each branch region into a private func.func and replace
//      `onnx.Yield` with `func.return`.
//   5. Replace the onnx.If op with hip.if and erase it.
//
// Limitations of this initial version:
//   - Branches with non-empty region block arguments (explicit subgraph
//     inputs) are rejected — only capture-based outer values are wired.
//   - Nested `onnx.If` inside a branch is rejected.
//   - Fully dynamic output shapes (no static dim to anchor `tensor.empty`)
//     are rejected at outline time.
//
//===----------------------------------------------------------------------===//

#include "ReadbackScalar.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx-if-outline"

namespace mlir {
namespace hip {
namespace {

/// True when every dimension is static and the tensor holds one element.
/// ONNX `If` requires a single element; rank may be greater than 0.
static bool isSingleElement(RankedTensorType type) {
  if (type.getNumDynamicDims() != 0)
    return false;
  int64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0)
      return false;
    count *= dim;
  }
  return count == 1;
}

/// Collapse a static unit tensor (`tensor<1xui8>`, `tensor<1x1xui8>`, ...)
/// to a rank-0 tensor of the same element type.
static FailureOr<Value> collapseUnitToRank0(OpBuilder &builder, Location loc,
                                            Value cond, RankedTensorType type) {
  if (type.getRank() == 0)
    return cond;
  Type elemType = type.getElementType();
  Value current = cond;
  if (type.getRank() > 1) {
    SmallVector<ReassociationIndices, 1> reassoc(1);
    for (int64_t axis : llvm::seq<int64_t>(0, type.getRank()))
      reassoc.front().push_back(axis);
    current = tensor::CollapseShapeOp::create(
        builder, loc, RankedTensorType::get({1}, elemType), current, reassoc);
  }
  return tensor::CollapseShapeOp::create(
             builder, loc, RankedTensorType::get({}, elemType), current,
             ArrayRef<ReassociationIndices>{})
      .getResult();
}

/// Materialize an `i1` scalar from a single-element BOOL-like tensor.
/// Same importer spellings as `onnx-loop-outline` (i1, ui8, i8).
///
/// A rank-0 non-constant still uses `tensor.extract` (the historical path).
/// A unit tensor of higher rank is collapsed and read back with
/// `hip.readback_scalar`: the producer is a device kernel (`onnx.Greater`
/// of a length-1 vector broadcasts to `tensor<1xui8>`), and a bare extract
/// would load device memory without a stream sync.
static FailureOr<Value> unboxCond(OpBuilder &builder, Location loc, Value ctx,
                                  Value condTensor) {
  auto type = dyn_cast<RankedTensorType>(condTensor.getType());
  if (!type || !isSingleElement(type))
    return failure();
  auto intTy = dyn_cast<IntegerType>(type.getElementType());
  if (!intTy)
    return failure();
  unsigned width = intTy.getWidth();
  if (width != 1 && width != 8)
    return failure();

  if (DenseElementsAttr dense = getConstantDense(condTensor)) {
    if (dense.getNumElements() == 1) {
      bool truthy = (*dense.getValues<APInt>().begin()).getZExtValue() != 0;
      return arith::ConstantIntOp::create(builder, loc, builder.getI1Type(),
                                          truthy ? 1 : 0)
          .getResult();
    }
  }

  Value scalarTensor = condTensor;
  if (type.getRank() != 0) {
    FailureOr<Value> collapsed =
        collapseUnitToRank0(builder, loc, condTensor, type);
    if (failed(collapsed))
      return failure();
    scalarTensor = *collapsed;
  }

  // A rank-0 i1 is already the branch bit. Everything else is read back:
  // the unit-tensor case is a device value, and ui8/i8 need a host byte.
  Value extracted;
  if (type.getRank() == 0 && width == 1) {
    extracted =
        tensor::ExtractOp::create(builder, loc, scalarTensor, ValueRange{});
    return extracted;
  }
  extracted = readbackScalarToHost(builder, loc, ctx, scalarTensor);
  if (width == 1)
    return extracted;

  // arith.cmpi takes a signless integer. ui8 and i8 lower to the same LLVM
  // integer; HipToLLVM folds this cast once both types have been converted.
  Type signlessI8 = builder.getIntegerType(8);
  if (!intTy.isSignless()) {
    extracted = UnrealizedConversionCastOp::create(
                    builder, loc, TypeRange{signlessI8}, ValueRange{extracted})
                    .getResult(0);
  }
  Value zero = arith::ConstantIntOp::create(builder, loc, signlessI8, 0);
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                               extracted, zero)
      .getResult();
}

static FailureOr<SmallVector<Value>>
createIfOutputInits(OpBuilder &b, Location loc, TypeRange resultTypes) {
  SmallVector<Value> inits;
  inits.reserve(resultTypes.size());
  for (Type ty : resultTypes) {
    auto rankedTy = dyn_cast<RankedTensorType>(ty);
    if (!rankedTy)
      return failure();
    for (int64_t i : llvm::seq<int64_t>(0, rankedTy.getRank()))
      if (rankedTy.isDynamicDim(i))
        return failure();
    Value empty = tensor::EmptyOp::create(
        b, loc, rankedTy.getShape(), rankedTy.getElementType(), ValueRange{});
    inits.push_back(empty);
  }
  return inits;
}

/// Values defined outside `region` and used inside it, in first-use order.
static LogicalResult collectBranchCaptures(Region &region,
                                           llvm::SetVector<Value> &out) {
  if (!region.hasOneBlock())
    return failure();
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, captures);
  for (Value c : captures)
    out.insert(c);
  return success();
}

static LogicalResult outlineBranch(Region &region, Location loc,
                                   ModuleOp module, func::FuncOp parentFn,
                                   llvm::StringRef suffix, unsigned &counter,
                                   llvm::ArrayRef<Value> signatureCaptures,
                                   std::string &fnNameOut) {
  if (!region.hasOneBlock())
    return failure();
  Block &body = region.front();
  if (body.getNumArguments() != 0)
    return failure();

  Operation *yieldOp = body.getTerminator();
  if (yieldOp->getName().getStringRef() != "onnx.Yield")
    return failure();

  // Both branches take the union, in the same order hip.if passes it.
  // A value used only on the other side is an unused argument here. The
  // HipToLLVM trampoline calls both bodies with that one capture list plus
  // the out-params, so a per-branch signature fails legalization.
  MLIRContext *ctx = region.getContext();
  Type ctxType = ContextType::get(ctx);
  SmallVector<Type> argTypes = {ctxType};
  for (Value c : signatureCaptures)
    argTypes.push_back(c.getType());

  SmallVector<Type> resultTypes;
  resultTypes.reserve(yieldOp->getNumOperands());
  for (Value v : yieldOp->getOperands())
    resultTypes.push_back(v.getType());

  auto fnType = FunctionType::get(ctx, argTypes, resultTypes);

  auto buildName = [&](unsigned n) -> std::string {
    if (parentFn)
      return (parentFn.getName() + suffix + Twine(n)).str();
    return (suffix.drop_front() + Twine(n)).str();
  };
  std::string fnName;
  do {
    fnName = buildName(counter++);
  } while (module.lookupSymbol(fnName));

  OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto newFn = func::FuncOp::create(modBuilder, loc, fnName, fnType);
  newFn.setPrivate();

  Block *entry = newFn.addEntryBlock();
  IRMapping mapping;
  unsigned argIdx = 1;
  for (Value c : signatureCaptures)
    mapping.map(c, entry->getArgument(argIdx++));

  OpBuilder bodyBuilder(entry, entry->end());
  for (Operation &op : body) {
    if (&op == yieldOp)
      continue;
    bodyBuilder.clone(op, mapping);
  }

  SmallVector<Value> returnVals;
  returnVals.reserve(yieldOp->getNumOperands());
  for (Value v : yieldOp->getOperands())
    returnVals.push_back(mapping.lookupOrDefault(v));
  func::ReturnOp::create(bodyBuilder, loc, returnVals);

  fnNameOut = fnName;
  return success();
}

struct OnnxIfOutlinePass
    : public PassWrapper<OnnxIfOutlinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnnxIfOutlinePass)

  StringRef getArgument() const override { return "onnx-if-outline"; }

  StringRef getDescription() const override {
    return "Outline each onnx.If branch into separate func.func ops and "
           "replace the If with hip.if";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;
  LogicalResult outlineIf(Operation *ifOp, unsigned &counter);
};

LogicalResult OnnxIfOutlinePass::outlineIf(Operation *ifOp, unsigned &counter) {
  Location loc = ifOp->getLoc();
  ModuleOp module = ifOp->getParentOfType<ModuleOp>();

  if (ifOp->getNumOperands() != 1)
    return ifOp->emitOpError("expected exactly one cond operand");
  if (ifOp->getNumRegions() != 2)
    return ifOp->emitOpError("onnx.If with != 2 regions not supported");

  Region &thenRegion = ifOp->getRegion(0);
  Region &elseRegion = ifOp->getRegion(1);

  if (thenRegion.front().getNumArguments() != 0 ||
      elseRegion.front().getNumArguments() != 0)
    return ifOp->emitOpError(
        "onnx.If branches with subgraph block arguments are not yet supported");

  Operation *thenYield = thenRegion.front().getTerminator();
  Operation *elseYield = elseRegion.front().getTerminator();
  if (thenYield->getNumOperands() != elseYield->getNumOperands())
    return ifOp->emitOpError("then/else yield operand count mismatch");

  llvm::SetVector<Value> capturesUnion;
  if (failed(collectBranchCaptures(thenRegion, capturesUnion)) ||
      failed(collectBranchCaptures(elseRegion, capturesUnion)))
    return ifOp->emitOpError("failed to collect branch captures");

  std::string thenFnName;
  std::string elseFnName;
  auto parentFn = ifOp->getParentOfType<func::FuncOp>();
  llvm::ArrayRef<Value> signatureCaptures = capturesUnion.getArrayRef();

  if (failed(outlineBranch(thenRegion, loc, module, parentFn, "_if_then_n",
                           counter, signatureCaptures, thenFnName)))
    return ifOp->emitOpError("failed to outline then_branch");
  if (failed(outlineBranch(elseRegion, loc, module, parentFn, "_if_else_n",
                           counter, signatureCaptures, elseFnName)))
    return ifOp->emitOpError("failed to outline else_branch");

  if (!parentFn || parentFn.getNumArguments() == 0 ||
      !isa<ContextType>(parentFn.getArgument(0).getType()))
    return ifOp->emitOpError(
        "onnx.If is not inside a func.func with !hip.context as arg 0; "
        "ensure --hip-add-context-arg ran first");
  Value ctxVal = parentFn.getArgument(0);

  OpBuilder outerBuilder(ifOp);
  FailureOr<Value> condI1 =
      unboxCond(outerBuilder, loc, ctxVal, ifOp->getOperand(0));
  if (failed(condI1))
    return ifOp->emitOpError(
               "could not unbox cond to i1 (expected a single-element "
               "tensor<i1>, tensor<i8>, or tensor<ui8>, got ")
           << ifOp->getOperand(0).getType() << ")";

  FailureOr<SmallVector<Value>> oInits =
      createIfOutputInits(outerBuilder, loc, ifOp->getResultTypes());
  if (failed(oInits))
    return ifOp->emitOpError(
        "failed to build output inits (fully-static output shapes required)");

  SmallVector<Value> captureVals(capturesUnion.begin(), capturesUnion.end());
  unsigned numOutputs = ifOp->getNumResults();

  auto hipIfOp = IfOp::create(
      outerBuilder, loc, ctxVal, *condI1, *oInits, captureVals,
      FlatSymbolRefAttr::get(ifOp->getContext(), thenFnName),
      FlatSymbolRefAttr::get(ifOp->getContext(), elseFnName),
      outerBuilder.getI32IntegerAttr(static_cast<int32_t>(numOutputs)));

  ifOp->replaceAllUsesWith(hipIfOp.getResults());
  ifOp->erase();
  return success();
}

void OnnxIfOutlinePass::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult nestedFound = module.walk([&](Operation *outerIf) {
    if (outerIf->getName().getStringRef() != "onnx.If")
      return WalkResult::advance();
    if (outerIf->getNumRegions() != 2)
      return WalkResult::advance();
    Operation *innerIf = nullptr;
    auto scanRegion = [&](Region &r) {
      r.walk([&](Operation *inner) {
        if (inner->getName().getStringRef() == "onnx.If") {
          innerIf = inner;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    };
    scanRegion(outerIf->getRegion(0));
    if (!innerIf)
      scanRegion(outerIf->getRegion(1));
    if (innerIf) {
      InFlightDiagnostic diag = outerIf->emitOpError(
          "nested onnx.If is not supported by the MorphiZen EP");
      diag.attachNote(innerIf->getLoc()) << "inner onnx.If is here";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (nestedFound.wasInterrupted())
    return signalPassFailure();

  unsigned counter = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.If")
        worklist.push_back(op);
    });
    for (Operation *op : worklist) {
      if (failed(outlineIf(op, counter)))
        return signalPassFailure();
      changed = true;
    }
  }
}

} // namespace

std::unique_ptr<Pass> createOnnxIfOutlinePass() {
  return std::make_unique<OnnxIfOutlinePass>();
}

} // namespace hip
} // namespace mlir
