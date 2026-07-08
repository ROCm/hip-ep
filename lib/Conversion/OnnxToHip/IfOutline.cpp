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
//   1. Collect free variables (captures) used inside each branch.
//   2. Unbox the ONNX-style 0-D-tensor `cond` operand to `i1`.
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

/// Materialize an `i1` scalar from a 0-D BOOL-like tensor operand.
/// Same importer spellings as `onnx-loop-outline` (tensor<i1>, ui8, i8).
static FailureOr<Value> unboxCond(OpBuilder &builder, Location loc,
                                  Value condTensor) {
  auto t = dyn_cast<RankedTensorType>(condTensor.getType());
  if (!t || t.getRank() != 0)
    return failure();
  auto intTy = dyn_cast<IntegerType>(t.getElementType());
  if (!intTy)
    return failure();
  unsigned width = intTy.getWidth();
  if (width != 1 && width != 8)
    return failure();

  if (Operation *defOp = condTensor.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "onnx.Constant") {
      if (auto attr = defOp->getAttrOfType<DenseElementsAttr>("value")) {
        auto at = dyn_cast<RankedTensorType>(attr.getType());
        if (at && at.getRank() == 0 && at.getElementType() == intTy) {
          bool truthy = (*attr.getValues<APInt>().begin()).getZExtValue() != 0;
          return arith::ConstantIntOp::create(builder, loc, builder.getI1Type(),
                                              truthy ? 1 : 0);
        }
      }
    }
  }

  Value extracted =
      tensor::ExtractOp::create(builder, loc, condTensor, ValueRange{});
  if (width == 1)
    return extracted;

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

static LogicalResult outlineBranch(Region &region, Location loc,
                                   ModuleOp module, func::FuncOp parentFn,
                                   llvm::StringRef suffix, unsigned &counter,
                                   llvm::SetVector<Value> &capturesUnion,
                                   std::string &fnNameOut) {
  if (!region.hasOneBlock())
    return failure();
  Block &body = region.front();
  if (body.getNumArguments() != 0)
    return failure();

  Operation *yieldOp = body.getTerminator();
  if (yieldOp->getName().getStringRef() != "onnx.Yield")
    return failure();

  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, captures);
  for (Value c : captures)
    capturesUnion.insert(c);

  MLIRContext *ctx = region.getContext();
  Type ctxType = ContextType::get(ctx);
  SmallVector<Type> argTypes = {ctxType};
  for (Value c : captures)
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
  for (Value c : captures)
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
  std::string thenFnName;
  std::string elseFnName;
  auto parentFn = ifOp->getParentOfType<func::FuncOp>();

  if (failed(outlineBranch(thenRegion, loc, module, parentFn, "_if_then_n",
                           counter, capturesUnion, thenFnName)))
    return ifOp->emitOpError("failed to outline then_branch");
  if (failed(outlineBranch(elseRegion, loc, module, parentFn, "_if_else_n",
                           counter, capturesUnion, elseFnName)))
    return ifOp->emitOpError("failed to outline else_branch");

  if (!parentFn || parentFn.getNumArguments() == 0 ||
      !isa<ContextType>(parentFn.getArgument(0).getType()))
    return ifOp->emitOpError(
        "onnx.If is not inside a func.func with !hip.context as arg 0; "
        "ensure --hip-add-context-arg ran first");
  Value ctxVal = parentFn.getArgument(0);

  OpBuilder outerBuilder(ifOp);
  FailureOr<Value> condI1 = unboxCond(outerBuilder, loc, ifOp->getOperand(0));
  if (failed(condI1))
    return ifOp->emitOpError("could not unbox cond to i1 (expected 0-D "
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
