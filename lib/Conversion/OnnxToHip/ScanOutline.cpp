/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ScanOutline.cpp - Outline onnx.Scan bodies into func.func ----------===//

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace hip {
namespace {

static FailureOr<SmallVector<int64_t>> readIntArrayAttr(Operation *op,
                                                        StringRef name,
                                                        unsigned expected,
                                                        int64_t defaultValue) {
  SmallVector<int64_t> values(expected, defaultValue);
  Attribute attr = op->getAttr(name);
  if (!attr)
    return values;
  if (auto dense = dyn_cast<DenseI64ArrayAttr>(attr)) {
    if (dense.size() != expected)
      return failure();
    values.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return values;
  }
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    if (array.size() != expected)
      return failure();
    for (auto [i, value] : llvm::enumerate(array)) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return failure();
      values[i] = integer.getSInt();
    }
    return values;
  }
  return failure();
}

static FailureOr<Value> makeFirstSlice(OpBuilder &builder, Location loc,
                                       Value input, Type bodyArgType,
                                       int64_t axis) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto sliceType = dyn_cast<RankedTensorType>(bodyArgType);
  if (!inputType || !sliceType ||
      inputType.getRank() != sliceType.getRank() + 1)
    return failure();
  if (axis < 0)
    axis += inputType.getRank();
  if (axis < 0 || axis >= inputType.getRank())
    return failure();

  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    offsets.push_back(builder.getIndexAttr(0));
    strides.push_back(builder.getIndexAttr(1));
    if (dim == axis) {
      sizes.push_back(builder.getIndexAttr(1));
    } else if (inputType.isDynamicDim(dim)) {
      sizes.push_back(
          tensor::DimOp::create(builder, loc, input, dim).getResult());
    } else {
      sizes.push_back(builder.getIndexAttr(inputType.getDimSize(dim)));
    }
  }
  return tensor::ExtractSliceOp::create(builder, loc, sliceType, input, offsets,
                                        sizes, strides)
      .getResult();
}

struct OnnxScanOutlinePass
    : public PassWrapper<OnnxScanOutlinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnnxScanOutlinePass)

  StringRef getArgument() const override { return "onnx-scan-outline"; }
  StringRef getDescription() const override {
    return "Outline onnx.Scan bodies and replace them with hip.scan";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<HipDialect, arith::ArithDialect,
                bufferization::BufferizationDialect, cf::ControlFlowDialect,
                func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  LogicalResult outlineScan(Operation *scanOp, unsigned &counter);
  void runOnOperation() override;
};

LogicalResult OnnxScanOutlinePass::outlineScan(Operation *scanOp,
                                               unsigned &counter) {
  Location loc = scanOp->getLoc();
  ModuleOp module = scanOp->getParentOfType<ModuleOp>();
  MLIRContext *ctx = scanOp->getContext();

  if (scanOp->getNumRegions() != 1)
    return scanOp->emitOpError(
        "expected one body region (the importer emitted a region-less Scan)");
  Region &bodyRegion = scanOp->getRegion(0);
  if (!bodyRegion.hasOneBlock())
    return scanOp->emitOpError("Scan body must contain exactly one block");
  Block &bodyBlock = bodyRegion.front();
  Operation *yieldOp = bodyBlock.getTerminator();
  if (!yieldOp || yieldOp->getName().getStringRef() != "onnx.Yield")
    return scanOp->emitOpError("Scan body must end in onnx.Yield");

  auto numScanInputsAttr =
      scanOp->getAttrOfType<IntegerAttr>("num_scan_inputs");
  if (!numScanInputsAttr || numScanInputsAttr.getSInt() <= 0)
    return scanOp->emitOpError("requires a positive num_scan_inputs");
  unsigned numScanInputs = numScanInputsAttr.getSInt();
  if (numScanInputs > scanOp->getNumOperands())
    return scanOp->emitOpError("num_scan_inputs exceeds operand count");
  unsigned numState = scanOp->getNumOperands() - numScanInputs;
  if (bodyBlock.getNumArguments() != numState + numScanInputs)
    return scanOp->emitOpError("body argument count does not match operands");
  if (yieldOp->getNumOperands() < numState)
    return scanOp->emitOpError("body yields fewer values than state variables");
  unsigned numScanOutputs = yieldOp->getNumOperands() - numState;
  if (scanOp->getNumResults() != numState + numScanOutputs)
    return scanOp->emitOpError("result count does not match body yields");

  FailureOr<SmallVector<int64_t>> inputAxes = readIntArrayAttr(
      scanOp, "scan_input_axes", numScanInputs, /*defaultValue=*/0);
  FailureOr<SmallVector<int64_t>> inputDirections = readIntArrayAttr(
      scanOp, "scan_input_directions", numScanInputs, /*defaultValue=*/0);
  FailureOr<SmallVector<int64_t>> outputAxes = readIntArrayAttr(
      scanOp, "scan_output_axes", numScanOutputs, /*defaultValue=*/0);
  FailureOr<SmallVector<int64_t>> outputDirections = readIntArrayAttr(
      scanOp, "scan_output_directions", numScanOutputs, /*defaultValue=*/0);
  if (failed(inputAxes) || failed(inputDirections) || failed(outputAxes) ||
      failed(outputDirections))
    return scanOp->emitOpError("malformed Scan axis/direction attributes");
  if (llvm::any_of(*inputAxes, [](int64_t v) { return v != 0; }) ||
      llvm::any_of(*outputAxes, [](int64_t v) { return v != 0; }) ||
      llvm::any_of(*inputDirections, [](int64_t v) { return v != 0; }) ||
      llvm::any_of(*outputDirections, [](int64_t v) { return v != 0; }))
    return scanOp->emitOpError(
        "only forward Scan with input/output axis 0 is currently supported");

  ValueRange stateInitials = scanOp->getOperands().take_front(numState);
  ValueRange scanInputs = scanOp->getOperands().drop_front(numState);
  auto replaceResult = [&](OpResult oldResult,
                           Value newResult) -> LogicalResult {
    if (isa<NoneType>(oldResult.getType())) {
      if (!oldResult.use_empty())
        return scanOp->emitOpError("none-typed result unexpectedly has users");
      return success();
    }
    if (!newResult || oldResult.getType() != newResult.getType())
      return scanOp->emitOpError("result type changed while outlining Scan");
    oldResult.replaceAllUsesWith(newResult);
    return success();
  };
  auto firstInputType =
      dyn_cast<RankedTensorType>(scanInputs.front().getType());
  if (!firstInputType || firstInputType.getRank() == 0)
    return scanOp->emitOpError("scan inputs must be ranked non-scalars");

  // Avoid forming an invalid one-element slice when the trip count is known
  // to be zero. Explicit alloc_tensor ops keep same-typed empty graph outputs
  // distinct through CSE so each output allocator callback is still invoked.
  if (!firstInputType.isDynamicDim(0) && firstInputType.getDimSize(0) == 0) {
    OpBuilder builder(scanOp);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    for (unsigned i = 1; i < numScanInputs; ++i) {
      auto inputType = dyn_cast<RankedTensorType>(scanInputs[i].getType());
      if (!inputType || inputType.getRank() == 0)
        return scanOp->emitOpError("scan inputs must be ranked non-scalars");
      if (!inputType.isDynamicDim(0) && inputType.getDimSize(0) != 0)
        return scanOp->emitOpError(
            "statically known Scan input sequence lengths must match");
      if (inputType.isDynamicDim(0)) {
        Value length = tensor::DimOp::create(builder, loc, scanInputs[i], 0);
        Value isZero = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, length, zero);
        cf::AssertOp::create(builder, loc, isZero,
                             "Scan input sequence lengths must match");
      }
    }
    SmallVector<Value> zeroResults(stateInitials.begin(), stateInitials.end());
    for (unsigned i = 0; i < numScanOutputs; ++i) {
      OpResult oldResult = scanOp->getResult(numState + i);
      if (isa<NoneType>(oldResult.getType())) {
        zeroResults.push_back(Value());
        continue;
      }
      auto resultType = dyn_cast<RankedTensorType>(oldResult.getType());
      if (!resultType)
        return scanOp->emitOpError("scan output must be a ranked tensor at #")
               << i;
      SmallVector<Value> dynSizes;
      for (int64_t dim = 0; dim < resultType.getRank(); ++dim)
        if (resultType.isDynamicDim(dim))
          dynSizes.push_back(zero);
      zeroResults.push_back(bufferization::AllocTensorOp::create(
          builder, loc, resultType, dynSizes));
    }
    for (auto [oldResult, newResult] :
         llvm::zip(scanOp->getResults(), zeroResults)) {
      if (failed(replaceResult(oldResult, newResult)))
        return failure();
    }
    scanOp->erase();
    return success();
  }

  llvm::SetVector<Value> capturedSet;
  getUsedValuesDefinedAbove(bodyRegion, capturedSet);
  SmallVector<Value> captures(capturedSet.begin(), capturedSet.end());
  SmallVector<Value> runtimeCaptures;
  runtimeCaptures.reserve(captures.size());
  OpBuilder captureBuilder(scanOp);
  for (Value capture : captures) {
    auto type = dyn_cast<RankedTensorType>(capture.getType());
    Operation *producer = capture.getDefiningOp();
    if (!type || type.getRank() != 0 || !producer ||
        producer->getBlock() != scanOp->getBlock()) {
      runtimeCaptures.push_back(capture);
      continue;
    }
    OpBuilder::InsertionGuard guard(captureBuilder);
    captureBuilder.setInsertionPointAfter(producer);
    runtimeCaptures.push_back(bufferization::AllocTensorOp::create(
        captureBuilder, loc, type, ValueRange{}, capture));
  }

  auto parentFn = scanOp->getParentOfType<func::FuncOp>();
  if (!parentFn || parentFn.getNumArguments() == 0 ||
      !isa<ContextType>(parentFn.getArgument(0).getType()))
    return scanOp->emitOpError("expected parent func arg 0 to be !hip.context");
  Value context = parentFn.getArgument(0);

  SmallVector<Type> bodyArgTypes{ContextType::get(ctx)};
  for (BlockArgument arg : bodyBlock.getArguments())
    bodyArgTypes.push_back(arg.getType());
  for (Value capture : captures)
    bodyArgTypes.push_back(capture.getType());
  SmallVector<Type> bodyResultTypes(yieldOp->getOperandTypes().begin(),
                                    yieldOp->getOperandTypes().end());

  std::string bodyName;
  do {
    unsigned bodyId = counter++;
    bodyName = (parentFn.getName() + "_scan_body_n" + Twine(bodyId)).str();
  } while (module.lookupSymbol(bodyName));

  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto createBodyFunction = [&](StringRef name) {
    auto function = func::FuncOp::create(
        moduleBuilder, loc, name,
        FunctionType::get(ctx, bodyArgTypes, bodyResultTypes));
    function.setPrivate();
    Block *entry = function.addEntryBlock();
    IRMapping mapping;
    unsigned argIndex = 1;
    for (BlockArgument arg : bodyBlock.getArguments())
      mapping.map(arg, entry->getArgument(argIndex++));
    for (Value capture : captures)
      mapping.map(capture, entry->getArgument(argIndex++));
    OpBuilder builder(entry, entry->end());
    for (Operation &bodyOp : bodyBlock)
      if (&bodyOp != yieldOp)
        builder.clone(bodyOp, mapping);
    SmallVector<Value> functionReturns;
    for (Value value : yieldOp->getOperands())
      functionReturns.push_back(mapping.lookupOrDefault(value));
    func::ReturnOp::create(builder, loc, functionReturns);
    return function;
  };
  auto bodyFn = createBodyFunction(bodyName);
  bodyFn->setAttr("hip.scan_num_state_variables",
                  moduleBuilder.getI32IntegerAttr(numState));

  // Execute iteration zero exactly once in the non-empty branch. Besides
  // being the real first iteration, its results provide any data-dependent
  // trailing dimensions needed by stacked scan outputs. The tail hip.scan
  // starts at input element one, so shape discovery never adds an extra body
  // execution or advances random/stateful body operations twice.
  OpBuilder outerBuilder(scanOp);
  Value tripCount =
      tensor::DimOp::create(outerBuilder, loc, scanInputs.front(), 0);
  for (unsigned i = 1; i < numScanInputs; ++i) {
    Value otherCount =
        tensor::DimOp::create(outerBuilder, loc, scanInputs[i], 0);
    Value equal = arith::CmpIOp::create(
        outerBuilder, loc, arith::CmpIPredicate::eq, tripCount, otherCount);
    cf::AssertOp::create(outerBuilder, loc, equal,
                         "Scan input sequence lengths must match");
  }

  Value zero = arith::ConstantIndexOp::create(outerBuilder, loc, 0);
  Value nonEmpty = arith::CmpIOp::create(
      outerBuilder, loc, arith::CmpIPredicate::ugt, tripCount, zero);
  SmallVector<Value> emptyBodyOutputs;
  for (unsigned i = 0; i < numScanOutputs; ++i) {
    auto bodyOutputType =
        dyn_cast<RankedTensorType>(bodyResultTypes[numState + i]);
    if (!bodyOutputType)
      return scanOp->emitOpError(
                 "body scan output must be a ranked tensor at #")
             << i;
    SmallVector<Value> dynSizes;
    for (int64_t dim = 0; dim < bodyOutputType.getRank(); ++dim)
      if (bodyOutputType.isDynamicDim(dim))
        dynSizes.push_back(zero);
    emptyBodyOutputs.push_back(
        tensor::EmptyOp::create(outerBuilder, loc, bodyOutputType.getShape(),
                                bodyOutputType.getElementType(), dynSizes));
  }
  auto ifOp = scf::IfOp::create(outerBuilder, loc, bodyResultTypes, nonEmpty,
                                /*withElseRegion=*/true);

  outerBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  IRMapping firstMapping;
  for (unsigned i = 0; i < numState; ++i)
    firstMapping.map(bodyBlock.getArgument(i), stateInitials[i]);
  for (unsigned i = 0; i < numScanInputs; ++i) {
    FailureOr<Value> slice = makeFirstSlice(
        outerBuilder, loc, scanInputs[i],
        bodyBlock.getArgument(numState + i).getType(), (*inputAxes)[i]);
    if (failed(slice))
      return scanOp->emitOpError("could not form rank-reduced scan slice #")
             << i;
    firstMapping.map(bodyBlock.getArgument(numState + i), *slice);
  }
  for (auto [capture, runtimeCapture] : llvm::zip(captures, runtimeCaptures))
    firstMapping.map(capture, runtimeCapture);
  for (Operation &bodyOp : bodyBlock)
    if (&bodyOp != yieldOp)
      outerBuilder.clone(bodyOp, firstMapping);
  SmallVector<Value> firstResults;
  for (Value value : yieldOp->getOperands())
    firstResults.push_back(firstMapping.lookupOrDefault(value));
  scf::YieldOp::create(outerBuilder, loc, firstResults);

  outerBuilder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  SmallVector<Value> emptyFirstResults(stateInitials.begin(),
                                       stateInitials.end());
  emptyFirstResults.append(emptyBodyOutputs.begin(), emptyBodyOutputs.end());
  scf::YieldOp::create(outerBuilder, loc, emptyFirstResults);

  outerBuilder.setInsertionPointAfter(ifOp);
  SmallVector<Value> firstYields(ifOp.getResults().begin(),
                                 ifOp.getResults().end());
  Value one = arith::ConstantIndexOp::create(outerBuilder, loc, 1);
  Value nonEmptyTailCount =
      arith::SubIOp::create(outerBuilder, loc, tripCount, one);
  Value tailCount = arith::SelectOp::create(outerBuilder, loc, nonEmpty,
                                            nonEmptyTailCount, zero);
  Value tailOffset =
      arith::SelectOp::create(outerBuilder, loc, nonEmpty, one, zero);
  SmallVector<Value> tailInputs;
  for (Value input : scanInputs) {
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> tailShape(inputType.getShape());
    tailShape[0] = ShapedType::kDynamic;
    auto tailType =
        RankedTensorType::get(tailShape, inputType.getElementType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
      offsets.push_back(dim == 0 ? OpFoldResult(tailOffset)
                                 : OpFoldResult(outerBuilder.getIndexAttr(0)));
      strides.push_back(outerBuilder.getIndexAttr(1));
      if (dim == 0)
        sizes.push_back(tailCount);
      else if (inputType.isDynamicDim(dim))
        sizes.push_back(
            tensor::DimOp::create(outerBuilder, loc, input, dim).getResult());
      else
        sizes.push_back(outerBuilder.getIndexAttr(inputType.getDimSize(dim)));
    }
    tailInputs.push_back(tensor::ExtractSliceOp::create(outerBuilder, loc,
                                                        tailType, input,
                                                        offsets, sizes, strides)
                             .getResult());
  }

  SmallVector<Value> tailInits(firstYields.begin(),
                               firstYields.begin() + numState);
  SmallVector<RankedTensorType> bodyOutputTypes;
  for (unsigned i = 0; i < numScanOutputs; ++i) {
    Value firstOutput = firstYields[numState + i];
    auto bodyOutputType = dyn_cast<RankedTensorType>(firstOutput.getType());
    Type resultType = scanOp->getResult(numState + i).getType();
    auto rankedResultType = dyn_cast<RankedTensorType>(resultType);
    if (!bodyOutputType ||
        (!isa<NoneType>(resultType) &&
         (!rankedResultType ||
          rankedResultType.getRank() != bodyOutputType.getRank() + 1)))
      return scanOp->emitOpError("scan output/body output rank mismatch at #")
             << i;
    bodyOutputTypes.push_back(bodyOutputType);

    SmallVector<int64_t> tailShape{ShapedType::kDynamic};
    tailShape.append(bodyOutputType.getShape().begin(),
                     bodyOutputType.getShape().end());
    auto tailType =
        RankedTensorType::get(tailShape, bodyOutputType.getElementType());
    SmallVector<Value> dynSizes;
    dynSizes.push_back(tailCount);
    for (int64_t dim = 0; dim < bodyOutputType.getRank(); ++dim)
      if (bodyOutputType.isDynamicDim(dim))
        dynSizes.push_back(
            tensor::DimOp::create(outerBuilder, loc, firstOutput, dim));
    tailInits.push_back(
        tensor::EmptyOp::create(outerBuilder, loc, tailType.getShape(),
                                tailType.getElementType(), dynSizes));
  }

  Value hasTail = arith::CmpIOp::create(
      outerBuilder, loc, arith::CmpIPredicate::ugt, tailCount, zero);
  SmallVector<Type> tailResultTypes;
  for (Value value : tailInits)
    tailResultTypes.push_back(value.getType());
  auto tailIf = scf::IfOp::create(outerBuilder, loc, tailResultTypes, hasTail,
                                  /*withElseRegion=*/true);

  outerBuilder.setInsertionPointToStart(&tailIf.getThenRegion().front());
  OperationState tailState(loc, "hip.scan");
  tailState.addOperands(context);
  tailState.addOperands(tailInputs);
  tailState.addOperands(runtimeCaptures);
  tailState.addOperands(tailInits);
  tailState.addTypes(llvm::to_vector(
      llvm::map_range(tailInits, [](Value value) { return value.getType(); })));
  tailState.addAttribute("body_func", FlatSymbolRefAttr::get(ctx, bodyName));
  tailState.addAttribute("num_state_variables",
                         outerBuilder.getI32IntegerAttr(numState));
  tailState.addAttribute("num_scan_inputs",
                         outerBuilder.getI32IntegerAttr(numScanInputs));
  tailState.addAttribute("num_scan_outputs",
                         outerBuilder.getI32IntegerAttr(numScanOutputs));
  tailState.addAttribute("operand_segment_sizes",
                         outerBuilder.getDenseI32ArrayAttr(
                             {1, static_cast<int32_t>(numScanInputs),
                              static_cast<int32_t>(runtimeCaptures.size()), 0,
                              static_cast<int32_t>(tailInits.size())}));
  Operation *tailScan = outerBuilder.create(tailState);
  scf::YieldOp::create(outerBuilder, loc, tailScan->getResults());

  outerBuilder.setInsertionPointToStart(&tailIf.getElseRegion().front());
  scf::YieldOp::create(outerBuilder, loc, tailInits);
  outerBuilder.setInsertionPointAfter(tailIf);

  for (unsigned i = 0; i < numState; ++i) {
    if (failed(replaceResult(scanOp->getResult(i), tailIf.getResult(i))))
      return failure();
  }
  for (unsigned i = 0; i < numScanOutputs; ++i) {
    OpResult oldResult = scanOp->getResult(numState + i);
    if (isa<NoneType>(oldResult.getType()))
      continue;
    Value firstOutput = firstYields[numState + i];
    Value tailOutput = tailIf.getResult(numState + i);
    RankedTensorType bodyType = bodyOutputTypes[i];
    auto resultType = cast<RankedTensorType>(oldResult.getType());

    SmallVector<Value> dynSizes;
    for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
      if (!resultType.isDynamicDim(dim))
        continue;
      dynSizes.push_back(dim == 0 ? tripCount
                                  : tensor::DimOp::create(outerBuilder, loc,
                                                          firstOutput, dim - 1)
                                        .getResult());
    }
    Value stacked =
        tensor::EmptyOp::create(outerBuilder, loc, resultType.getShape(),
                                resultType.getElementType(), dynSizes);

    SmallVector<OpFoldResult> firstOffsets(resultType.getRank(),
                                           outerBuilder.getIndexAttr(0));
    SmallVector<OpFoldResult> firstSizes{outerBuilder.getIndexAttr(1)};
    SmallVector<OpFoldResult> firstStrides(resultType.getRank(),
                                           outerBuilder.getIndexAttr(1));
    for (int64_t dim = 0; dim < bodyType.getRank(); ++dim) {
      if (bodyType.isDynamicDim(dim))
        firstSizes.push_back(
            tensor::DimOp::create(outerBuilder, loc, firstOutput, dim)
                .getResult());
      else
        firstSizes.push_back(
            outerBuilder.getIndexAttr(bodyType.getDimSize(dim)));
    }
    auto insertFirst =
        scf::IfOp::create(outerBuilder, loc, TypeRange{resultType}, nonEmpty,
                          /*withElseRegion=*/true);
    outerBuilder.setInsertionPointToStart(&insertFirst.getThenRegion().front());
    Value withFirst =
        tensor::InsertSliceOp::create(outerBuilder, loc, firstOutput, stacked,
                                      firstOffsets, firstSizes, firstStrides)
            .getResult();
    scf::YieldOp::create(outerBuilder, loc, withFirst);
    outerBuilder.setInsertionPointToStart(&insertFirst.getElseRegion().front());
    scf::YieldOp::create(outerBuilder, loc, stacked);
    outerBuilder.setInsertionPointAfter(insertFirst);
    stacked = insertFirst.getResult(0);

    SmallVector<OpFoldResult> tailOffsets(resultType.getRank(),
                                          outerBuilder.getIndexAttr(0));
    tailOffsets[0] = tailOffset;
    SmallVector<OpFoldResult> tailSizes{tailCount};
    for (int64_t dim = 0; dim < bodyType.getRank(); ++dim)
      tailSizes.push_back(firstSizes[dim + 1]);
    stacked =
        tensor::InsertSliceOp::create(outerBuilder, loc, tailOutput, stacked,
                                      tailOffsets, tailSizes, firstStrides)
            .getResult();
    if (failed(replaceResult(oldResult, stacked)))
      return failure();
  }
  scanOp->erase();
  return success();
}

void OnnxScanOutlinePass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Operation *> scans;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "onnx.Scan")
      scans.push_back(op);
  });
  unsigned counter = 0;
  for (Operation *scan : scans)
    if (failed(outlineScan(scan, counter)))
      return signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> createOnnxScanOutlinePass() {
  return std::make_unique<OnnxScanOutlinePass>();
}

} // namespace hip
} // namespace mlir
